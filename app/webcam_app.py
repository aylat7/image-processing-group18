"""Live ASL letter recognition from the webcam (Phases 3–4, owner: Ayla).

Per frame, using the SAME functions as training (project rule 1):
    frame -> common.orient_frame -> common.crop_box -> HOG + SVM  (common.hog_features)
                                                    -> ResNet18   (common.preprocess_cnn)
Shows the letter, its confidence, the top 3, and the CPU time per frame. Models can be
switched live, or both run side by side.

Usage (from the project root, with the venv active):
    python app/webcam_app.py                     # both models if both files exist
    python app/webcam_app.py --model cnn         # start with ResNet18 only
    python app/webcam_app.py --camera 1          # if webcam 0 is the wrong camera
    python app/webcam_app.py --image path.jpg    # no camera: predict one image and print it

Needs models/hog_svm.joblib and/or models/resnet18.pt (from the shared Drive).

Keys:
    1 / 2      HOG + SVM / ResNet18
    B          both side by side
    TAB        cycle HOG -> CNN -> both
    P          pause (freeze the frame)
    S          screenshot to results/figures/demo_<model>_<time>.png
    ESC / Q    quit

Notes:
  * The displayed letter is averaged over the last --smooth frames to stop flicker.
    The ms/frame numbers are NOT smoothed: they are the real cost of one frame
    (preprocessing + model), averaged over the last 30 frames.
  * For the timing numbers in the report, use app/benchmark.py instead (fixed images,
    no camera or drawing overhead).
"""

import argparse
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402
from predictors import load_available  # noqa: E402

WINDOW = "ASL recognition - group 18"
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, ORANGE, WHITE, GRAY, BLACK = (80, 220, 80), (0, 165, 255), (255, 255, 255), (160, 160, 160), (0, 0, 0)
MODES = ["hog", "cnn", "both"]


class Track:
    """Per-model rolling state: recent probabilities (for display smoothing) and timings."""

    def __init__(self, model, smooth: int):
        self.model = model
        self.probs = deque(maxlen=max(1, smooth))
        self.ms = deque(maxlen=30)
        self.pre_ms = deque(maxlen=30)
        self.last = None

    def update(self, crop):
        raw = self.model.predict(crop)
        self.probs.append(raw.probs)
        self.ms.append(raw.total_ms)
        self.pre_ms.append(raw.preprocess_ms)
        self.last = self.model.make_prediction(np.mean(self.probs, axis=0))
        return self.last

    def reset(self):
        self.probs.clear()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def text(img, s, org, scale=0.55, color=WHITE, thick=1):
    cv2.putText(img, s, org, FONT, scale, BLACK, thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def shade(img, x1, y1, x2, y2, alpha=0.55):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    roi = img[y1:y2, x1:x2]
    roi[:] = (roi * (1 - alpha)).astype(np.uint8)


def draw_panel(img, track: Track, x, y, min_conf):
    """One model's result block, 370 x 165 px, top-left corner at (x, y)."""
    W, H = 370, 165
    shade(img, x, y, x + W, y + H)
    m, p = track.model, track.last
    title = m.name + ("" if m.calibrated else "  [score]")
    text(img, title, (x + 10, y + 22), 0.6, WHITE, 1)

    sure = p.confidence >= min_conf or not m.calibrated
    color = (GREEN if sure else ORANGE) if m.calibrated else WHITE
    text(img, p.letter if sure else p.letter + "?", (x + 10, y + 82), 2.0, color, 4)
    label = f"{100 * p.confidence:.0f}%" if m.calibrated else f"{p.confidence:.2f}"
    text(img, label, (x + 110, y + 62), 0.8, color, 2)

    # confidence bar
    bx, by, bw = x + 110, y + 72, 165
    cv2.rectangle(img, (bx, by), (bx + bw, by + 10), GRAY, 1)
    cv2.rectangle(img, (bx, by), (bx + int(bw * min(1.0, p.confidence)), by + 10), color, -1)

    top3 = "   ".join(f"{c} {100 * v:.0f}%" if m.calibrated else f"{c} {v:.2f}" for c, v in p.top3)
    text(img, "top 3:  " + top3, (x + 10, y + 115), 0.5, WHITE)
    ms, pre = float(np.mean(track.ms)), float(np.mean(track.pre_ms))
    text(img, f"{ms:.1f} ms/frame  (prep {pre:.1f} + model {ms - pre:.1f})", (x + 10, y + 145), 0.5, WHITE)


def model_input_view(crop, mode, size=120):
    """Small thumbnail of exactly what the model receives, so we can check the crop."""
    if mode == "hog":
        g = (common.preprocess_hog(crop) * 255).astype(np.uint8)
        v = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    else:
        v = cv2.cvtColor(common.prepare_rgb(crop), cv2.COLOR_RGB2BGR)
    return cv2.resize(v, (size, size), interpolation=cv2.INTER_NEAREST)


def draw(frame, crop, tracks, mode, fps, min_conf, paused, status):
    view = frame.copy()
    x1, y1, x2, y2 = common.box_coords(frame.shape)
    cv2.rectangle(view, (x1, y1), (x2, y2), GREEN, 2)
    text(view, "hand here", (x1 + 6, y1 + 22), 0.6, GREEN)

    shown = [k for k in (("hog", "cnn") if mode == "both" else (mode,)) if k in tracks]
    for i, k in enumerate(shown):
        draw_panel(view, tracks[k], 10, 10 + i * 175, min_conf)

    # model input thumbnail(s), bottom-left
    h = view.shape[0]
    for i, k in enumerate(shown):
        thumb = model_input_view(crop, k)
        tx, ty = 10 + i * 130, h - 160
        if ty > 0:
            view[ty:ty + 120, tx:tx + 120] = thumb
            text(view, f"{k} input", (tx, ty - 6), 0.45, WHITE)

    keys = "1 HOG  2 CNN  B both  TAB cycle  P pause  S screenshot  ESC quit"
    text(view, f"{fps:4.1f} FPS" + ("   PAUSED" if paused else "") + (f"   {status}" if status else ""),
         (10, h - 30), 0.5, ORANGE if paused else WHITE)
    text(view, keys, (10, h - 10), 0.45, GRAY)
    return view


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def predict_image(path, models):
    img = common.load_image(path)
    print(f"{path}  ({img.shape[1]}x{img.shape[0]}, used as-is: it should already be a hand crop)")
    for m in models.values():
        p = m.predict(img)
        fmt = (lambda v: f"{100 * v:.1f}%") if m.calibrated else (lambda v: f"{v:.3f}")
        top = ", ".join(f"{c} {fmt(v)}" for c, v in p.top3)
        print(f"  {m.name:10s} -> {p.letter}  ({fmt(p.confidence)})   top 3: {top}   {p.total_ms:.1f} ms")


def run_webcam(args, models):
    mode = args.model if args.model in models or args.model == "both" else next(iter(models))
    if mode == "both" and len(models) < 2:
        mode = next(iter(models))
    tracks = {k: Track(m, args.smooth) for k, m in models.items()}

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open webcam {args.camera}. Try --camera 1. "
                 "(macOS: allow camera access for your terminal / VS Code in System Settings > Privacy.)")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    frame = crop = None
    paused, status, status_until = False, "", 0.0
    fps, last_t = 0.0, time.perf_counter()

    while True:
        if not paused or frame is None:
            ok, raw = cap.read()
            if not ok:
                print("Camera read failed.")
                break
            frame = common.orient_frame(raw)
            crop = common.crop_box(frame)
            active = ("hog", "cnn") if mode == "both" else (mode,)
            for k in active:
                if k in tracks:
                    tracks[k].update(crop)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6)) if fps else 1.0 / max(now - last_t, 1e-6)
        last_t = now
        if status and now > status_until:
            status = ""

        view = draw(frame, crop, tracks, mode, fps, args.min_conf, paused, status)
        cv2.imshow(WINDOW, view)

        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            continue
        new_mode = None
        if key in (27, ord("q"), ord("Q")):
            break
        elif key == ord("1"):
            new_mode = "hog"
        elif key == ord("2"):
            new_mode = "cnn"
        elif key in (ord("b"), ord("B")):
            new_mode = "both"
        elif key == 9:                                          # TAB
            new_mode = MODES[(MODES.index(mode) + 1) % len(MODES)]
            while new_mode != "both" and new_mode not in models:
                new_mode = MODES[(MODES.index(new_mode) + 1) % len(MODES)]
        elif key in (ord("p"), ord("P"), 32):
            paused = not paused
        elif key in (ord("s"), ord("S")):
            out = common.RESULTS_DIR / "figures" / f"demo_{mode}_{datetime.now():%Y%m%d_%H%M%S}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), view)
            status, status_until = f"saved {out.name}", now + 2.5

        if new_mode:
            missing = [k for k in (("hog", "cnn") if new_mode == "both" else (new_mode,)) if k not in models]
            if missing:
                status, status_until = f"no model file for {', '.join(missing)}", now + 2.5
            else:
                mode = new_mode
                for t in tracks.values():
                    t.reset()
                if paused:                                       # recompute for the frozen frame
                    for k in (("hog", "cnn") if mode == "both" else (mode,)):
                        tracks[k].update(crop)

    cap.release()
    cv2.destroyAllWindows()
    for t in tracks.values():
        if t.ms:
            print(f"{t.model.name}: {np.mean(t.ms):.1f} ms/frame over the last {len(t.ms)} frames")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODES, default="both", help="which model to start with")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (try 1 if 0 doesn't work)")
    ap.add_argument("--smooth", type=int, default=5, help="average the display over this many frames (1 = off)")
    ap.add_argument("--min-conf", type=float, default=0.5, help="below this, the letter is shown orange with '?'")
    ap.add_argument("--threads", type=int, default=None, help="PyTorch CPU threads (default: PyTorch's choice)")
    ap.add_argument("--image", type=Path, help="predict one image file instead of opening the webcam")
    args = ap.parse_args()

    models = load_available(threads=args.threads)
    if not models:
        sys.exit("No models found. Put hog_svm.joblib and/or resnet18.pt from the shared Drive into models/.")

    if args.image:
        predict_image(args.image, models)
    else:
        run_webcam(args, models)


if __name__ == "__main__":
    main()
