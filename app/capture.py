"""Collect and label our own webcam images.

Shows the webcam with the SAME fixed box the live app uses (common.crop_box), and saves
the cropped box with its label and recording conditions.

Usage (from the project root, with the venv active):
    python app/capture.py --split train --participant ayla --lighting normal --background plain
    python app/capture.py --split test  --participant ryan --lighting dim    --background cluttered

Keys:
    A–Z        choose the letter you are about to sign
    SPACE      train: save one image
               test:  record a 2-second clip of the box and save its middle frame as the trial image
    ENTER      train only: start/stop burst mode (saves an image every --burst-interval seconds)
    BACKSPACE  delete the last saved image (and its clip)
    ESC        quit

Output:
    data/webcam_<split>/<LETTER>/<session>_<time>.jpg
    data/webcam_test/clips/<LETTER>/<session>_<time>.mp4      (test only)
    data/webcam_<split>/metadata.csv                          (one row per saved image)

Rule: data/webcam_test is NEVER used for training or tuning.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

FIELDS = ["path", "label", "split", "participant", "lighting", "background",
          "session", "timestamp", "clip"]
CLIP_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Saving (no GUI, so these can be tested without a camera)
# ---------------------------------------------------------------------------
class Saver:
    def __init__(self, split, participant, lighting, background, data_dir=common.DATA_DIR):
        self.split = split
        self.root = Path(data_dir) / f"webcam_{split}"
        self.meta_path = self.root / "metadata.csv"
        self.info = dict(split=split, participant=participant.lower(),
                         lighting=lighting, background=background)
        # One session = one person in one condition on one day. Used later to keep
        # a session's images together when splitting webcam_train.
        self.session = f"{participant.lower()}_{lighting}_{background}_{datetime.now():%Y%m%d}"
        self.history = []   # rows saved during this run, for undo
        self.counts = {}    # letter -> images saved during this run

    def _name(self):
        return f"{self.session}_{datetime.now():%H%M%S-%f}"[:-3]

    def save(self, letter, crop, clip_frames=None):
        """Save one crop (and optionally a clip). Returns the metadata row."""
        name = self._name()
        img_path = self.root / letter / f"{name}.jpg"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_path), crop)

        clip_rel = ""
        if clip_frames:
            clip_path = self.root / "clips" / letter / f"{name}.mp4"
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            h, w = clip_frames[0].shape[:2]
            fps = max(1.0, len(clip_frames) / CLIP_SECONDS)
            vw = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for f in clip_frames:
                vw.write(f)
            vw.release()
            clip_rel = clip_path.relative_to(self.root.parent).as_posix()

        row = dict(path=img_path.relative_to(self.root.parent).as_posix(), label=letter,
                   session=self.session, timestamp=datetime.now().isoformat(timespec="seconds"),
                   clip=clip_rel, **self.info)
        new_file = not self.meta_path.exists()
        with open(self.meta_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(row)
        self.history.append(row)
        self.counts[letter] = self.counts.get(letter, 0) + 1
        return row

    def undo(self):
        """Delete the last image saved in this run (file, clip and metadata row)."""
        if not self.history:
            return None
        row = self.history.pop()
        base = self.root.parent
        (base / row["path"]).unlink(missing_ok=True)
        if row["clip"]:
            (base / row["clip"]).unlink(missing_ok=True)
        with open(self.meta_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["path"] != row["path"]]
        with open(self.meta_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        self.counts[row["label"]] -= 1
        return row


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def draw_overlay(frame, letter, saver, status, recording):
    view = frame.copy()
    x1, y1, x2, y2 = common.box_coords(frame.shape)
    color = (0, 0, 255) if recording else (0, 255, 0)
    cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)

    lines = [
        f"{saver.split.upper()} | {saver.session}",
        f"Letter: {letter or '-- press A-Z --'}   saved this run: {saver.counts.get(letter, 0) if letter else 0}",
        "SPACE " + ("record trial" if saver.split == "test" else "save") +
        ("   ENTER burst" if saver.split == "train" else "") + "   BACKSPACE undo   ESC quit",
        status,
    ]
    for i, text in enumerate(lines):
        y = 25 + i * 24
        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return view


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--participant", required=True, help="first name, e.g. ayla")
    ap.add_argument("--lighting", choices=["normal", "dim"], required=True)
    ap.add_argument("--background", choices=["plain", "cluttered"], required=True)
    ap.add_argument("--camera", type=int, default=0, help="webcam index (try 1 if 0 doesn't work)")
    ap.add_argument("--burst-interval", type=float, default=0.25, help="seconds between burst images")
    args = ap.parse_args()

    saver = Saver(args.split, args.participant, args.lighting, args.background)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open webcam {args.camera}. Try --camera 1.")

    window = "ASL capture"
    letter, status = None, "Put your hand inside the box so it fills it."
    burst, last_burst = False, 0.0
    clip, clip_start = None, 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            status = "Camera read failed"
            break
        frame = common.orient_frame(frame)
        crop = common.crop_box(frame)
        now = time.time()

        # test mode: collecting a clip
        if clip is not None:
            clip.append(crop)
            if now - clip_start >= CLIP_SECONDS:
                row = saver.save(letter, clip[len(clip) // 2], clip_frames=clip)
                status = f"Saved trial {Path(row['path']).name} ({len(clip)} frames)"
                clip = None

        # train mode: burst
        if burst and now - last_burst >= args.burst_interval:
            saver.save(letter, crop)
            last_burst = now
            status = f"Burst: {saver.counts[letter]} {letter} images this run (ENTER to stop)"

        cv2.imshow(window, draw_overlay(frame, letter, saver, status, clip is not None or burst))
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            continue
        if key == 27:                                   # ESC
            break
        if clip is not None:                            # ignore keys while recording
            continue
        if ord("a") <= key <= ord("z") or ord("A") <= key <= ord("Z"):
            letter, burst = chr(key).upper(), False
            status = f"Letter {letter} selected"
        elif key == 32:                                 # SPACE
            if not letter:
                status = "Choose a letter first (press A-Z)"
            elif args.split == "test":
                clip, clip_start = [], now
                status = "Recording... hold the pose"
            else:
                row = saver.save(letter, crop)
                status = f"Saved {Path(row['path']).name}"
        elif key == 13 and args.split == "train":       # ENTER
            if not letter:
                status = "Choose a letter first (press A-Z)"
            else:
                burst = not burst
                status = "Burst ON" if burst else "Burst OFF"
        elif key in (8, 127):                           # BACKSPACE
            row = saver.undo()
            status = f"Deleted {Path(row['path']).name}" if row else "Nothing to undo"

    cap.release()
    cv2.destroyAllWindows()
    total = sum(saver.counts.values())
    print(f"Saved {total} images this run -> {saver.root}")
    for k in sorted(saver.counts):
        print(f"  {k}: {saver.counts[k]}")


if __name__ == "__main__":
    main()
