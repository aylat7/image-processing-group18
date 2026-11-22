"""CPU timing benchmark for both models (Phase 4, Roneet + Ayla).

Measures the cost of ONE image at a time (batch size 1, like the live app), split into
preprocessing (common.py) and model inference. Images are loaded into memory first, so
disk reads are not counted. The first --warmup images per model are run but not timed.

Usage (from the project root, with the venv active):
    python app/benchmark.py                          # 300 images from data/splits/test.csv
    python app/benchmark.py --threads 1 --tag 1thread
    python app/benchmark.py --source webcam_test     # our own held-out webcam crops
    python app/benchmark.py --source synthetic       # no dataset needed (random 300x300 images)

Output:
    printed table (mean / median / p95 ms, images per second)
    results/timing/cpu_timing_<tag>.json      summary + machine info, for 4_compare.ipynb
    results/timing/cpu_timing_<tag>.csv       one row per timed image per model
"""

import argparse
import csv
import json
import platform
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402
from predictors import load_available  # noqa: E402


def load_images(source: str, n: int, seed: int):
    rng = random.Random(seed)
    if source == "synthetic":
        g = np.random.default_rng(seed)
        return [(f"synthetic_{i}", g.integers(0, 256, (300, 300, 3), dtype=np.uint8)) for i in range(n)]
    if source == "test":
        paths = list(common.load_split("test")["abs_path"])
    else:  # webcam_test: timing only, no labels are used, so this does not touch training/tuning
        paths = sorted(str(p) for p in common.WEBCAM_TEST_DIR.glob("*/*.jpg"))
    if not paths:
        sys.exit(f"No images found for --source {source}.")
    paths = rng.sample(paths, min(n, len(paths)))
    try:
        return [(p, common.load_image(p)) for p in paths]
    except FileNotFoundError as e:
        sys.exit(f"{e}\nIs the dataset in data/raw/asl_raw/? (see data/README.md) Or use --source synthetic.")


def stats(xs):
    a = np.asarray(xs)
    return dict(mean=float(a.mean()), median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                std=float(a.std()), min=float(a.min()), max=float(a.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["test", "webcam_test", "synthetic"], default="test")
    ap.add_argument("--n", type=int, default=300, help="number of timed images")
    ap.add_argument("--warmup", type=int, default=20, help="untimed images run first")
    ap.add_argument("--threads", type=int, default=None, help="PyTorch CPU threads (default: PyTorch's choice)")
    ap.add_argument("--tag", default="local", help="name for the output files, e.g. ayla_macbook or 1thread")
    args = ap.parse_args()

    models = load_available(threads=args.threads)
    if not models:
        sys.exit("No models found in models/.")
    images = load_images(args.source, args.n + args.warmup, common.SEED)
    warm, timed = images[:args.warmup], images[args.warmup:]
    print(f"\nTiming {len(timed)} images from '{args.source}' (+{len(warm)} warm-up), batch size 1, CPU\n")

    import torch
    summary = dict(
        date=datetime.now().isoformat(timespec="seconds"), source=args.source, n_images=len(timed),
        warmup=len(warm), batch_size=1,
        machine=dict(system=platform.system(), release=platform.release(), machine=platform.machine(),
                     processor=platform.processor(), python=platform.python_version(),
                     torch=torch.__version__, torch_threads=torch.get_num_threads()),
        models={},
    )
    rows = []
    print(f"{'model':10s} {'part':10s} {'mean':>7s} {'median':>7s} {'p95':>7s}   (ms per image)")
    for m in models.values():
        for _, img in warm:
            m.predict(img)
        pre, inf, tot = [], [], []
        for path, img in timed:
            p = m.predict(img)
            pre.append(p.preprocess_ms); inf.append(p.inference_ms); tot.append(p.total_ms)
            rows.append(dict(model=m.key, path=path, preprocess_ms=f"{p.preprocess_ms:.3f}",
                             inference_ms=f"{p.inference_ms:.3f}", total_ms=f"{p.total_ms:.3f}"))
        s = dict(preprocess_ms=stats(pre), inference_ms=stats(inf), total_ms=stats(tot))
        s["images_per_second"] = 1000.0 / s["total_ms"]["mean"]
        summary["models"][m.key] = dict(name=m.name, **s)
        for part in ("preprocess_ms", "inference_ms", "total_ms"):
            st = s[part]
            print(f"{m.name:10s} {part[:-3]:10s} {st['mean']:7.2f} {st['median']:7.2f} {st['p95']:7.2f}")
        print(f"{'':10s} -> {s['images_per_second']:.0f} images/s\n")

    out = common.RESULTS_DIR / "timing"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"cpu_timing_{args.tag}.json").write_text(json.dumps(summary, indent=1))
    with open(out / f"cpu_timing_{args.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out.relative_to(common.PROJECT_ROOT)}/cpu_timing_{args.tag}.json and .csv")


if __name__ == "__main__":
    main()
