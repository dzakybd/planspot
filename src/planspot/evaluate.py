"""One-shot spotting benchmark on FloorPlanCAD.

Episode = (drawing tile, symbol class with >= 2 instances). One instance is the exemplar; the
task is to find the rest. Detections are matched to ground truth greedily by score at IoU >= 0.5.

Splits are by source drawing (the part of the file name before '-'), never by tile, so tiles
of one drawing never land in both the threshold-tuning (val) and the reported (test) split.
"""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from planspot.data import NON_SYMBOL, build_annotations, load_image
from planspot.spotter import DinoSpotter, NCCSpotter, box_iou

SIZE_BUCKETS = [
    (0, 48, "small (<48 px)"),
    (48, 96, "medium (48-96 px)"),
    (96, 10_000, "large (>=96 px)"),
]


def split_of(file: str) -> str:
    drawing = Path(file).stem.split("-")[0]
    return (
        "val" if int(hashlib.md5(drawing.encode()).hexdigest(), 16) % 5 == 0 else "test"
    )


def episodes(rec: dict, max_per_image: int):
    by_cls = defaultdict(list)
    for o in rec["objects"]:
        if o["label"] not in NON_SYMBOL:
            by_cls[o["label"]].append(o["box"])
    eps = []
    for label, boxes in sorted(by_cls.items()):
        if len(boxes) >= 2:
            boxes = sorted(boxes, key=lambda b: (round(b[1]), round(b[0])))
            eps.append((label, boxes[0], boxes[1:]))
    return eps[:max_per_image]


def run_episode(spotter, img, label, exemplar, gts, min_score):
    dets = spotter.detect(img, exemplar, min_score=min_score)
    dets = [
        d for d in dets if box_iou(d.box, exemplar) < 0.5
    ]  # finding the exemplar itself is free
    used = [False] * len(gts)
    scored = []
    for d in sorted(dets, key=lambda d: -d.score):
        best, bi = 0.5, -1
        for i, g in enumerate(gts):
            if not used[i]:
                iou = box_iou(d.box, g)
                if iou >= best:
                    best, bi = iou, i
        if bi >= 0:
            used[bi] = True
        scored.append((d.score, bi >= 0))
    size = max(exemplar[2] - exemplar[0], exemplar[3] - exemplar[1])
    return {"label": label, "size": size, "n_gt": len(gts), "scored": scored}


def prf(eps, thr):
    tp = sum(sum(1 for s, ok in e["scored"] if s >= thr and ok) for e in eps)
    npred = sum(sum(1 for s, _ in e["scored"] if s >= thr) for e in eps)
    ngt = sum(e["n_gt"] for e in eps)
    p = tp / npred if npred else 0.0
    r = tp / ngt if ngt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    count_err = [
        abs(sum(1 for s, _ in e["scored"] if s >= thr) - e["n_gt"]) / e["n_gt"]
        for e in eps
    ]
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "count_mape": float(np.mean(count_err)) if eps else 0.0,
        "episodes": len(eps),
        "instances": ngt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        required=True,
        help="folder with samples.json and data/*.png",
    )
    ap.add_argument("--method", choices=["ncc", "dino"], default="dino")
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument(
        "--images",
        type=int,
        default=400,
        help="tiles per split (first N after sorting)",
    )
    ap.add_argument("--max-episodes", type=int, default=4, help="classes per tile")
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    recs = [r for r in build_annotations(args.data) if (args.data / r["file"]).exists()]
    spotter = (
        NCCSpotter()
        if args.method == "ncc"
        else DinoSpotter(args.model, scale=args.scale)
    )
    tag = (
        args.method
        if args.method == "ncc"
        else f"dino-{args.model.split('/')[-1]}-x{args.scale:g}"
    )

    results = {}
    t0 = time.time()
    for split in ("val", "test"):
        pool = [
            r
            for r in recs
            if split_of(r["file"]) == split and episodes(r, args.max_episodes)
        ][: args.images]
        eps = []
        for rec in tqdm(pool, desc=f"{tag} {split}"):
            img = load_image(args.data, rec)
            for label, ex, gts in episodes(rec, args.max_episodes):
                eps.append(run_episode(spotter, img, label, ex, gts, args.min_score))
        results[split] = eps
    secs = time.time() - t0

    thrs = np.round(np.arange(args.min_score, 0.96, 0.01), 2)
    best = max(thrs, key=lambda t: prf(results["val"], t)["f1"])
    test = results["test"]
    report = {
        "method": tag,
        "threshold_from_val": float(best),
        "val": prf(results["val"], best),
        "test": prf(test, best),
        "test_by_size": {
            name: prf([e for e in test if lo <= e["size"] < hi], best)
            for lo, hi, name in SIZE_BUCKETS
        },
        "test_by_class": {
            c: prf([e for e in test if e["label"] == c], best)
            for c in sorted({e["label"] for e in test})
        },
        "seconds_total": round(secs, 1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{tag}.json").write_text(json.dumps(report, indent=2))
    (args.out / f"{tag}.episodes.json").write_text(json.dumps(results))
    t = report["test"]
    print(
        f"{tag}: thr={best:.2f}  test P={t['precision']:.3f} R={t['recall']:.3f} F1={t['f1']:.3f} count MAPE={t['count_mape']:.3f}  ({secs:.0f}s)"
    )
    for name, m in report["test_by_size"].items():
        print(f"  {name:20s} F1={m['f1']:.3f}  n={m['instances']}")


if __name__ == "__main__":
    main()
