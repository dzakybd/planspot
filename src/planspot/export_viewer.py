"""Precompute detections for a few example queries and write a static viewer bundle.

The viewer (space/index.html) draws the boxes client-side, so a threshold slider and a
method toggle work without any server.
"""

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from planspot.data import build_annotations
from planspot.spotter import DinoSpotter, NCCSpotter, box_iou, to_gray

# (tile, class label as stored in the dataset, display name verified by eye)
QUERIES = [
    ("data/0396-0047.png", "chair", "auditorium seat"),
    ("data/0800-4437.png", "chair", "classroom chair"),
    ("data/0035-0043.png", "chair", "classroom chair"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("space"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()

    recs = {r["file"]: r for r in build_annotations(args.data)}
    spotters = [DinoSpotter(), NCCSpotter()]
    (args.out / "img").mkdir(parents=True, exist_ok=True)
    items = []
    for file, label, name in QUERIES:
        rec = recs[file]
        gts = sorted(
            (o["box"] for o in rec["objects"] if o["label"] == label),
            key=lambda b: (round(b[1]), round(b[0])),
        )
        exemplar = gts[0]
        png = args.out / "img" / Path(file).name
        gray = to_gray(Image.open(args.data / file))
        Image.fromarray((gray * 255).astype("uint8")).save(png)
        entry = {
            "image": f"img/{png.name}",
            "name": name,
            "exemplar": exemplar,
            "gt": gts[1:],
            "methods": {},
        }
        for sp in spotters:
            dets = sp.detect(Image.open(args.data / file), exemplar, min_score=0.3)
            entry["methods"][sp.name] = [
                {"box": [round(v, 1) for v in d.box], "score": round(d.score, 3)}
                for d in dets
                if box_iou(d.box, exemplar) < 0.5
            ]
        items.append(entry)
    summary = {
        m: json.loads((args.results / f"{f}.json").read_text())
        for m, f in [("ncc", "ncc"), ("dino", "dino-dinov2-small-x2")]
    }
    adaptive = json.loads((args.results / "adaptive.json").read_text())
    walls = {
        p.stem: json.loads(p.read_text())
        for p in sorted((args.results / "walls").glob("*.json"))
        if not p.stem.endswith(".tiles")
    }
    (args.out / "data.json").write_text(
        json.dumps(
            {"items": items, "spotting": summary, "adaptive": adaptive, "walls": walls}
        )
    )
    for f in (
        (args.results / "figures").glob("*.png")
        if (args.results / "figures").exists()
        else []
    ):
        shutil.copy(f, args.out / "img" / f.name)
    print(f"wrote {len(items)} items to {args.out}")


if __name__ == "__main__":
    main()
