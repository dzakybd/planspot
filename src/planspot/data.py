"""FloorPlanCAD (Voxel51 FiftyOne export on the Hugging Face Hub) -> compact box annotations."""

import ast
import json
from pathlib import Path

from PIL import Image

# "Stuff" or page-sized classes: not countable symbols, so not used for spotting.
NON_SYMBOL = {"wall", "parking", "railing", "single_door"}


def _parse(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def build_annotations(root: Path) -> list[dict]:
    """Read root/samples.json once and cache root/annotations.json: [{file, w, h, objects:[{label, box}]}]."""
    cache = root / "annotations.json"
    if cache.exists():
        return json.loads(cache.read_text())
    samples = json.loads((root / "samples.json").read_text())["samples"]
    out = []
    for s in samples:
        meta = _parse(s["metadata"])
        w, h = meta["width"], meta["height"]
        objs = []
        for d in _parse(s["ground_truth"])["detections"]:
            x, y, bw, bh = d["bounding_box"]
            objs.append(
                {"label": d["label"], "box": [x * w, y * h, (x + bw) * w, (y + bh) * h]}
            )
        out.append({"file": s["filepath"], "w": w, "h": h, "objects": objs})
    out.sort(key=lambda r: r["file"])
    cache.write_text(json.dumps(out))
    return out


def load_image(root: Path, rec: dict) -> Image.Image:
    return Image.open(root / rec["file"])
