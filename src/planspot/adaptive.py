"""Per-query (adaptive) thresholds, evaluated offline from saved episode scores.

A single global threshold is a poor fit: the right cut-off depends on the drawing (clutter,
how many look-alike symbols). Two per-episode rules, both tuned on val and scored on test:

  global   keep score >= t
  gap      above a floor f, cut at the largest drop between consecutive sorted scores
  relative keep score >= f and score >= r * (best match score in this episode)
"""

import argparse
import json
from pathlib import Path

import numpy as np


def cut(scores: list[float], rule: str, a: float, b: float) -> float:
    """Return the threshold this rule picks for one episode."""
    s = sorted(scores, reverse=True)
    if rule == "global" or not s:
        return a
    if rule == "relative":
        return max(a, b * s[0])
    above = [x for x in s if x >= a]
    if len(above) < 2:
        return a
    gaps = [above[i] - above[i + 1] for i in range(len(above) - 1)]
    k = int(np.argmax(gaps))
    return above[k] if gaps[k] >= b else above[-1]


def score(eps, rule, a, b):
    tp = npred = ngt = 0
    for e in eps:
        t = cut([s for s, _ in e["scored"]], rule, a, b)
        kept = [ok for s, ok in e["scored"] if s >= t]
        tp += sum(kept)
        npred += len(kept)
        ngt += e["n_gt"]
    p = tp / npred if npred else 0.0
    r = tp / ngt if ngt else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


GRIDS = {
    "global": (np.arange(0.3, 0.99, 0.01), [0.0]),
    "gap": (np.arange(0.3, 0.95, 0.02), np.arange(0.0, 0.21, 0.01)),
    "relative": (np.arange(0.3, 0.95, 0.02), np.arange(0.5, 1.0, 0.02)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    out = {}
    for tag in ("ncc", "dino-dinov2-small-x2"):
        eps = json.loads((args.results / f"{tag}.episodes.json").read_text())
        out[tag] = {}
        for rule, (ga, gb) in GRIDS.items():
            a, b = max(
                ((a, b) for a in ga for b in gb),
                key=lambda ab: score(eps["val"], rule, *ab)["f1"],
            )
            t = score(eps["test"], rule, a, b)
            out[tag][rule] = {
                "params": [round(float(a), 2), round(float(b), 2)],
                **{k: round(v, 3) for k, v in t.items()},
            }
            print(
                f"{tag:22s} {rule:9s} params={out[tag][rule]['params']}  test P={t['precision']:.3f} R={t['recall']:.3f} F1={t['f1']:.3f}"
            )
    (args.results / "adaptive.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
