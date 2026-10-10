"""Train and evaluate one wall-segmentation variant.

Held-out set = tiles whose source drawing hashes to the 'val' bucket of planspot.evaluate.split_of
(about 20% of drawings); everything else trains. Splitting by drawing keeps tiles of one drawing
out of both sides.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from planspot.evaluate import split_of
from planspot.walls import (
    WallCrops,
    WallSegmenter,
    build_wall_masks,
    cldice_score,
    components,
    dice_loss,
    load_pair,
    predict_tile,
    soft_cldice_loss,
    tolerant_f1,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument(
        "--variant", choices=["unet", "frozen", "lora", "lora_cldice"], default="lora"
    )
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--train-tiles", type=int, default=1500)
    ap.add_argument("--eval-tiles", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/walls"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    files = build_wall_masks(args.data)
    train_files = [f for f in files if split_of(f) == "test"][: args.train_tiles]
    eval_files = [f for f in files if split_of(f) == "val"][: args.eval_tiles]

    model = WallSegmenter(args.variant).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1
    )
    loader = torch.utils.data.DataLoader(
        WallCrops(
            args.data, train_files, length=args.steps * args.batch, seed=args.seed
        ),
        batch_size=args.batch,
    )

    t0 = time.time()
    model.train()
    for step, (x, y) in enumerate(tqdm(loader, desc=args.variant, mininterval=30)):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(prob, y)
        if args.variant == "lora_cldice":
            loss = loss + 0.5 * soft_cldice_loss(prob, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.4f}", flush=True)
    train_secs = time.time() - t0

    model.eval()
    rows = []
    for f in tqdm(eval_files, desc="eval", mininterval=30):
        img, gt = load_pair(args.data, f)
        pred = predict_tile(model, img, device) > 0.5
        gt = gt > 0.5
        p, r, f1 = tolerant_f1(pred, gt)
        rows.append(
            {
                "file": f,
                "precision": p,
                "recall": r,
                "f1": f1,
                "cldice": cldice_score(pred, gt),
                "components_pred": components(pred),
                "components_gt": components(gt),
            }
        )
    report = {
        "variant": args.variant,
        "trainable_params": n_train,
        "steps": args.steps,
        "batch": args.batch,
        "train_tiles": len(train_files),
        "eval_tiles": len(rows),
        "train_minutes": round(train_secs / 60, 1),
        "f1_tol2px": float(np.mean([r["f1"] for r in rows])),
        "precision": float(np.mean([r["precision"] for r in rows])),
        "recall": float(np.mean([r["recall"] for r in rows])),
        "cldice": float(np.mean([r["cldice"] for r in rows])),
        "component_count_error": float(
            np.mean([abs(r["components_pred"] - r["components_gt"]) for r in rows])
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.variant}.json").write_text(json.dumps(report, indent=2))
    (args.out / f"{args.variant}.tiles.json").write_text(json.dumps(rows))
    torch.save(
        {k: v for k, v in model.state_dict().items()}, args.out / f"{args.variant}.pt"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
