"""Phase 2: thin-structure (wall) segmentation with DINOv2 + LoRA and a topology-aware loss.

Walls in CAD drawings are 1-3 px strokes. A ViT alone sees 14 px patches, so the model fuses
DINOv2 features (context: "is this stroke a wall?") with a light full-resolution pixel branch
(geometry: "exactly where is the stroke?").

Ablation variants:
  unet       pixel branch + decoder only (no DINO)
  frozen     + frozen DINOv2 features
  lora       + LoRA (r=8) on DINOv2 attention q_proj/v_proj
  lora_cldice  lora + soft-clDice loss (Shit et al., CVPR 2021) for connectivity
"""

import base64
import io
import json
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from planspot.spotter import MEAN, PATCH, STD, dihedral, to_gray

# ---------------------------------------------------------------- data


def build_wall_masks(root: Path, label: str = "wall") -> list[str]:
    """Write root/masks/<label>/<tile>.png (union of instance masks); return tile files that have one."""
    out_dir = root / "masks" / label
    index = root / "masks" / f"{label}.json"
    if index.exists():
        return json.loads(index.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for s in json.loads((root / "samples.json").read_text())["samples"]:
        meta = s["metadata"]
        W, H = meta["width"], meta["height"]
        full = np.zeros((H, W), bool)
        for d in s["ground_truth"]["detections"]:
            if d["label"] != label or not d.get("mask"):
                continue
            m = (
                np.load(
                    io.BytesIO(
                        zlib.decompress(
                            base64.b64decode(d["mask"]["$binary"]["base64"])
                        )
                    )
                )
                > 0
            )
            x, y = round(d["bounding_box"][0] * W), round(d["bounding_box"][1] * H)
            h, w = min(m.shape[0], H - y), min(m.shape[1], W - x)
            full[y : y + h, x : x + w] |= m[:h, :w]
        if full.any():
            Image.fromarray(full.astype(np.uint8) * 255).save(
                out_dir / Path(s["filepath"]).name
            )
            files.append(s["filepath"])
    files.sort()
    index.write_text(json.dumps(files))
    return files


def load_pair(root: Path, file: str, label: str = "wall"):
    img = to_gray(Image.open(root / file))
    mask = np.asarray(Image.open(root / "masks" / label / Path(file).name)) > 0
    return img, mask.astype(np.float32)


class WallCrops(torch.utils.data.Dataset):
    """Random crops with random dihedral pose; ~80% of crops are forced to contain wall pixels."""

    def __init__(
        self,
        root: Path,
        files: list[str],
        crop: int = 448,
        length: int = 4000,
        seed: int = 0,
    ):
        self.pairs = [load_pair(root, f) for f in files]
        self.crop, self.length = crop, length
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        img, mask = self.pairs[self.rng.integers(len(self.pairs))]
        c = self.crop
        H, W = img.shape
        if self.rng.random() < 0.8:
            ys, xs = np.nonzero(mask)
            k = self.rng.integers(len(ys))
            y0 = int(np.clip(ys[k] - self.rng.integers(c), 0, H - c))
            x0 = int(np.clip(xs[k] - self.rng.integers(c), 0, W - c))
        else:
            y0, x0 = self.rng.integers(H - c + 1), self.rng.integers(W - c + 1)
        pose = int(self.rng.integers(8))
        im = dihedral(img[y0 : y0 + c, x0 : x0 + c], pose)
        ms = dihedral(mask[y0 : y0 + c, x0 : x0 + c], pose)
        return torch.from_numpy(im)[None], torch.from_numpy(ms)[None]


# ---------------------------------------------------------------- model


def conv_block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class WallSegmenter(nn.Module):
    def __init__(
        self,
        variant: str = "lora",
        model_id: str = "facebook/dinov2-small",
        lora_r: int = 8,
    ):
        super().__init__()
        self.variant = variant
        self.p1, self.p2, self.p4 = (
            conv_block(1, 16),
            conv_block(16, 32, 2),
            conv_block(32, 64, 2),
        )
        self.use_dino = variant != "unet"
        fuse_in = 64
        if self.use_dino:
            from transformers import AutoModel

            dino = AutoModel.from_pretrained(model_id)
            for p in dino.parameters():
                p.requires_grad = False
            if variant.startswith("lora"):
                from peft import LoraConfig, get_peft_model

                dino = get_peft_model(
                    dino,
                    LoraConfig(
                        r=lora_r,
                        lora_alpha=2 * lora_r,
                        target_modules=["q_proj", "v_proj"],
                        lora_dropout=0.0,
                    ),
                )
            self.dino = dino
            self.proj = nn.Sequential(
                nn.Conv2d(dino.config.hidden_size, 64, 1), nn.ReLU(inplace=True)
            )
            fuse_in = 128
        self.d4 = conv_block(fuse_in, 64)
        self.d2 = conv_block(64 + 32, 32)
        self.d1 = conv_block(32 + 16, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def dino_features(self, x):
        """x: (B,1,H,W) gray in [0,1], H and W multiples of 14 -> (B,C,H/14,W/14)."""
        rgb = (x.repeat(1, 3, 1, 1) - MEAN.to(x.device)) / STD.to(x.device)
        out = self.dino(pixel_values=rgb).last_hidden_state[:, 1:]
        B, _, H, W = x.shape
        return out.reshape(B, H // PATCH, W // PATCH, -1).permute(0, 3, 1, 2)

    def forward(self, x):
        f1 = self.p1(x)
        f2 = self.p2(f1)
        f4 = self.p4(f2)
        if self.use_dino:
            g = self.proj(self.dino_features(x))
            f4 = torch.cat(
                [
                    f4,
                    F.interpolate(
                        g, size=f4.shape[-2:], mode="bilinear", align_corners=False
                    ),
                ],
                1,
            )
        y = self.d4(f4)
        y = self.d2(
            torch.cat(
                [
                    F.interpolate(
                        y, size=f2.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    f2,
                ],
                1,
            )
        )
        y = self.d1(
            torch.cat(
                [
                    F.interpolate(
                        y, size=f1.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    f1,
                ],
                1,
            )
        )
        return self.head(y)


# ---------------------------------------------------------------- losses and metrics


def soft_skeleton(x, iters: int = 5):
    def erode(t):
        return -F.max_pool2d(-t, 3, 1, 1)

    def open_(t):
        return F.max_pool2d(erode(t), 3, 1, 1)

    skel = F.relu(x - open_(x))
    for _ in range(iters):
        x = erode(x)
        delta = F.relu(x - open_(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(prob, target, iters: int = 5, eps: float = 1e-6):
    sp, st = soft_skeleton(prob, iters), soft_skeleton(target, iters)
    tprec = ((sp * target).sum() + eps) / (sp.sum() + eps)
    tsens = ((st * prob).sum() + eps) / (st.sum() + eps)
    return 1 - 2 * tprec * tsens / (tprec + tsens)


def dice_loss(prob, target, eps: float = 1.0):
    return 1 - (2 * (prob * target).sum() + eps) / (prob.sum() + target.sum() + eps)


def tolerant_f1(pred: np.ndarray, gt: np.ndarray, tol: int = 2):
    """Pixel F1 where a pixel counts as matched if the other mask has a pixel within `tol` px."""
    import cv2

    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    gt_d = cv2.dilate(gt.astype(np.uint8), k) > 0
    pr_d = cv2.dilate(pred.astype(np.uint8), k) > 0
    p = (pred & gt_d).sum() / max(pred.sum(), 1)
    r = (gt & pr_d).sum() / max(gt.sum(), 1)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def cldice_score(pred: np.ndarray, gt: np.ndarray):
    from skimage.morphology import skeletonize

    sp, sg = skeletonize(pred), skeletonize(gt)
    tprec = (sp & gt).sum() / max(sp.sum(), 1)
    tsens = (sg & pred).sum() / max(sg.sum(), 1)
    return 2 * tprec * tsens / (tprec + tsens) if tprec + tsens else 0.0


def components(mask: np.ndarray) -> int:
    import cv2

    return cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1


@torch.no_grad()
def predict_tile(model, img: np.ndarray, device) -> np.ndarray:
    """Full-tile probability map; pads to a multiple of 14 (DINO) and of 4 (pixel branch)."""
    H, W = img.shape
    m = 28
    ph, pw = -(-H // m) * m, -(-W // m) * m
    pad = np.ones((ph, pw), np.float32)
    pad[:H, :W] = img
    x = torch.from_numpy(pad)[None, None].to(device)
    return torch.sigmoid(model(x))[0, 0, :H, :W].float().cpu().numpy()
