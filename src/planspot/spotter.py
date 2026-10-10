"""One-shot symbol spotting on drawings.

Given a drawing and one exemplar box ("this symbol"), find every other instance.
Two matchers share the same peak-picking and NMS:

- ``ncc``:  pixel template matching (OpenCV normalized cross-correlation), a classical baseline.
- ``dino``: feature template matching on frozen DINOv2 patch features, extracted with
            upsampling + overlapping tiles so 40 px symbols still span several patches.

Both try the 8 dihedral poses of the exemplar (4 rotations x mirror), since drawings
place the same symbol in any orientation.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PATCH = 14
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in original pixels
    score: float
    pose: int


def to_gray(img: Image.Image) -> np.ndarray:
    """Drawing as float32 gray in [0, 1], ink dark on white (transparent -> white)."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def dihedral(a: np.ndarray, pose: int) -> np.ndarray:
    """Pose 0-3: rotate k*90 degrees; 4-7: mirror, then rotate."""
    if pose >= 4:
        a = a[:, ::-1]
    return np.ascontiguousarray(np.rot90(a, pose % 4))


def nms(dets: list[Detection], iou_thr: float) -> list[Detection]:
    if not dets:
        return []
    from torchvision.ops import nms as tv_nms

    boxes = torch.tensor([d.box for d in dets], dtype=torch.float32)
    scores = torch.tensor([d.score for d in dets], dtype=torch.float32)
    return [dets[i] for i in tv_nms(boxes, scores, iou_thr).tolist()]


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def peaks(
    score: np.ndarray, min_score: float, radius: int, top_k: int = 300
) -> list[tuple[int, int, float]]:
    """Up to top_k local maxima (y, x, value) of a 2-D score map above min_score."""
    k = 2 * radius + 1
    dil = cv2.dilate(score, np.ones((k, k), np.uint8))
    ys, xs = np.nonzero((score >= dil) & (score >= min_score))
    vals = score[ys, xs]
    order = np.argsort(-vals)[:top_k]
    return [(int(ys[i]), int(xs[i]), float(vals[i])) for i in order]


class NCCSpotter:
    name = "ncc"

    def detect(
        self, img: Image.Image, box, min_score=0.5, iou_thr=0.3, poses: int = 8
    ) -> list[Detection]:
        gray = to_gray(img)
        x0, y0, x1, y1 = (int(round(float(v))) for v in box)  # noqa: RUF046 (box may hold numpy floats)
        tmpl = gray[y0:y1, x0:x1]
        dets = []
        for pose in {1: [0], 4: [0, 1, 2, 3], 8: list(range(8))}[poses]:
            t = dihedral(tmpl, pose)
            th, tw = t.shape
            if th >= gray.shape[0] or tw >= gray.shape[1] or t.std() < 1e-3:
                continue
            res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
            res = np.nan_to_num(res, nan=-1.0)
            for y, x, s in peaks(res, min_score, max(1, min(th, tw) // 4)):
                dets.append(Detection((x, y, x + tw, y + th), s, pose))
        return nms(dets, iou_thr)


class DinoSpotter:
    name = "dino"

    def __init__(
        self,
        model_id="facebook/dinov2-small",
        scale=2.0,
        tile=518,
        overlap=126,
        device=None,
    ):
        from transformers import AutoModel

        self.device = device or (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.scale, self.tile, self.overlap = scale, tile, overlap
        self._cache: tuple[int, dict[int, torch.Tensor]] | None = None

    @torch.no_grad()
    def _patch_features(self, gray: np.ndarray) -> torch.Tensor:
        """(C, H/14, W/14) L2-normalized patch features of a gray image (already scaled)."""
        x = torch.from_numpy(gray)[None, None].repeat(1, 3, 1, 1)
        x = ((x - MEAN) / STD).to(self.device)
        h, w = gray.shape
        out = self.model(pixel_values=x).last_hidden_state[:, 1 + self._n_reg() :]
        feats = out.reshape(1, h // PATCH, w // PATCH, -1).permute(0, 3, 1, 2)[0]
        return F.normalize(feats, dim=0)

    def _n_reg(self) -> int:
        return getattr(self.model.config, "num_register_tokens", 0) or 0

    def feature_map(self, gray: np.ndarray) -> torch.Tensor:
        """Dense features for a full drawing via upsampling + overlapping tiles.

        Returns (C, Hp, Wp) on a grid with stride PATCH / scale in original pixels.
        """
        big = cv2.resize(
            gray,
            None,
            fx=self.scale,
            fy=self.scale,
            interpolation=cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR,
        )
        H, W = big.shape
        t, step = self.tile, self.tile - self.overlap
        Hp, Wp = -(-H // PATCH), -(-W // PATCH)
        pad = np.ones((max(Hp * PATCH, t), max(Wp * PATCH, t)), np.float32)
        pad[:H, :W] = big
        acc = cross = None
        ys = list(range(0, max(pad.shape[0] - t, 0) + 1, step))
        xs = list(range(0, max(pad.shape[1] - t, 0) + 1, step))
        if ys[-1] + t < pad.shape[0]:
            ys.append(pad.shape[0] - t)
        if xs[-1] + t < pad.shape[1]:
            xs.append(pad.shape[1] - t)
        for y in ys:
            for x in xs:
                f = self._patch_features(pad[y : y + t, x : x + t])
                if acc is None:
                    acc = torch.zeros(
                        f.shape[0],
                        pad.shape[0] // PATCH,
                        pad.shape[1] // PATCH,
                        device=f.device,
                    )
                    cross = torch.zeros(1, *acc.shape[1:], device=f.device)
                py, px = y // PATCH, x // PATCH
                acc[:, py : py + f.shape[1], px : px + f.shape[2]] += f
                cross[:, py : py + f.shape[1], px : px + f.shape[2]] += 1
        return F.normalize(acc / cross.clamp(min=1), dim=0)[:, :Hp, :Wp]

    def _pose_map(self, gray: np.ndarray, pose: int) -> torch.Tensor:
        """Feature map of the whole drawing in one pose (cached per drawing), so templates keep their context."""
        key = hash(gray.tobytes())
        if not self._cache or self._cache[0] != key:
            self._cache = (key, {})
        maps = self._cache[1]
        if pose not in maps:
            maps[pose] = self.feature_map(dihedral(gray, pose))
        return maps[pose]

    def detect(
        self, img: Image.Image, box, min_score=0.5, iou_thr=0.3, poses: int = 8
    ) -> list[Detection]:
        gray = to_gray(img)
        H, W = gray.shape
        pose_ids = {1: [0], 4: [0, 1, 2, 3], 8: list(range(8))}[poses]
        stride = PATCH / self.scale
        x0, y0, x1, y1 = box
        # Template = pose-0 cells under the exemplar box (at least 1x1).
        cx0, cy0 = int(x0 // stride), int(y0 // stride)
        cx1, cy1 = (
            max(cx0 + 1, int(np.ceil(x1 / stride))),
            max(cy0 + 1, int(np.ceil(y1 / stride))),
        )
        tmpl = self._pose_map(gray, 0)[:, cy0:cy1, cx0:cx1]
        th, tw = tmpl.shape[1:]
        dets = []
        for pose in pose_ids:
            fmap = self._pose_map(gray, pose)
            if th > fmap.shape[1] or tw > fmap.shape[2]:
                continue
            # Mean cosine similarity between template cells and each window of the posed map.
            s = (
                (F.conv2d(fmap[None], tmpl[None])[0, 0] / (th * tw))
                .float()
                .cpu()
                .numpy()
            )
            pw, ph = (W, H) if pose % 2 == 0 else (H, W)
            for y, x, v in peaks(s, min_score, max(1, min(th, tw) // 2)):
                bx0, by0 = x * stride, y * stride
                bx1, by1 = min(pw, (x + tw) * stride), min(ph, (y + th) * stride)
                dets.append(
                    Detection(unpose_box((bx0, by0, bx1, by1), pose, W, H), v, pose)
                )
        return nms(dets, iou_thr)


def _rot_ccw(x, y, w, h):
    """Point in a w x h image -> point after np.rot90 (counter-clockwise); new size h x w."""
    return y, w - x


def unpose_box(box, pose: int, W: int, H: int):
    """Map a box from the posed image back to the original W x H drawing."""
    pts = [(box[0], box[1]), (box[2], box[3])]
    k = pose % 4
    out = []
    for x, y in pts:
        # Undo k CCW rotations by applying 4-k more CCW rotations (sizes alternate).
        w, h = (W, H) if k % 2 == 0 else (H, W)
        for _ in range((4 - k) % 4):
            x, y = _rot_ccw(x, y, w, h)
            w, h = h, w
        if pose >= 4:
            x = W - x
        out.append((x, y))
    (ax, ay), (bx, by) = out
    return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
