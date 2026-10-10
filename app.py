"""Gradio demo: click two corners around one symbol, get every other instance on the drawing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
from PIL import Image, ImageDraw

from planspot.adaptive import cut
from planspot.spotter import DinoSpotter, NCCSpotter

SPOTTERS = {
    "DINOv2 features (default)": DinoSpotter(),
    "Pixel NCC (classical baseline)": NCCSpotter(),
}
EXAMPLES = sorted(str(p) for p in (Path(__file__).parent / "examples").glob("*.png"))


def draw(img: Image.Image, clicks, dets=()):
    out = img.convert("RGBA")
    bg = Image.new("RGBA", out.size, (255, 255, 255, 255))
    out = Image.alpha_composite(bg, out).convert("RGB")
    d = ImageDraw.Draw(out)
    for det in dets:
        d.rectangle(det.box, outline=(220, 38, 38), width=3)
    if len(clicks) == 1:
        x, y = clicks[0]
        d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(37, 99, 235))
    if len(clicks) == 2:
        d.rectangle(box_of(clicks), outline=(37, 99, 235), width=3)
    return out


def box_of(clicks):
    (ax, ay), (bx, by) = clicks
    return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))


def on_upload(img):
    return (
        img,
        [],
        (draw(img, []) if img is not None else None),
        "Click the top-left, then the bottom-right corner of one symbol.",
    )


def on_click(img, clicks, evt: gr.SelectData):
    if img is None:
        return clicks, None, "Upload a drawing first."
    clicks = (
        (clicks + [tuple(evt.index)])[-2:] if len(clicks) < 2 else [tuple(evt.index)]
    )
    msg = (
        "Now click the opposite corner." if len(clicks) == 1 else "Box set. Press Spot."
    )
    return clicks, draw(img, clicks), msg


def on_spot(img, clicks, method, poses, threshold, adaptive=False):
    if img is None or len(clicks) != 2:
        return None, "Upload a drawing and click two corners first."
    box = box_of(clicks)
    if box[2] - box[0] < 6 or box[3] - box[1] < 6:
        return draw(
            img, clicks
        ), "The box is too small: click two opposite corners of the symbol."
    dets = SPOTTERS[method].detect(img, box, min_score=threshold, poses=int(poses))
    if adaptive:
        # Drop the exemplar's own match, then cut at the largest score gap above the floor.
        others = [d.score for d in dets if d.score < 0.999]
        t = cut(others, "gap", threshold, 0.0)
        dets = [d for d in dets if d.score >= t]
    return draw(
        img, clicks, dets
    ), f"Found {len(dets)} instance(s), including the exemplar."


with gr.Blocks(title="PlanSpot: one-shot symbol spotting") as demo:
    gr.Markdown(
        "# PlanSpot: one-shot symbol spotting on drawings\n"
        "Box **one** symbol (a door, a fixture, a device) and PlanSpot finds every other instance in any of 8 orientations, "
        "with no training for that symbol. Frozen DINOv2 patch features, tiled at 2x so small symbols span several patches. "
        "[Code and benchmark](https://github.com/dzakybd/planspot)."
    )
    src, clicks = gr.State(), gr.State([])
    with gr.Row():
        with gr.Column(scale=3):
            upload = gr.Image(
                type="pil", label="1. Upload a drawing (PNG/JPG)", height=220
            )
            canvas = gr.Image(
                type="pil",
                label="2. Click two corners around one symbol",
                interactive=False,
            )
        with gr.Column(scale=2):
            method = gr.Radio(
                list(SPOTTERS), value=next(iter(SPOTTERS)), label="Matcher"
            )
            poses = gr.Radio(
                ["1", "4", "8"],
                value="4",
                label="Orientations (8 = rotations + mirror; slower on CPU)",
            )
            threshold = gr.Slider(
                0.3,
                0.99,
                value=0.82,
                step=0.01,
                label="Score threshold (defaults tuned on validation drawings)",
            )
            adaptive = gr.Checkbox(
                False,
                label="Adaptive cut: stop at the largest score gap above the threshold (better on average, fails on ties)",
            )
            run = gr.Button("3. Spot", variant="primary")
            status = gr.Markdown("Upload a drawing or pick an example.")
            result = gr.Image(type="pil", label="Matches (red), exemplar (blue)")
    if EXAMPLES:
        gr.Examples(
            EXAMPLES,
            inputs=upload,
            label="Example tiles from FloorPlanCAD (CC BY-NC 4.0)",
        )
    gr.Markdown(
        "First query on a new drawing computes its feature maps (a few seconds on a GPU or Apple Silicon, longer on CPU). "
        "Later queries on the same drawing are fast. Examples: FloorPlanCAD (Fan et al., ICCV 2021), CC BY-NC 4.0."
    )
    method.change(
        lambda m, a: (
            (0.70 if a else 0.82) if m.startswith("DINO") else (0.82 if a else 0.90)
        ),
        [method, adaptive],
        threshold,
    )
    adaptive.change(
        lambda m, a: (
            (0.70 if a else 0.82) if m.startswith("DINO") else (0.82 if a else 0.90)
        ),
        [method, adaptive],
        threshold,
    )
    upload.change(on_upload, upload, [src, clicks, canvas, status])
    canvas.select(on_click, [src, clicks], [clicks, canvas, status])
    run.click(
        on_spot, [src, clicks, method, poses, threshold, adaptive], [result, status]
    )

if __name__ == "__main__":
    demo.launch()
