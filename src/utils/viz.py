"""Visualization helpers: box overlays, image grids, GIFs."""

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.utils.coords import IMG, PANEL_H, PANEL_ORDER

CLASS_COLORS = {  # outline colors, distinct from the RdBu_r map
    "ERD_C4": (0, 200, 0),
    "ERD_C3": (255, 140, 0),
    "ERD_Cz": (160, 32, 240),
    "ERD_lateral": (0, 0, 0),
    "ERS_rebound": (255, 215, 0),
}
MARGIN_LEFT, MARGIN_TOP = 40, 22


def draw_boxes(img: np.ndarray, boxes: pd.DataFrame, title: str = "", scores: list[float] | None = None) -> Image.Image:
    """Overlay boxes (columns x1, y1, x2, y2, class_name) with their row index on a rendered image.

    Adds a left margin with channel names, thin panel separators, and a title line. The data area
    keeps its 640 x 640 pixel geometry. If a `status` column is present, rejected candidates
    (status != "kept") are drawn as thin grey outlines without a label.
    """
    if "status" in boxes.columns:
        rejected, boxes = boxes[boxes["status"] != "kept"], boxes[boxes["status"] == "kept"]
    else:
        rejected = boxes.iloc[:0]
    canvas = Image.new("RGB", (IMG + MARGIN_LEFT, IMG + MARGIN_TOP), "white")
    canvas.paste(Image.fromarray(img), (MARGIN_LEFT, MARGIN_TOP))
    d = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    d.text((4, 4), title, fill="black", font=font)
    for p, ch in enumerate(PANEL_ORDER):
        d.text((6, MARGIN_TOP + p * PANEL_H + PANEL_H // 2 - 6), ch, fill="black", font=font)
        if p:
            y = MARGIN_TOP + p * PANEL_H
            d.line([(MARGIN_LEFT, y), (MARGIN_LEFT + IMG, y)], fill=(90, 90, 90), width=1)
    for b in rejected.itertuples():
        d.rectangle([b.x1 + MARGIN_LEFT, b.y1 + MARGIN_TOP, b.x2 + MARGIN_LEFT, b.y2 + MARGIN_TOP],
                    outline=(150, 150, 150), width=1)
    for i, b in enumerate(boxes.itertuples()):
        color = CLASS_COLORS.get(b.class_name, (0, 0, 0))
        x1, y1, x2, y2 = b.x1 + MARGIN_LEFT, b.y1 + MARGIN_TOP, b.x2 + MARGIN_LEFT, b.y2 + MARGIN_TOP
        d.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{i}:{b.class_name.replace('ERS_', '')}"
        if scores is not None:
            label += f" {scores[i]:.2f}"
        tw = d.textlength(label, font=font)
        d.rectangle([x1, y1 - 12, x1 + tw + 4, y1], fill=color)
        d.text((x1 + 2, y1 - 12), label, fill="white" if color != (255, 215, 0) else "black", font=font)
    return canvas

