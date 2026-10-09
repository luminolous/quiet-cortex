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


QC_FREQ_TICKS = [4, 8, 13, 30, 40]  # Hz, per panel


def qc_figure(img: np.ndarray, boxes: pd.DataFrame, title: str, path) -> None:
    """Annotated QC overlay for manual review (not a training image).

    Time axis in seconds, frequency ticks per panel, dashed lines at the cue (0 s) and the end of
    imagery (4 s), thin two-tone box outlines with labels above the boxes, and a box list below.
    `boxes` needs x1, y1, x2, y2, class_name, channel, t_on, t_off, f_low, f_high, score.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from src.utils.coords import F_MAX, F_MIN, T_MAX, T_MIN, f_to_y

    n = len(boxes)
    fig = plt.figure(figsize=(8.6, 9.2 + 0.22 * n))
    ax = fig.add_axes([0.12, 0.08 + 0.026 * n / (1 + 0.024 * n), 0.84, 0.82 / (1 + 0.024 * n)])
    ax.imshow(img, extent=[T_MIN, T_MAX, IMG, 0], aspect="auto", interpolation="nearest")
    ticks, labels = [0.0], [f"{QC_FREQ_TICKS[-1]}"]
    for p, ch in enumerate(PANEL_ORDER):
        for f in QC_FREQ_TICKS[1:-1]:
            ticks.append(f_to_y(p, f))
            labels.append(f"{ch}  {f}" if f == 13 else str(f))
        # a panel boundary is 4 Hz of the panel above and 40 Hz of the panel below
        ticks.append((p + 1) * PANEL_H)
        labels.append(f"{QC_FREQ_TICKS[0]}" if p == len(PANEL_ORDER) - 1 else f"{QC_FREQ_TICKS[0]} | {QC_FREQ_TICKS[-1]}")
        if p:
            ax.axhline(p * PANEL_H, color="k", lw=0.8)
    ax.set_yticks(ticks, labels, fontsize=6.5)
    ax.set_ylabel(f"frequency (Hz) per panel, {F_MIN:.0f}-{F_MAX:.0f} Hz (top = {F_MAX:.0f} Hz)", fontsize=8)
    ax.set_xticks(np.arange(-1, 6, 1.0))
    ax.set_xlabel("time from cue (s)")
    for t in (0.0, 4.0):
        ax.axvline(t, color="k", ls="--", lw=0.9)
    sx = (T_MAX - T_MIN) / IMG  # pixel -> seconds
    for i, b in enumerate(boxes.itertuples()):
        x1, x2 = T_MIN + b.x1 * sx, T_MIN + b.x2 * sx
        color = np.array(CLASS_COLORS.get(b.class_name, (0, 0, 0))) / 255
        for lw, c in ((2.4, "white"), (1.0, color)):
            ax.add_patch(Rectangle((x1, b.y1), x2 - x1, b.y2 - b.y1, fill=False, lw=lw, ec=c))
        ax.annotate(f"{i}:{b.class_name}", (x1, b.y1), xytext=(0, 2), textcoords="offset points", fontsize=7,
                    va="bottom", ha="left", color="black",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.8, alpha=0.9))
    ax.set_title(title, fontsize=9)
    lines = ["idx  class         channel  time (s)       freq (Hz)   mean z"]
    lines += [f"{i:<4} {b.class_name:<13} {b.channel:<8} {b.t_on:5.2f}-{b.t_off:5.2f}   {b.f_low:4.0f}-{b.f_high:<4.0f}   "
              f"{b.score:+.2f}" for i, b in enumerate(boxes.itertuples())] or ["(no boxes)"]
    fig.text(0.12, 0.01, "\n".join(lines), family="monospace", fontsize=7.5, va="bottom")
    fig.savefig(path, dpi=110)
    plt.close(fig)



# --------------------------------------------------------------------------- result figures (Phase 6)


def preds_frame(pred: dict, classes: list[str], conf: float = 0.25) -> tuple[pd.DataFrame, list[float]]:
    """Prediction dict (boxes, scores, labels) -> (box DataFrame for draw_boxes, scores), filtered by confidence."""
    keep = pred["scores"] >= conf
    b = pred["boxes"][keep]
    df = pd.DataFrame(b, columns=["x1", "y1", "x2", "y2"]).assign(class_name=[classes[i] for i in pred["labels"][keep]])
    return df, pred["scores"][keep].tolist()


def gt_vs_pred(img: np.ndarray, gt: dict, pred: dict, classes: list[str], title: str, conf: float = 0.25) -> Image.Image:
    """Side-by-side overlays: ground truth (left) and predictions above `conf` with scores (right)."""
    g = pd.DataFrame(gt["boxes"], columns=["x1", "y1", "x2", "y2"]).assign(class_name=[classes[i] for i in gt["labels"]])
    p, s = preds_frame(pred, classes, conf)
    left = draw_boxes(img, g, title=f"{title}  | ground truth")
    right = draw_boxes(img, p, title=f"prediction (conf >= {conf})", scores=s)
    out = Image.new("RGB", (left.width + right.width + 10, left.height), "white")
    out.paste(left, (0, 0))
    out.paste(right, (left.width + 10, 0))
    return out


def grid(tiles: list[Image.Image], n_cols: int, scale: float = 0.5) -> Image.Image:
    w, h = tiles[0].size
    n_rows = -(-len(tiles) // n_cols)
    out = Image.new("RGB", (w * n_cols, h * n_rows), "white")
    for i, t in enumerate(tiles):
        out.paste(t, ((i % n_cols) * w, (i // n_cols) * h))
    return out.resize((int(out.width * scale), int(out.height * scale)), Image.LANCZOS)


def save_gif(frames: list[Image.Image], path, seconds_per_frame: float = 1.2) -> None:
    import imageio.v2 as imageio

    imageio.mimsave(path, [np.asarray(f.convert("RGB")) for f in frames], duration=seconds_per_frame, loop=0)


def plot_training_curves(curves: dict[str, pd.DataFrame], path=None):
    """Validation mAP@0.5 and mAP@0.5:0.95 per epoch. Each frame needs epoch, map50, map50_95."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
    for name, d in curves.items():
        axes[0].plot(d["epoch"], d["map50"], label=name, lw=1.3)
        axes[1].plot(d["epoch"], d["map50_95"], label=name, lw=1.3)
    for ax, t in zip(axes, ["val mAP@0.5", "val mAP@0.5:0.95"]):
        ax.set_xlabel("epoch")
        ax.set_ylabel(t)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    if path:
        fig.savefig(path, dpi=120)
    return fig


def plot_confusions(tables: dict[str, pd.DataFrame], path=None):
    """Confusion matrices (rows: true incl. background, cols: predicted), colored by row share."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(tables), figsize=(5.2 * len(tables), 4.6), constrained_layout=True)
    for ax, (name, cm) in zip(np.atleast_1d(axes), tables.items()):
        v = cm.to_numpy(dtype=float)
        share = v / np.clip(v.sum(axis=1, keepdims=True), 1, None)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                ax.text(j, i, int(v[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if share[i, j] > 0.6 else "black")
        labels = [c.replace("ERD_", "").replace("ERS_", "") for c in cm.columns]
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(name, fontsize=10)
    if path:
        fig.savefig(path, dpi=120)
    return fig


def plot_e4(e4: pd.DataFrame, path=None):
    """mAP@0.5 and mAP@0.5:0.95 versus nominal SNR (clean on the left)."""
    import matplotlib.pyplot as plt

    levels = [c.replace("map50_", "") for c in e4.columns if c.startswith("map50_") and not c.startswith("map50_95")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for _, r in e4.iterrows():
        for ax, key in zip(axes, ["map50_", "map50_95_"]):
            ax.plot(levels, [r[f"{key}{lv}"] for lv in levels], marker="o", label=r["model"])
    for ax, t in zip(axes, ["test mAP@0.5", "test mAP@0.5:0.95"]):
        ax.set_xlabel("noise level (nominal SNR per channel)")
        ax.set_ylabel(t)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    if path:
        fig.savefig(path, dpi=120)
    return fig


def plot_pipeline(path=None):
    """Simple block diagram of the pipeline (signal -> groups -> z map -> labels / detectors -> evaluation)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    steps = ["EEG (22 ch, 250 Hz)\nband-pass 4-40 Hz\nepochs -1.5...5.75 s", "small Laplacian\nC5, C3, Cz, C4, C6",
             "Morlet power\n4-40 Hz", "mean of 5 trials\n(same cue, subject,\nsplit)", "session z map (dB)\nsmoothing\nimage z -4...+4",
             "auto-labels\nz <= -3 ERD / >= +3 ERS\nclass = panel", "YOLO11n / YOLO11s /\nFaster R-CNN", "common evaluator\nmAP, domain errors,\ndecoding vs CSP+LDA"]
    fig, ax = plt.subplots(figsize=(16, 2.6))
    ax.set_xlim(0, len(steps) * 2)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for i, s in enumerate(steps):
        ax.add_patch(FancyBboxPatch((i * 2 + 0.1, 0.35), 1.7, 1.3, boxstyle="round,pad=0.05",
                                    fc="#e8f0fb" if i < 5 else "#fdf1e3", ec="0.3"))
        ax.text(i * 2 + 0.95, 1.0, s, ha="center", va="center", fontsize=8)
        if i < len(steps) - 1:
            ax.annotate("", (i * 2 + 2.1, 1.0), (i * 2 + 1.85, 1.0), arrowprops=dict(arrowstyle="->", lw=1.2))
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig
