"""Image geometry constants and (panel, t, f) <-> pixel <-> YOLO conversions.

Pixel coordinates are continuous: x = 0 is the left edge of the image (t = T_MIN) and
x = IMG is the right edge (t = T_MAX). Within panel p, y = p * PANEL_H is the top edge
(f = F_MAX) and y = (p + 1) * PANEL_H is the bottom edge (f = F_MIN).
"""

PANEL_ORDER = ["C5", "C3", "Cz", "C4", "C6"]  # top -> bottom, left -> right hemisphere
T_MIN, T_MAX = -1.0, 5.5  # seconds relative to cue
F_MIN, F_MAX = 4.0, 40.0  # Hz
IMG = 640  # image width and height in pixels
PANEL_H = IMG // len(PANEL_ORDER)  # 128 px per panel

CLASSES_5 = ["ERD_left_hand", "ERD_right_hand", "ERD_feet", "ERD_tongue", "ERS_rebound"]
CLASSES_2 = ["ERD_left_hand", "ERD_right_hand"]
MI_CLASSES = CLASSES_5[:4]  # cue classes; index = class_id = classlabel - 1


def t_to_x(t: float) -> float:
    """Time (s, relative to cue) -> x pixel."""
    return (t - T_MIN) / (T_MAX - T_MIN) * IMG


def x_to_t(x: float) -> float:
    """x pixel -> time (s, relative to cue)."""
    return T_MIN + x / IMG * (T_MAX - T_MIN)


def f_to_y(panel: int, f: float) -> float:
    """(panel index, frequency in Hz) -> y pixel."""
    return panel * PANEL_H + (1.0 - (f - F_MIN) / (F_MAX - F_MIN)) * PANEL_H


def y_to_panel_f(y: float) -> tuple[int, float]:
    """y pixel -> (panel index, frequency in Hz). The bottom image edge maps to the last panel."""
    panel = min(int(y // PANEL_H), len(PANEL_ORDER) - 1)
    f = F_MIN + (1.0 - (y - panel * PANEL_H) / PANEL_H) * (F_MAX - F_MIN)
    return panel, f


def box_to_xyxy(panel: int, t_on: float, t_off: float, f_low: float, f_high: float) -> tuple[float, float, float, float]:
    """Physical box -> absolute pixel box [x1, y1, x2, y2]."""
    return t_to_x(t_on), f_to_y(panel, f_high), t_to_x(t_off), f_to_y(panel, f_low)


def xyxy_to_box(x1: float, y1: float, x2: float, y2: float) -> tuple[int, float, float, float, float]:
    """Absolute pixel box -> physical box (panel, t_on, t_off, f_low, f_high).

    The panel is taken from the box center; frequencies are computed within that panel.
    """
    panel, _ = y_to_panel_f((y1 + y2) / 2)
    base = panel * PANEL_H
    f_high = F_MIN + (1.0 - (y1 - base) / PANEL_H) * (F_MAX - F_MIN)
    f_low = F_MIN + (1.0 - (y2 - base) / PANEL_H) * (F_MAX - F_MIN)
    return panel, x_to_t(x1), x_to_t(x2), f_low, f_high


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    """Absolute pixel box -> normalized YOLO (cx, cy, w, h)."""
    return (x1 + x2) / 2 / IMG, (y1 + y2) / 2 / IMG, (x2 - x1) / IMG, (y2 - y1) / IMG


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    """Normalized YOLO (cx, cy, w, h) -> absolute pixel box [x1, y1, x2, y2]."""
    return (cx - w / 2) * IMG, (cy - h / 2) * IMG, (cx + w / 2) * IMG, (cy + h / 2) * IMG
