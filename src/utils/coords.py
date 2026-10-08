"""Image geometry constants and (panel, t, f) <-> pixel <-> YOLO conversions."""

PANEL_ORDER = ["C5", "C3", "Cz", "C4", "C6"]  # top -> bottom, left -> right hemisphere
T_MIN, T_MAX = -1.0, 5.5  # seconds relative to cue
F_MIN, F_MAX = 4.0, 40.0  # Hz
IMG = 640  # image width and height in pixels
PANEL_H = IMG // len(PANEL_ORDER)  # 128 px per panel

CLASSES_5 = ["ERD_left_hand", "ERD_right_hand", "ERD_feet", "ERD_tongue", "ERS_rebound"]
CLASSES_2 = ["ERD_left_hand", "ERD_right_hand"]
