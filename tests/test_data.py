"""Tests for Laplacian and rendering geometry in src.data."""

import numpy as np

from src.data import EEG_NAMES, laplacian, render_erd
from src.utils import coords as C
from src.utils.io import load_config

CFG = load_config("configs/preprocess.yaml")
N_FREQS, N_TIMES = 37, 1626


def test_laplacian_constant_signal_is_zero():
    epochs = np.ones((2, len(EEG_NAMES), 10))
    out = laplacian(epochs, EEG_NAMES, CFG["laplacian"])
    assert out.shape == (2, 5, 10)
    assert np.allclose(out, 0.0)


def test_laplacian_c3():
    epochs = np.zeros((1, len(EEG_NAMES), 4))
    epochs[0, EEG_NAMES.index("C3")] = 4.0
    epochs[0, EEG_NAMES.index("FC3")] = 2.0
    out = laplacian(epochs, EEG_NAMES, CFG["laplacian"])
    assert np.allclose(out[0, C.PANEL_ORDER.index("C3")], 4.0 - 2.0 / 4)


def test_render_places_blob_at_mapped_pixels():
    erd = np.zeros((5, N_FREQS, N_TIMES), dtype=np.float32)
    panel, (t_on, t_off), (f_lo, f_hi) = 3, (1.0, 2.0), (10, 20)
    t_idx = slice(int((t_on - C.T_MIN) * 250), int((t_off - C.T_MIN) * 250) + 1)
    erd[panel, f_lo - 4:f_hi - 4 + 1, t_idx] = -60.0
    img = render_erd(erd, CFG)
    assert img.shape == (C.IMG, C.IMG, 3) and img.dtype == np.uint8

    x1, y1, x2, y2 = C.box_to_xyxy(panel, t_on, t_off, f_lo, f_hi)
    inside = img[int(y1) + 3:int(y2) - 3, int(x1) + 3:int(x2) - 3].reshape(-1, 3)
    assert (inside[:, 2] > inside[:, 0] + 50).all()  # blue (ERD)
    white = img[int(y1) + 3:int(y2) - 3, int(x2) + 8:int(x2) + 40].reshape(-1, 3)
    assert (white.min(axis=1) > 230).all()  # 0 % -> near white, outside the blob
    other_panel = img[int(C.f_to_y(1, 15)), int(x1) + 5:int(x2) - 5]
    assert (other_panel.min(axis=1) > 230).all()
