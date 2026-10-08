"""Tests for the (panel, t, f) <-> pixel <-> YOLO mapping."""

import pytest

from src.utils import coords as C


def test_concept_example():
    """CONCEPT §5.3: right-hand ERD on C3, 0.8-3.4 s, 9-26 Hz."""
    yolo = C.xyxy_to_yolo(*C.box_to_xyxy(1, 0.8, 3.4, 9.0, 26.0))
    assert yolo == pytest.approx((0.4769, 0.3250, 0.4000, 0.0944), abs=1e-3)


def test_image_edges():
    assert C.t_to_x(C.T_MIN) == 0.0
    assert C.t_to_x(C.T_MAX) == pytest.approx(C.IMG)
    assert C.f_to_y(0, C.F_MAX) == 0.0
    assert C.f_to_y(4, C.F_MIN) == pytest.approx(C.IMG)
    assert C.f_to_y(2, C.F_MIN) == pytest.approx(3 * C.PANEL_H)


@pytest.mark.parametrize("panel", range(5))
def test_round_trip(panel):
    box = (panel, 0.5, 4.0, 8.0, 30.0)
    xyxy = C.yolo_to_xyxy(*C.xyxy_to_yolo(*C.box_to_xyxy(*box)))
    out = C.xyxy_to_box(*xyxy)
    assert out[0] == panel
    assert out[1:] == pytest.approx(box[1:], abs=1e-9)
