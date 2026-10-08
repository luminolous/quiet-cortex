"""Tests for the common evaluator."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import decoding_metrics, detection_metrics, domain_metrics, iou_matrix, match
from src.utils import coords as C


def box(panel, t0, t1, f0, f1):
    return C.box_to_xyxy(panel, t0, t1, f0, f1)


GT = {
    "img0": {"boxes": np.array([box(3, 0.5, 3.0, 8, 25), box(0, 4.2, 5.0, 14, 20)]), "labels": np.array([0, 4])},
    "img1": {"boxes": np.array([box(1, 1.0, 2.0, 10, 20)]), "labels": np.array([1])},
    "img2": {"boxes": np.zeros((0, 4)), "labels": np.zeros(0, dtype=int)},  # background image
}


def perfect():
    return {k: {"boxes": v["boxes"].copy(), "labels": v["labels"].copy(), "scores": np.full(len(v["labels"]), 0.9)}
            for k, v in GT.items()}


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    assert iou_matrix(a, b)[0] == pytest.approx([1.0, 1 / 3, 0.0])


def test_perfect_predictions():
    m = detection_metrics(GT, perfect(), 5)
    assert m["map50"] == pytest.approx(1.0) and m["map50_95"] == pytest.approx(1.0)
    assert (m["precision"], m["recall"], m["f1"]) == (1.0, 1.0, 1.0)
    assert np.trace(m["confusion"]) == 3 and m["confusion"].sum() == 3
    d = domain_metrics(GT, perfect())
    assert d["onset_err_ms"] == pytest.approx(0) and d["fhigh_err_hz"] == pytest.approx(0)


def test_wrong_class_and_false_positive():
    p = perfect()
    p["img1"]["labels"] = np.array([2])  # right place, wrong class
    p["img2"] = {"boxes": np.array([box(2, 1, 2, 10, 20)]), "labels": np.array([2]), "scores": np.array([0.8])}
    m = detection_metrics(GT, p, 5)
    assert m["precision"] == pytest.approx(2 / 4) and m["recall"] == pytest.approx(2 / 3)
    cm = m["confusion"]
    assert cm[1, 2] == 1  # true ERD_C3 predicted as ERD_Cz
    assert cm[5, 2] == 1  # background predicted as ERD_Cz


def test_low_confidence_is_ignored_by_matching():
    p = perfect()
    p["img1"]["scores"] = np.array([0.1])
    pairs, fp, fn = match(GT["img1"], p["img1"])
    assert pairs == [] and len(fp) == 0 and list(fn) == [0]


def test_shifted_box_domain_error():
    p = perfect()
    p["img1"]["boxes"] = np.array([box(1, 1.1, 2.0, 10, 20)])  # onset 100 ms late, IoU 0.9
    d = domain_metrics({"img1": GT["img1"]}, {"img1": p["img1"]})
    assert d["onset_err_ms"] == pytest.approx(100, abs=1e-6) and d["offset_err_ms"] == pytest.approx(0, abs=1e-6)


def test_decoding_metrics():
    dec = pd.DataFrame({"subject": ["A01"] * 4, "true_class": [0, 1, 2, 3], "pred_class": [0, 1, -1, 2]})
    m = decoding_metrics(dec).set_index("subject").loc["all"]
    assert m.accuracy == 0.5 and m.kappa == pytest.approx((0.5 - 0.25) / 0.75) and m.no_decision_pct == 25.0
