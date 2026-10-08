"""Tests for auto-labeling on synthetic ERD/ERS blobs and the YOLO label round trip."""

import numpy as np
import pandas as pd
import pytest

from src.autolabel import assign_splits, cue_match_stats, deep_update, label_trial, read_yolo_label, write_yolo_label
from src.data import crop_times, freqs_from_cfg
from src.utils import coords as C
from src.utils.io import load_config

CFG_LOC = load_config("configs/autolabel.yaml")  # project rules: location classes, strict z-dB rule
CFG = deep_update(CFG_LOC, {  # CONCEPT §5.4 rules, used by the cue-mode tests below
    "class_mode": "cue", "threshold_mode": "percent", "zscore_space": "percent", "min_duration_s": 0.2,
    "max_components_per_panel": None, "dominance": {"enabled": False}})
PCFG = load_config("configs/preprocess.yaml")
FREQS, TIMES = freqs_from_cfg(PCFG), crop_times(PCFG)
RIGHT, LEFT = 1, 0


def blob(erd, panel, t, f, value):
    ti = (TIMES >= t[0]) & (TIMES <= t[1])
    fi = (FREQS >= f[0]) & (FREQS <= f[1])
    erd[panel][np.ix_(fi, ti)] = value


def empty_map():
    return np.zeros((5, len(FREQS), len(TIMES)), dtype=np.float32)


def test_erd_blob_on_valid_panel():
    erd = empty_map()
    blob(erd, C.PANEL_ORDER.index("C3"), (1.0, 3.0), (10, 20), -50)
    boxes, _ = label_trial(erd, RIGHT, CFG, FREQS, TIMES)
    assert len(boxes) == 1
    b = boxes[0]
    assert (b["class_id"], b["channel"], b["status"]) == (RIGHT, "C3", "kept")
    assert b["score"] == pytest.approx(-50)
    assert (b["t_on"], b["t_off"], b["f_low"], b["f_high"]) == pytest.approx((1.0, 3.0, 10, 20))


def test_erd_clipped_to_window_and_band():
    erd = empty_map()
    blob(erd, C.PANEL_ORDER.index("C4"), (-1.0, 5.5), (4, 40), -50)
    boxes, _ = label_trial(erd, LEFT, CFG, FREQS, TIMES)
    b = [x for x in boxes if x["class_id"] == LEFT][0]
    assert (b["t_on"], b["t_off"], b["f_low"], b["f_high"]) == pytest.approx((0.5, 4.0, 8, 30))


def test_erd_out_of_panel_is_flagged():
    erd = empty_map()
    blob(erd, C.PANEL_ORDER.index("C4"), (1.0, 3.0), (10, 20), -50)
    boxes, _ = label_trial(erd, RIGHT, CFG, FREQS, TIMES)
    assert [b["status"] for b in boxes] == ["out_of_panel"]


def test_tongue_valid_on_c5_and_c6():
    erd = empty_map()
    blob(erd, C.PANEL_ORDER.index("C5"), (1.0, 2.0), (10, 14), -50)
    blob(erd, C.PANEL_ORDER.index("C6"), (1.0, 2.0), (10, 14), -50)
    boxes, _ = label_trial(erd, 3, CFG, FREQS, TIMES)
    assert [b["status"] for b in boxes] == ["kept", "kept"]


def test_ers_rebound_any_panel():
    erd = empty_map()
    blob(erd, C.PANEL_ORDER.index("Cz"), (4.3, 5.2), (16, 25), 50)
    boxes, _ = label_trial(erd, RIGHT, CFG, FREQS, TIMES)
    assert len(boxes) == 1 and boxes[0]["class_id"] == 4 and boxes[0]["status"] == "kept"
    assert (boxes[0]["f_low"], boxes[0]["f_high"]) == pytest.approx((16, 25))


def test_size_filter():
    erd = empty_map()
    blob(erd, C.PANEL_ORDER.index("C3"), (1.0, 1.1), (10, 20), -50)  # 100 ms
    blob(erd, C.PANEL_ORDER.index("C3"), (2.0, 3.0), (12, 13), -50)  # 1 Hz
    boxes, n_small = label_trial(erd, RIGHT, CFG, FREQS, TIMES)
    assert boxes == [] and n_small == 2


def test_four_connectivity_splits_diagonal_touch():
    erd = empty_map()
    p = C.PANEL_ORDER.index("C3")
    blob(erd, p, (1.0, 2.0), (10, 15), -50)
    t_next = TIMES[np.searchsorted(TIMES, 2.0) + 1]
    blob(erd, p, (t_next, 3.0), (16, 20), -50)  # touches only diagonally
    boxes, _ = label_trial(erd, RIGHT, CFG, FREQS, TIMES)
    assert len(boxes) == 2


def test_yolo_label_round_trip(tmp_path):
    x1, y1, x2, y2 = C.box_to_xyxy(1, 0.8, 3.4, 9, 26)
    df = pd.DataFrame([{"class_id": 1, "x1": x1, "y1": y1, "x2": x2, "y2": y2}])
    write_yolo_label(tmp_path / "a.txt", df)
    (c, *yolo), = read_yolo_label(tmp_path / "a.txt")
    assert c == 1
    assert C.yolo_to_xyxy(*yolo) == pytest.approx((x1, y1, x2, y2), abs=1e-3)
    write_yolo_label(tmp_path / "empty.txt", df.iloc[:0])
    assert read_yolo_label(tmp_path / "empty.txt") == []


def test_split_stratified_and_disjoint():
    rows = [{"subject": s, "session": ses, "trial_idx": i, "class_id": i % 4, "rejected": i == 3}
            for s in ("A01", "A02") for ses in ("T", "E") for i in range(40)]
    m = assign_splits(pd.DataFrame(rows), 0.2, 0)
    t = m[(m.session == "T") & ~m.rejected]
    assert set(m[m.session == "E"].split) == {"test", ""}
    assert (m[m.rejected].split == "").all()
    assert set(t.split) == {"train", "val"}
    assert t.groupby("subject").split.apply(lambda s: (s == "val").sum()).tolist() == [8, 8]


def test_max_one_component_keeps_largest():
    erd = empty_map()
    p = C.PANEL_ORDER.index("C3")
    blob(erd, p, (0.6, 1.4), (10, 13), -50)
    blob(erd, p, (2.0, 3.8), (15, 25), -50)
    cfg = dict(CFG, max_components_per_panel=1)
    boxes, _ = label_trial(erd, RIGHT, cfg, FREQS, TIMES)
    assert len(boxes) == 1 and boxes[0]["t_on"] == pytest.approx(2.0)


def test_dominance_rejects_bilateral_erd():
    erd = empty_map()
    blob(erd, C.PANEL_ORDER.index("C3"), (1.0, 3.0), (10, 20), -40)
    blob(erd, C.PANEL_ORDER.index("C4"), (1.0, 3.0), (10, 20), -60)  # stronger on the homolog
    cfg = dict(CFG, dominance=dict(CFG["dominance"], enabled=True))
    status = {b["channel"]: b["status"] for b in label_trial(erd, RIGHT, cfg, FREQS, TIMES)[0]}
    assert status == {"C3": "not_dominant", "C4": "out_of_panel"}


def test_zscore_values_and_threshold():
    from src.autolabel import baseline_stats, thresholds, to_values

    rng = np.random.default_rng(0)
    maps = rng.normal(5.0, 10.0, size=(20, 5, len(FREQS), len(TIMES))).astype(np.float32)
    cfg = dict(CFG, threshold_mode="zscore", zscore_k=2.5)
    mean, sd = baseline_stats(maps, TIMES, cfg["baseline_window_s"])
    assert mean.shape == sd.shape == (5, len(FREQS))
    z = to_values(maps, cfg, (mean, sd))
    assert z.mean() == pytest.approx(0.0, abs=0.05) and z.std() == pytest.approx(1.0, abs=0.05)
    assert thresholds(cfg) == (-2.5, 2.5)


def test_location_mode_ignores_cue():
    z = empty_map()  # values in z units
    blob(z, C.PANEL_ORDER.index("C4"), (1.0, 3.0), (10, 20), -5)
    blob(z, C.PANEL_ORDER.index("C5"), (1.0, 2.0), (10, 14), -5)
    for cue in (0, 1, 2, 3, None):
        boxes, _ = label_trial(z, cue, CFG_LOC, FREQS, TIMES)
        assert {(b["class_name"], b["status"]) for b in boxes} == {("ERD_C4", "kept"), ("ERD_lateral", "kept")}


def test_location_mode_strict_rules():
    z = empty_map()
    blob(z, C.PANEL_ORDER.index("C3"), (1.0, 1.3), (10, 20), -5)  # 300 ms < 0.5 s
    blob(z, C.PANEL_ORDER.index("Cz"), (1.0, 2.0), (10, 20), -3)  # weaker than mean(C3, C4) = -3.26 there
    blob(z, C.PANEL_ORDER.index("C4"), (1.0, 2.0), (10, 20), -5)
    boxes, n_small = label_trial(z, None, CFG_LOC, FREQS, TIMES)  # project default: no dominance
    assert n_small == 1
    assert {b["channel"]: b["status"] for b in boxes} == {"Cz": "kept", "C4": "kept"}
    dom = deep_update(CFG_LOC, {"dominance": {"enabled": True}})
    boxes, _ = label_trial(z, None, dom, FREQS, TIMES)
    assert {b["channel"]: b["status"] for b in boxes} == {"Cz": "not_dominant", "C4": "kept"}


def test_cue_match_stats():
    meta = pd.DataFrame({"subject": "A01", "session": "T", "trial_idx": [0, 1, 2], "class_id": [0, 1, 2],
                         "split": "train"})
    boxes = pd.DataFrame({"subject": "A01", "session": "T", "trial_idx": [0, 0, 1], "class_id": [0, 1, 0],
                          "score": [-4.0, -3.0, -5.0], "status": "kept"})
    row = cue_match_stats(meta, boxes).iloc[0]
    assert (row.n_trials, row.no_erd_box_pct, row.any_match_pct, row.strongest_match_pct) == (3, 33.3, 33.3, 33.3)


def test_phase_randomize_keeps_amplitude_spectrum():
    from src.autolabel import phase_randomize

    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 2, 1813))
    x[..., 500:700] += 5 * np.sin(np.arange(200) / 3)  # time-locked event
    y = phase_randomize(x, np.random.default_rng(1))
    assert y.shape == x.shape
    assert np.allclose(np.abs(np.fft.rfft(y)), np.abs(np.fft.rfft(x)), atol=1e-8)
    assert not np.allclose(y, x)
