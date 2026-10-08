"""Tests for k-trial group construction and power averaging."""

import math

import numpy as np
import pandas as pd

from src.groups import group_power, make_groups, overlap_stats, sample_groups


def test_sample_groups_distinct_and_capped():
    rng = np.random.default_rng(0)
    g = sample_groups(np.arange(20), 5, 30, rng)
    assert len(g) == 30 and len(set(g)) == 30 and all(len(set(x)) == 5 for x in g)
    small = sample_groups(np.arange(6), 5, 30, rng)  # only C(6, 5) = 6 subsets exist
    assert len(small) == math.comb(6, 5)
    assert sample_groups(np.arange(3), 5, 10, rng) == []


def test_make_groups_never_mixes_splits_or_classes():
    rows = [{"subject": "A01", "session": "T", "trial_idx": i, "class_id": i % 4,
             "split": "train" if i < 60 else "val"} for i in range(80)]
    rows += [{"subject": "A01", "session": "E", "trial_idx": i, "class_id": i % 4, "split": "test"} for i in range(40)]
    meta = pd.DataFrame(rows)
    g = make_groups(meta, 3, {"train": 4, "val": 2, "test": 3}, seed=0)
    lookup = meta.set_index(["session", "trial_idx"])
    for r in g.itertuples():
        members = [lookup.loc[(r.session, int(t))] for t in r.trials.split(";")]
        assert {m.split for m in members} == {r.split}
        assert {m.class_id for m in members} == {r.class_id}
    assert g.groupby(["split"]).size().to_dict() == {"test": 12, "train": 16, "val": 8}
    assert make_groups(meta, 3, {"train": 4, "val": 2, "test": 3}, seed=0).equals(g)  # reproducible
    ov = overlap_stats(g, 3).set_index("split")
    assert (ov["n_unique"] == ov["n_groups"]).all()


def test_group_power_is_mean_of_members():
    power = np.arange(4 * 2, dtype=float).reshape(4, 2)[:, :, None, None] * np.ones((1, 1, 3, 5))
    groups = pd.DataFrame({"trials": ["10;30", "20;30;40"]})
    gp = group_power(power, np.array([10, 20, 30, 40]), groups)
    assert np.allclose(gp[0], power[[0, 2]].mean(0)) and np.allclose(gp[1], power[[1, 2, 3]].mean(0))


def test_session_z_matches_autolabel_zscore_values():
    from src.autolabel import zscore_values
    from src.data import crop_times, smooth
    from src.groups import session_z
    from src.utils.io import load_config

    p = load_config("configs/preprocess.yaml")
    t = crop_times(p)
    trials = np.random.default_rng(0).gamma(2.0, size=(6, 5, 37, len(t))).astype(np.float64)
    groups = pd.DataFrame({"trials": ["0;1;2", "3;4;5", "0;2;4"]})
    idx = np.arange(6)
    direct = zscore_values(group_power(trials, idx, groups), "session_db", p, t, [-1.0, 0.0])
    fast = session_z(group_power(smooth(trials, p), idx, groups), t, [-1.0, 0.0])
    assert np.allclose(direct, fast, atol=1e-3)
