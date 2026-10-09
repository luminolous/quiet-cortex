"""Baselines: threshold detector (= the label rule) and CSP + LDA decoder evaluated per trial group.

Usage:
    python -m src.baselines threshold --config configs/experiments/threshold.yaml --split val
    python -m src.baselines csp_lda --config configs/experiments/csp_lda.yaml [--subjects A01]

The threshold detector applies the auto-labeling rule to the group z maps; its confidence is mean |z|
inside the box / 4, clipped to 1. On clean data its boxes are the labels themselves,
so it is an upper reference for E2, not a competitor.
"""

import argparse
import logging
from typing import Any

import numpy as np
import pandas as pd

from src.evaluate import decoding_metrics, group_info, save_predictions, split_images, upsert
from src.utils.io import load_config, resolve, setup_logging

log = logging.getLogger(__name__)

Z_CONF_SCALE = 4.0  # confidence = min(mean |z| / 4, 1)


# --------------------------------------------------------------------------- threshold detector


def threshold_predictions(run_cfg: dict[str, Any], split: str) -> dict[str, dict[str, np.ndarray]]:
    """Clean-data threshold detections for every image of a split, from the stored label boxes."""
    gcfg = load_config("configs/groups.yaml")
    pcfg = load_config(load_config(gcfg["autolabel_config"])["preprocess_config"])
    classes = list(load_config(run_cfg["data"])["names"].values())
    boxes = pd.read_csv(resolve(pcfg["paths"]["processed_dir"]) / "groups" / f"boxes_k{gcfg['k']}.csv")
    boxes = boxes[(boxes["status"] == "kept") & (boxes["split"] == split) & boxes["class_name"].isin(classes)]
    info = group_info().reset_index()
    boxes = boxes.merge(info[["subject", "session", "trial_idx", "image"]], on=["subject", "session", "trial_idx"])
    by_image = {k: g for k, g in boxes.groupby("image")}
    out = {}
    for img in split_images(run_cfg["data"], split):
        b = by_image.get(img.stem, boxes.iloc[:0])
        out[img.stem] = {"boxes": b[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float).reshape(-1, 4),
                         "scores": np.clip(b["score"].abs().to_numpy() / Z_CONF_SCALE, 0, 1),
                         "labels": b["class_name"].map(classes.index).to_numpy(dtype=int)}
    return out


# --------------------------------------------------------------------------- CSP + LDA


def csp_lda_decode(run_cfg: dict[str, Any], subjects: list[str] | None = None) -> pd.DataFrame:
    """Per subject: fit CSP + LDA on all kept session T trials; predict every session E test group by
    averaging the LDA class probabilities of its trials."""
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline

    from src.data import load_bandpassed_epochs

    c = run_cfg["csp_lda"]
    pcfg = load_config("configs/preprocess.yaml")
    pcfg = dict(pcfg, bandpass={"l_freq": c["band_hz"][0], "h_freq": c["band_hz"][1]})
    i0 = int(round((c["window_s"][0] - pcfg["epoch"]["tmin"]) * pcfg["sfreq"]))
    i1 = int(round((c["window_s"][1] - pcfg["epoch"]["tmin"]) * pcfg["sfreq"]))
    info = group_info().reset_index()
    tests = info[info["split"] == "test"]
    rows = []
    for subject in subjects or sorted(tests["subject"].unique()):
        _, kept_t, x_t, _ = load_bandpassed_epochs(subject, "T", pcfg)
        _, kept_e, x_e, _ = load_bandpassed_epochs(subject, "E", pcfg)
        clf = make_pipeline(CSP(n_components=c["n_components"], reg=c.get("reg"), log=True, norm_trace=False),
                            LinearDiscriminantAnalysis())
        clf.fit(x_t[..., i0:i1], kept_t["class_id"].to_numpy())
        proba = clf.predict_proba(x_e[..., i0:i1])
        pos = {t: i for i, t in enumerate(kept_e["trial_idx"])}
        trial_acc = float((proba.argmax(1) == kept_e["class_id"].to_numpy()).mean())
        for g in tests[tests["subject"] == subject].itertuples():
            p = proba[[pos[int(t)] for t in g.trials.split(";")]].mean(axis=0)
            rows.append({"image": g.image, "subject": subject, "true_class": int(g.class_id),
                         "pred_class": int(clf.classes_[np.argmax(p)]), "single_trial_acc": trial_acc})
        log.info("%s: single-trial acc %.3f", subject, trial_acc)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("method", choices=["threshold", "csp_lda"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    if args.method == "threshold":
        preds = threshold_predictions(cfg, args.split)
        path = save_predictions(cfg["run_id"], args.split, preds, list(load_config(cfg["data"])["names"].values()))
        log.info("threshold predictions -> %s (evaluate with: python -m src.evaluate --run %s --split %s)",
                 path, cfg["run_id"], args.split)
    else:
        dec = csp_lda_decode(cfg, args.subjects)
        out = resolve("results/predictions") / cfg["run_id"] / "test_decoding.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        dec.to_csv(out, index=False)
        if not args.subjects:  # only full runs go into the results tables
            m = decoding_metrics(dec)
            m.insert(0, "split", "test")
            m.insert(0, "run_id", cfg["run_id"])
            upsert(resolve("results/tables") / "decoding_by_subject.csv", m, ["run_id", "split"])
        log.info("CSP + LDA group decoding:\n%s", decoding_metrics(dec).to_string())


if __name__ == "__main__":
    main()
