"""Threshold-based auto-labeling, train/val/test split, and YOLO dataset export (CONCEPT §5).

Per trial and panel: ERD mask (8-30 Hz, 0.5-4.0 s) and ERS mask (13-30 Hz, 4.0-5.5 s) -> 4-connected
components -> bounding boxes -> size filter -> optional largest-component and dominance rules ->
class rules -> YOLO labels. Box edges are the centers of the outermost time samples / frequency bins.

Usage:
    python -m src.autolabel --config configs/autolabel.yaml
    python -m src.autolabel --diagnose configs/autolabel_variants.yaml   # label-variant diagnosis (session T)
    python -m src.autolabel --null-test configs/autolabel_null.yaml      # real vs surrogate labels (session T)
"""

import argparse
import copy
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data import (crop_times, erd_path, freqs_from_cfg, laplacian, load_bandpassed_epochs, load_laplacian_epochs,
                      percent_change, render_erd, save_image, smooth, tfr_power)
from src.utils.coords import CLASSES_2, CLASSES_5, MI_CLASSES, PANEL_ORDER, box_to_xyxy, xyxy_to_yolo
from src.utils.io import load_config, resolve, setup_logging

log = logging.getLogger(__name__)

ERS_CLASS = CLASSES_5.index("ERS_rebound")
SPLITS = ["train", "val", "test"]
BOX_COLUMNS = ["subject", "session", "trial_idx", "split", "class_id", "class_name", "panel", "channel",
               "t_on", "t_off", "f_low", "f_high", "x1", "y1", "x2", "y2", "score", "status"]
TRIAL_KEY = ["subject", "session", "trial_idx"]


# --------------------------------------------------------------------------- values and thresholds


def baseline_stats(maps: np.ndarray, times: np.ndarray, window_s: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SD of baseline-window pixels over all trials and times -> (5, n_freqs) each."""
    b = (times >= window_s[0]) & (times <= window_s[1])
    vals = maps[..., b]
    return vals.mean(axis=(0, 3)), vals.std(axis=(0, 3))


def zscore_input(maps: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Scale in which z-scores are computed: % change as is, or dB = 10 log10(1 + % / 100).

    ("session_db" works on absolute power, not on % maps; see `zscore_values`.)
    """
    space = cfg.get("zscore_space", "percent")
    if space == "session_db":
        raise ValueError("zscore_space session_db needs absolute power; use zscore_values()")
    if space == "db":
        return 10.0 * np.log10(np.clip(1.0 + maps / 100.0, 1e-3, None))
    return maps


def file_stats(maps: np.ndarray, cfg: dict[str, Any], times: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Baseline statistics of one subject-session for z-scoring (None in percent mode)."""
    if cfg["threshold_mode"] != "zscore":
        return None
    return baseline_stats(zscore_input(maps, cfg), times, cfg["baseline_window_s"])


def to_values(maps: np.ndarray, cfg: dict[str, Any], stats: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    """Map % change maps (..., 5, n_freqs, n_times) into the space the thresholds apply to."""
    if cfg["threshold_mode"] == "percent":
        return maps
    mean, sd = stats
    return (zscore_input(maps, cfg) - mean[..., None]) / sd[..., None]


def thresholds(cfg: dict[str, Any]) -> tuple[float, float]:
    """(ERD threshold, ERS threshold) in the value space of `cfg['threshold_mode']`."""
    if cfg["threshold_mode"] == "percent":
        return cfg["erd"]["threshold_pct"], cfg["ers"]["threshold_pct"]
    if cfg["threshold_mode"] == "zscore":
        return -cfg["zscore_k"], cfg["zscore_k"]
    raise ValueError(f"unknown threshold_mode {cfg['threshold_mode']}")


# --------------------------------------------------------------------------- boxes


def find_components(panel_map: np.ndarray, freqs: np.ndarray, times: np.ndarray, threshold: float,
                    band_hz: list[float], window_s: list[float], connectivity: int = 4
                    ) -> list[tuple[float, float, float, float, int, tuple[slice, slice]]]:
    """Connected regions beyond `threshold` inside a band x window.

    A negative threshold selects values <= threshold (ERD), a positive one values >= threshold (ERS).
    Returns [(t_on, t_off, f_low, f_high, area_px, (freq_slice, time_slice))].
    """
    mask = panel_map <= threshold if threshold < 0 else panel_map >= threshold
    mask &= ((freqs >= band_hz[0]) & (freqs <= band_hz[1]))[:, None]
    mask &= ((times >= window_s[0]) & (times <= window_s[1]))[None, :]
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labels, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return []
    areas = ndimage.sum_labels(mask, labels, index=np.arange(1, n + 1))
    out = []
    for (f_sl, t_sl), area in zip(ndimage.find_objects(labels), areas):
        out.append((times[t_sl.start], times[t_sl.stop - 1], freqs[f_sl.start], freqs[f_sl.stop - 1],
                    int(area), (f_sl, t_sl)))
    return out


def label_trial(values: np.ndarray, class_id: int | None, cfg: dict[str, Any], freqs: np.ndarray,
                times: np.ndarray) -> tuple[list[dict[str, Any]], int]:
    """All candidate boxes of one trial with their status, plus the number removed by the size filter.

    `values` is (5, n_freqs, n_times) in the threshold space (see `to_values`). `class_id` (the cue)
    is only used when `class_mode` is "cue"; in "location" mode the ERD class comes from the panel.
    status: "kept"; "not_dominant" (ERD weaker than on the homolog panels, if dominance is enabled);
    "out_of_panel" (cue mode only: ERD outside the valid panels of the cue class).
    `score` is the mean value inside the box rectangle (threshold space).
    """
    erd_thr, ers_thr = thresholds(cfg)
    cue_mode = cfg.get("class_mode", "cue") == "cue"
    dom = cfg.get("dominance", {})
    max_n = cfg.get("max_components_per_panel")
    out, n_small = [], 0
    for p, ch in enumerate(PANEL_ORDER):
        if cue_mode:
            erd_cls, erd_valid = class_id, cfg["erd"]["valid_panels"][MI_CLASSES[class_id]]
        else:
            erd_cls, erd_valid = CLASSES_5.index(cfg["erd"]["location_classes"][ch]), PANEL_ORDER
        rules = [(erd_cls, cfg["erd"], erd_thr, erd_valid), (ERS_CLASS, cfg["ers"], ers_thr, cfg["ers"]["valid_panels"])]
        for cls, rule, thr, valid in rules:
            comps = []
            for c in find_components(values[p], freqs, times, thr, rule["band_hz"], rule["window_s"],
                                     cfg["connectivity"]):
                t_on, t_off, f_low, f_high = c[:4]
                if t_off - t_on < cfg["min_duration_s"] - 1e-9 or f_high - f_low < cfg["min_bandwidth_hz"] - 1e-9:
                    n_small += 1
                else:
                    comps.append(c)
            if max_n:
                comps = sorted(comps, key=lambda c: c[4], reverse=True)[:max_n]
            for t_on, t_off, f_low, f_high, _, rect in comps:
                status = "kept" if ch in valid else "out_of_panel"
                own = float(values[p][rect].mean())
                if cls != ERS_CLASS and dom.get("enabled"):
                    homolog = np.mean([values[PANEL_ORDER.index(h)][rect].mean() for h in dom["homologs"][ch]])
                    if not own < homolog:
                        status = "not_dominant"
                x1, y1, x2, y2 = box_to_xyxy(p, t_on, t_off, f_low, f_high)
                out.append({"class_id": cls, "class_name": CLASSES_5[cls], "panel": p, "channel": ch,
                            "t_on": t_on, "t_off": t_off, "f_low": f_low, "f_high": f_high,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": own, "status": status})
    return out, n_small


def label_file(maps: np.ndarray, trials: pd.DataFrame, cfg: dict[str, Any], freqs: np.ndarray, times: np.ndarray
               ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Label all kept trials of one subject-session from % maps. `maps` rows follow `trials` rows."""
    stats = file_stats(maps, cfg, times)
    return label_values(np.stack([to_values(m, cfg, stats) for m in maps]), trials, cfg, freqs, times)


def label_values(values: np.ndarray, trials: pd.DataFrame, cfg: dict[str, Any], freqs: np.ndarray, times: np.ndarray
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Label all kept trials of one subject-session from values already in threshold space."""
    rows, small = [], []
    for v, r in zip(values, trials.itertuples()):
        boxes, n_small = label_trial(v, r.class_id, cfg, freqs, times)
        key = {"subject": r.subject, "session": r.session, "trial_idx": r.trial_idx, "split": r.split}
        rows += [{**key, **b} for b in boxes]
        small.append({**key, "n_small": n_small})
    return rows, small


# --------------------------------------------------------------------------- split


def assign_splits(meta: pd.DataFrame, val_fraction: float, seed: int) -> pd.DataFrame:
    """Session T -> train/val (stratified by class within each subject); session E -> test; rejected -> ''."""
    meta = meta.copy()
    meta["split"] = ""
    meta.loc[(meta["session"] == "E") & ~meta["rejected"], "split"] = "test"
    for _, g in meta[(meta["session"] == "T") & ~meta["rejected"]].groupby("subject"):
        tr, va = train_test_split(g.index, test_size=val_fraction, stratify=g["class_id"], random_state=seed)
        meta.loc[tr, "split"] = "train"
        meta.loc[va, "split"] = "val"
    return meta


# --------------------------------------------------------------------------- export


def image_stem(subject: str, session: str, trial_idx: int) -> str:
    return f"{subject}{session}_trial_{trial_idx:03d}"


def write_yolo_label(path: Path, boxes: pd.DataFrame) -> None:
    """One line per box: `class cx cy w h` (normalized). Empty file for background images."""
    lines = [f"{int(b.class_id)} " + " ".join(f"{v:.6f}" for v in xyxy_to_yolo(b.x1, b.y1, b.x2, b.y2))
             for b in boxes.itertuples()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_yolo_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            c, *v = line.split()
            rows.append((int(c), *map(float, v)))
    return rows


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_data_yaml(path: Path, root: str, classes: list[str]) -> None:
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    path.write_text(
        f"# Ultralytics dataset config (generated by the dataset build: src.groups). Run commands from the repo root.\n"
        f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n",
        encoding="utf-8")


def build_datasets(meta: pd.DataFrame, boxes: pd.DataFrame, cfg: dict[str, Any], pcfg: dict[str, Any]) -> None:
    """Render images and write the 5-class and 2-class YOLO datasets plus their data YAMLs."""
    yolo = resolve(cfg["yolo_dir"])
    for name in ("5class", "2class"):
        if (yolo / name).exists():
            shutil.rmtree(yolo / name)
    kept_boxes = boxes[boxes["status"] == "kept"]
    by_trial = {k: g for k, g in kept_boxes.groupby(TRIAL_KEY)}
    empty = kept_boxes.iloc[:0]
    trials = meta[meta["split"] != ""]
    for r in tqdm(trials.itertuples(), total=len(trials), desc="render"):
        stem = image_stem(r.subject, r.session, r.trial_idx)
        img5 = yolo / "5class" / "images" / r.split / f"{stem}.png"
        save_image(render_erd(np.load(erd_path(pcfg, r.subject, r.session, r.trial_idx)), pcfg), img5)
        b = by_trial.get((r.subject, r.session, r.trial_idx), empty)
        write_yolo_label(yolo / "5class" / "labels" / r.split / f"{stem}.txt", b)
        if r.class_id < len(CLASSES_2):
            link_or_copy(img5, yolo / "2class" / "images" / r.split / f"{stem}.png")
            write_yolo_label(yolo / "2class" / "labels" / r.split / f"{stem}.txt", b[b["class_id"] < len(CLASSES_2)])
    write_data_yaml(resolve("configs/data_5class.yaml"), f"{cfg['yolo_dir']}/5class", CLASSES_5)
    write_data_yaml(resolve("configs/data_2class.yaml"), f"{cfg['yolo_dir']}/2class", CLASSES_2)


# --------------------------------------------------------------------------- statistics


def _dist(s: pd.Series) -> dict[str, float]:
    if s.empty:
        return {"mean": np.nan, "median": np.nan, "p10": np.nan, "p90": np.nan}
    return {"mean": s.mean(), "median": s.median(), "p10": s.quantile(0.1), "p90": s.quantile(0.9)}


def label_stats(meta: pd.DataFrame, boxes: pd.DataFrame) -> pd.DataFrame:
    """Per dataset x split x class: images, background images, boxes, duration and bandwidth distributions."""
    rows = []
    kept = boxes[boxes["status"] == "kept"]
    for ds, classes in (("5class", CLASSES_5), ("2class", CLASSES_2)):
        n_cls = len(classes)
        m = meta[meta["split"] != ""]
        b = kept[kept["class_id"] < n_cls]
        if ds == "2class":
            m = m[m["class_id"] < n_cls]
            b = b.merge(m[TRIAL_KEY], on=TRIAL_KEY)
        for split in SPLITS:
            ms, bs = m[m["split"] == split], b[b["split"] == split]
            with_box = bs.groupby(TRIAL_KEY).ngroups
            base = {"dataset": ds, "split": split, "n_images": len(ms), "n_background": len(ms) - with_box,
                    "background_pct": 100 * (len(ms) - with_box) / max(len(ms), 1)}
            for cname in ["all"] + classes:
                bc = bs if cname == "all" else bs[bs["class_name"] == cname]
                dur, bw = _dist(bc["t_off"] - bc["t_on"]), _dist(bc["f_high"] - bc["f_low"])
                rows.append({**base, "class_name": cname, "n_boxes": len(bc),
                             "n_images_with_class": bc.groupby(TRIAL_KEY).ngroups,
                             **{f"dur_s_{k}": v for k, v in dur.items()},
                             **{f"bw_hz_{k}": v for k, v in bw.items()}})
    return pd.DataFrame(rows).round(3)


def discard_stats(boxes: pd.DataFrame, n_small: pd.DataFrame) -> pd.DataFrame:
    """ERD boxes per status, cue class, and split (with out-of-panel counts per channel), plus size-filtered counts."""
    erd = boxes[boxes["class_id"] != ERS_CLASS]
    t = erd.pivot_table(index=["split", "class_name"], columns="status", values="t_on", aggfunc="size", fill_value=0)
    t = t.reindex(columns=["kept", "out_of_panel", "not_dominant"], fill_value=0)
    by_panel = (erd[erd["status"] == "out_of_panel"]
                .pivot_table(index=["split", "class_name"], columns="channel", values="t_on", aggfunc="size",
                             fill_value=0).add_prefix("out_of_panel_"))
    t = t.join(by_panel).fillna(0).astype(int)
    t["out_of_panel_pct"] = (100 * t["out_of_panel"] / (t["kept"] + t["out_of_panel"]).clip(lower=1)).round(1)
    small = n_small.groupby("split")["n_small"].sum().rename("size_filtered_all_classes")
    return t.reset_index().merge(small, on="split", how="left")


def cue_match_stats(meta: pd.DataFrame, boxes: pd.DataFrame) -> pd.DataFrame:
    """Per subject x session: how often the location class of the ERD boxes agrees with the cue.

    any_match: at least one ERD box of the cue's location class. strongest_match: the ERD box with the
    strongest ERD (lowest score) has the cue's location class. Both over all trials (no box = no match).
    Chance level for strongest_match is about 25 %.
    """
    erd = boxes[(boxes["status"] == "kept") & (boxes["class_id"] != ERS_CLASS)]
    strongest = erd.loc[erd.groupby(TRIAL_KEY)["score"].idxmin(), TRIAL_KEY + ["class_id"]]
    m = meta[meta["split"] != ""][TRIAL_KEY + ["class_id"]]
    m = m.merge(strongest.rename(columns={"class_id": "strongest"}), on=TRIAL_KEY, how="left")
    hits = erd[TRIAL_KEY + ["class_id"]].drop_duplicates().assign(any_match=True)
    m = m.merge(hits, on=TRIAL_KEY + ["class_id"], how="left")
    m["any_match"] = m["any_match"].astype("boolean").fillna(False).astype(bool)
    m["has_erd"] = m["strongest"].notna()
    m["strongest_match"] = m["strongest"] == m["class_id"]
    agg = lambda g: pd.Series({"n_trials": len(g), "no_erd_box_pct": 100 * (~g["has_erd"]).mean(),
                               "any_match_pct": 100 * g["any_match"].mean(),
                               "strongest_match_pct": 100 * g["strongest_match"].mean()})
    per = m.groupby(["subject", "session"]).apply(agg, include_groups=False).reset_index()
    tot = m.groupby("session").apply(agg, include_groups=False).reset_index().assign(subject="all")
    return pd.concat([per, tot], ignore_index=True).round(1)


# --------------------------------------------------------------------------- QC kit


def qc_kit(meta: pd.DataFrame, boxes: pd.DataFrame, cfg: dict[str, Any], pcfg: dict[str, Any]) -> None:
    """Random train images with at least one box (seed fixed): overlays + an empty verdict sheet."""
    from src.utils.viz import draw_boxes

    out = resolve("results/qc")
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()
    kept = boxes[(boxes["status"] == "kept") & (boxes["split"] == "train")]
    sample = kept[TRIAL_KEY].drop_duplicates().sample(n=cfg["qc"]["n_images"], random_state=cfg["qc"]["seed"])
    sheet = []
    for r in sample.sort_values(["subject", "trial_idx"]).itertuples(index=False):
        stem = image_stem(r.subject, r.session, r.trial_idx)
        b = kept[(kept["subject"] == r.subject) & (kept["session"] == r.session) & (kept["trial_idx"] == r.trial_idx)]
        img = render_erd(np.load(erd_path(pcfg, r.subject, r.session, r.trial_idx)), pcfg)
        cue = meta.loc[(meta["subject"] == r.subject) & (meta["session"] == r.session)
                       & (meta["trial_idx"] == r.trial_idx), "class_name"].iloc[0]
        draw_boxes(img, b, title=f"{stem}  cue: {cue}").save(out / f"{stem}.png")
        for i, bb in enumerate(b.itertuples()):
            sheet.append({"image": f"{stem}.png", "box_idx": i, "class": bb.class_name, "channel": bb.channel,
                          "t_on": bb.t_on, "t_off": bb.t_off, "f_low": bb.f_low, "f_high": bb.f_high,
                          "verdict": ""})
    pd.DataFrame(sheet).to_csv(out / "qc_sheet.csv", index=False)


def example_grid(meta: pd.DataFrame, boxes: pd.DataFrame, pcfg: dict[str, Any], path: Path,
                 n_per_class: int = 5, seed: int = 0, split: str = "train") -> None:
    """Grid of overlays: one row per cue class, `n_per_class` random trials each (rejected candidates in grey)."""
    from src.utils.viz import draw_boxes

    sample = (meta[meta["split"] == split].groupby("class_id").sample(n=n_per_class, random_state=seed)
              .sort_values(["class_id", "subject", "trial_idx"]))
    tiles = []
    for r in sample.itertuples():
        b = boxes[(boxes["subject"] == r.subject) & (boxes["session"] == r.session) & (boxes["trial_idx"] == r.trial_idx)]
        img = render_erd(np.load(erd_path(pcfg, r.subject, r.session, r.trial_idx)), pcfg)
        tiles.append(draw_boxes(img, b, title=f"{image_stem(r.subject, r.session, r.trial_idx)}  cue: {r.class_name}"))
    w, h = tiles[0].size
    grid = Image.new("RGB", (w * n_per_class, h * len(MI_CLASSES)), "white")
    for i, tile in enumerate(tiles):
        grid.paste(tile, ((i % n_per_class) * w, (i // n_per_class) * h))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.resize((grid.width // 2, grid.height // 2), Image.LANCZOS).save(path)


# --------------------------------------------------------------------------- variant diagnosis


def deep_update(base: dict[str, Any], upd: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def variant_metrics(name: str, boxes: pd.DataFrame, trials: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summary row and panel-specificity table (P(>=1 ERD candidate on panel | cue class)) for one variant."""
    erd = boxes[boxes["class_id"] != ERS_CLASS]
    cand = erd[erd["status"] != "not_dominant"]  # passed every rule except the valid-panel rule
    kept = boxes[boxes["status"] == "kept"]
    n = len(trials)
    row = {"variant": name, "n_trials": n,
           "erd_candidates": len(cand),
           "erd_valid_panel_pct": 100 * (cand["status"] == "kept").mean() if len(cand) else np.nan,
           "erd_not_dominant": int((erd["status"] == "not_dominant").sum())}
    for c, cname in enumerate(MI_CLASSES):
        cc = cand[cand["class_id"] == c]
        row[f"valid_pct_{cname}"] = 100 * (cc["status"] == "kept").mean() if len(cc) else np.nan
    for cname in CLASSES_5:
        row[f"n_{cname}"] = int((kept["class_name"] == cname).sum())
    with_any = kept[TRIAL_KEY].drop_duplicates().shape[0]
    with_erd = kept[kept["class_id"] != ERS_CLASS][TRIAL_KEY].drop_duplicates().shape[0]
    row["background_pct"] = 100 * (n - with_any) / n
    row["no_erd_box_pct"] = 100 * (n - with_erd) / n
    row["ers_per_image"] = row["n_ERS_rebound"] / n

    hit = cand.drop_duplicates(TRIAL_KEY + ["channel"]).merge(trials[TRIAL_KEY + ["class_id"]].rename(
        columns={"class_id": "cue"}), on=TRIAL_KEY)
    spec = (hit.groupby(["cue", "channel"]).size().unstack(fill_value=0)
            .reindex(index=range(len(MI_CLASSES)), columns=PANEL_ORDER, fill_value=0))
    spec = spec.div(trials.groupby("class_id").size().reindex(range(len(MI_CLASSES))), axis=0)
    spec.index = MI_CLASSES
    spec = spec.reset_index(names="cue_class").assign(variant=name)
    return row, spec


def diagnose(var_path: str) -> None:
    """Compare label variants on session T (train + val). Recomputes maps per smoothing setting from raw."""
    from src.utils.viz import draw_boxes

    vcfg = load_config(var_path)
    base = deep_update(load_config(vcfg["base_config"]), vcfg.get("base_overrides", {}))
    pcfg = load_config(base["preprocess_config"])
    meta = pd.read_csv(resolve(pcfg["paths"]["processed_dir"]) / "metadata.csv", keep_default_na=False)
    meta = meta[meta["session"].isin(vcfg["sessions"]) & (meta["split"] != "")]
    if vcfg.get("subjects"):
        meta = meta[meta["subject"].isin(vcfg["subjects"])]
    ex = vcfg["examples"]
    examples = (meta[meta["split"] == ex["split"]].groupby("class_id")
                .sample(n=ex["n_per_class"], random_state=ex["seed"]).sort_values(["class_id", "subject"]))
    ex_keys = list(zip(examples["subject"], examples["session"], examples["trial_idx"]))

    freqs, times = freqs_from_cfg(pcfg), crop_times(pcfg)
    lo, hi = vcfg["noise_window_s"]
    base_t = (times >= lo) & (times <= hi)
    erd_band = (freqs >= base["erd"]["band_hz"][0]) & (freqs <= base["erd"]["band_hz"][1])
    ers_band = (freqs >= base["ers"]["band_hz"][0]) & (freqs <= base["ers"]["band_hz"][1])
    variants = {k: (v["smoothing"], deep_update(base, v.get("overrides", {}))) for k, v in vcfg["variants"].items()}
    rows = {k: [] for k in variants}
    ex_imgs = {k: {} for k in variants}
    noise = {s: np.zeros(3) for s in vcfg["smoothing"]}

    for (subject, session), g in tqdm(list(meta.groupby(["subject", "session"])), desc="files"):
        _, kept, lap = load_laplacian_epochs(subject, session, pcfg)
        g = kept[["trial_idx"]].merge(g, on="trial_idx")  # trial order of `lap`
        pct = np.concatenate([percent_change(tfr_power(lap[i:i + pcfg["tfr_chunk"]], pcfg), pcfg).astype(np.float32)
                              for i in range(0, len(lap), pcfg["tfr_chunk"])])
        for s_name, s_par in vcfg["smoothing"].items():
            maps = smooth(pct, dict(pcfg, smoothing=s_par)).astype(np.float32)
            noise[s_name] += [(maps[:, :, erd_band][..., base_t] <= base["erd"]["threshold_pct"]).sum(),
                              (maps[:, :, ers_band][..., base_t] >= base["ers"]["threshold_pct"]).sum(),
                              maps[:, :, 0][..., base_t].size]
            for v_name, (vs, cfg) in variants.items():
                if vs != s_name:
                    continue
                r, _ = label_file(maps, g, cfg, freqs, times)
                rows[v_name] += r
                for i, t in enumerate(g.itertuples()):
                    if (subject, session, t.trial_idx) in ex_keys:
                        ex_imgs[v_name][(subject, session, t.trial_idx)] = render_erd(maps[i], pcfg)
        del pct

    tables, figs = resolve("results/tables"), resolve("results/figures/autolabel_variants")
    figs.mkdir(parents=True, exist_ok=True)
    summary, specs = [], []
    for v_name, (s_name, cfg) in variants.items():
        boxes = pd.DataFrame(rows[v_name], columns=BOX_COLUMNS)
        row, spec = variant_metrics(v_name, boxes, meta)
        e_cnt, s_cnt, tot = noise[s_name]
        s_par = vcfg["smoothing"][s_name]
        summary.append({**row, "smoothing": f"{s_par['sigma_hz']} Hz x {s_par['sigma_s']} s",
                        "threshold": (f"percent {cfg['erd']['threshold_pct']}/{cfg['ers']['threshold_pct']}"
                                      if cfg["threshold_mode"] == "percent"
                                      else f"zscore[{cfg.get('zscore_space', 'percent')}] k={cfg['zscore_k']}"),
                        "min_duration_s": cfg["min_duration_s"], "max_components": cfg["max_components_per_panel"],
                        "dominance": cfg["dominance"]["enabled"],
                        "noise_base_erd_pct": 100 * e_cnt / (tot * erd_band.sum()),
                        "noise_base_ers_pct": 100 * s_cnt / (tot * ers_band.sum())})
        specs.append(spec)

        tiles = []
        for key in ex_keys:
            b = boxes[(boxes["subject"] == key[0]) & (boxes["session"] == key[1]) & (boxes["trial_idx"] == key[2])]
            cue = MI_CLASSES[int(examples[(examples["subject"] == key[0]) & (examples["trial_idx"] == key[2])]["class_id"].iloc[0])]
            tiles.append(draw_boxes(ex_imgs[v_name][key], b, title=f"{key[0]}{key[1]} #{key[2]}  cue: {cue}"))
        w, h = tiles[0].size
        cols = ex["n_per_class"]
        grid = Image.new("RGB", (w * cols, h * len(MI_CLASSES)), "white")
        for i, tile in enumerate(tiles):
            grid.paste(tile, ((i % cols) * w, (i // cols) * h))
        grid.resize((grid.width // 2, grid.height // 2), Image.LANCZOS).save(figs / f"{v_name}.png")

    order = ["variant", "smoothing", "threshold", "min_duration_s", "max_components", "dominance"]
    df = pd.DataFrame(summary)
    df = df[order + [c for c in df.columns if c not in order]].round(2)
    df.to_csv(tables / "autolabel_variants.csv", index=False)
    pd.concat(specs).round(3).to_csv(tables / "autolabel_variants_specificity.csv", index=False)
    log.info("variant diagnosis written to %s and %s", tables, figs)


# --------------------------------------------------------------------------- null test (phase-randomized surrogates)


def phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fourier phase randomization along the last axis, independently per leading index.

    Keeps the amplitude spectrum of every (epoch, channel) and destroys cue-locked events.
    """
    spec = np.fft.rfft(x, axis=-1)
    phase = rng.uniform(0, 2 * np.pi, size=spec.shape)
    phase[..., 0] = 0.0
    if x.shape[-1] % 2 == 0:
        phase[..., -1] = 0.0
    return np.fft.irfft(np.abs(spec) * np.exp(1j * phase), n=x.shape[-1], axis=-1)


def zscore_values(power: np.ndarray, mode: str, pcfg: dict[str, Any], times: np.ndarray,
                  window_s: list[float]) -> np.ndarray:
    """z-scored dB values of cropped power (n, 5, n_freqs, n_times) for the null test.

    trial_db:   dB of the smoothed per-trial % change map (baseline-corrected per trial), as in labels.
    session_db: dB of smoothed absolute power, no per-trial correction (baseline and post-cue values
                come from the same distribution).
    z uses the mean / SD over the baseline-window pixels of all trials, per panel and frequency.
    """
    if mode == "trial_db":
        v = 10.0 * np.log10(np.clip(1.0 + smooth(percent_change(power, pcfg), pcfg) / 100.0, 1e-3, None))
    elif mode == "session_db":
        v = 10.0 * np.log10(smooth(power, pcfg))
    else:
        raise ValueError(mode)
    mean, sd = baseline_stats(v, times, window_s)
    return ((v - mean[..., None]) / sd[..., None]).astype(np.float32)


def null_test(null_path: str) -> None:
    """Real vs phase-randomized surrogate labels for several z thresholds and value modes (session T)."""
    ncfg = load_config(null_path)
    cfg = load_config(ncfg["base_config"])
    pcfg = load_config(cfg["preprocess_config"])
    meta = pd.read_csv(resolve(pcfg["paths"]["processed_dir"]) / "metadata.csv", keep_default_na=False)
    meta = meta[meta["session"].isin(ncfg["sessions"]) & (meta["split"] != "")]
    if ncfg.get("subjects"):
        meta = meta[meta["subject"].isin(ncfg["subjects"])]
    freqs, times = freqs_from_cfg(pcfg), crop_times(pcfg)
    rng = np.random.default_rng(ncfg["seed"])
    win = cfg["baseline_window_s"]
    post = (times >= cfg["erd"]["window_s"][0]) & (times <= cfg["erd"]["window_s"][1])
    base_t = (times >= win[0]) & (times <= win[1])
    band = (freqs >= cfg["erd"]["band_hz"][0]) & (freqs <= cfg["erd"]["band_hz"][1])

    rows, sd_rows = [], []
    for (subject, session), g in tqdm(list(meta.groupby(["subject", "session"])), desc="files"):
        _, kept, epochs, ch_names = load_bandpassed_epochs(subject, session, pcfg)
        g = kept[["trial_idx"]].merge(g, on="trial_idx")
        for data in ("real", "surrogate"):
            x = epochs if data == "real" else phase_randomize(epochs, rng)
            lap = laplacian(x, ch_names, pcfg["laplacian"])
            power = np.concatenate([tfr_power(lap[i:i + pcfg["tfr_chunk"]], pcfg).astype(np.float32)
                                    for i in range(0, len(lap), pcfg["tfr_chunk"])])
            for mode in ncfg["value_modes"]:
                z = zscore_values(power, mode, pcfg, times, win)
                sd_rows.append({"subject": subject, "data": data, "mode": mode,
                                "sd_post_over_sd_base": float(np.median(
                                    z[:, :, band][..., post].std(axis=(0, 3)) / z[:, :, band][..., base_t].std(axis=(0, 3))))})
                for k in ncfg["z_values"]:
                    c = deep_update(cfg, {"zscore_k": k})
                    for zi, r in zip(z, g.itertuples()):
                        boxes, _ = label_trial(zi, None, c, freqs, times)
                        kept_cls = [b["class_id"] for b in boxes if b["status"] == "kept"]
                        rows.append({"subject": subject, "trial_idx": r.trial_idx, "data": data, "mode": mode, "z": k,
                                     **{f"n_{n}": kept_cls.count(i) for i, n in enumerate(CLASSES_5)}})
            del power

    df = pd.DataFrame(rows)
    n_cols = [f"n_{n}" for n in CLASSES_5]
    agg = {**{c.replace("n_", "per_img_"): (c, "mean") for c in n_cols},
           **{c.replace("n_", "img_with_pct_"): (c, lambda s: 100 * (s > 0).mean()) for c in n_cols}}
    summ = df.assign(any_box=df[n_cols].sum(axis=1) > 0).groupby(["mode", "z", "data"]).agg(
        n_images=("trial_idx", "size"), background_pct=("any_box", lambda s: 100 * (1 - s.mean())), **agg)
    wide = summ.unstack("data")
    out = []
    for (mode, k), r in wide.iterrows():
        row = {"mode": mode, "z": k, "n_images": r[("n_images", "real")],
               "background_pct_real": r[("background_pct", "real")],
               "background_pct_surrogate": r[("background_pct", "surrogate")]}
        for n in CLASSES_5:
            re_, su = r[(f"per_img_{n}", "real")], r[(f"per_img_{n}", "surrogate")]
            row |= {f"{n}_per_img_real": re_, f"{n}_per_img_surr": su,
                    f"{n}_ratio": re_ / su if su > 0 else np.inf,
                    f"{n}_img_with_pct_surr": r[(f"img_with_pct_{n}", "surrogate")]}
        row["surr_max_img_with_pct"] = max(row[f"{n}_img_with_pct_surr"] for n in CLASSES_5)
        out.append(row)
    tables = resolve("results/tables")
    pd.DataFrame(out).round(3).to_csv(tables / "k1_label_null_test.csv", index=False)
    (pd.DataFrame(sd_rows).groupby(["mode", "data"])["sd_post_over_sd_base"].median().round(3)
     .reset_index().to_csv(tables / "k1_label_null_sd_ratio.csv", index=False))
    log.info("null test written to %s", tables)


# --------------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/autolabel.yaml")
    parser.add_argument("--skip-render", action="store_true", help="only boxes, split, and statistics")
    parser.add_argument("--diagnose", metavar="VARIANTS_YAML", help="compare label variants and exit")
    parser.add_argument("--null-test", metavar="NULL_YAML", help="real vs phase-randomized surrogate labels and exit")
    args = parser.parse_args()

    setup_logging()
    if args.diagnose:
        diagnose(args.diagnose)
        return
    if args.null_test:
        null_test(args.null_test)
        return
    cfg = load_config(args.config)
    pcfg = load_config(cfg["preprocess_config"])
    processed = resolve(pcfg["paths"]["processed_dir"])
    meta = pd.read_csv(processed / "metadata.csv", keep_default_na=False)
    meta = assign_splits(meta, cfg["split"]["val_fraction"], cfg["split"]["seed"])
    meta.to_csv(processed / "metadata.csv", index=False)

    freqs, times = freqs_from_cfg(pcfg), crop_times(pcfg)
    rows, small = [], []
    for (subject, session), g in tqdm(list(meta[meta["split"] != ""].groupby(["subject", "session"])), desc="autolabel"):
        if cfg["threshold_mode"] == "zscore" and cfg["zscore_space"] == "session_db":
            # absolute power is not stored: recompute it from the GDF file
            _, kept, lap = load_laplacian_epochs(subject, session, pcfg)
            g = kept[["trial_idx"]].merge(g, on="trial_idx")
            power = np.concatenate([tfr_power(lap[i:i + pcfg["tfr_chunk"]], pcfg).astype(np.float32)
                                    for i in range(0, len(lap), pcfg["tfr_chunk"])])
            values = zscore_values(power, "session_db", pcfg, times, cfg["baseline_window_s"])
            del power
            r, s = label_values(values, g, cfg, freqs, times)
        else:
            maps = np.stack([np.load(erd_path(pcfg, r.subject, r.session, r.trial_idx)) for r in g.itertuples()])
            r, s = label_file(maps, g, cfg, freqs, times)
        rows += r
        small += s
    boxes = pd.DataFrame(rows, columns=BOX_COLUMNS)
    boxes.to_csv(processed / "boxes.csv", index=False)
    log.info("boxes: %s", boxes.groupby("status").size().to_dict())

    tables = resolve("results/tables")
    tables.mkdir(parents=True, exist_ok=True)
    label_stats(meta, boxes).to_csv(tables / "label_stats.csv", index=False)
    discard_stats(boxes, pd.DataFrame(small)).to_csv(tables / "discarded_erd_boxes.csv", index=False)
    cue_match_stats(meta, boxes).to_csv(tables / "cue_match_by_subject.csv", index=False)
    example_grid(meta, boxes, pcfg, resolve("results/figures/k1_autolabel_examples.png"))

    if not args.skip_render:
        build_datasets(meta, boxes, cfg, pcfg)
        qc_kit(meta, boxes, cfg, pcfg)
    log.info("done")


if __name__ == "__main__":
    main()
