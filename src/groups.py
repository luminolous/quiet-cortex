"""k-trial groups (option B, Phase 2): power averaging, labels, null test, and YOLO dataset build.

One image = the mean Morlet power of k trials with the same cue, subject, session, and split (trial
pools of train / val / test never mix). The averaged power is smoothed and converted to session-level
dB z-scores (`configs/autolabel.yaml`, `zscore_space: session_db`); the same z map is rendered as the
image (RdBu_r, fixed z range) and thresholded for the location-based labels.

Usage:
    python -m src.groups --null-test      # real vs phase-randomized surrogate groups (session T)
    python -m src.groups --build          # YOLO datasets + statistics + examples + QC kit (selected k, z)
"""

import argparse
import itertools
import logging
import math
import shutil
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.autolabel import (BOX_COLUMNS, baseline_stats, cue_match_stats, deep_update, label_stats, label_values,
                           link_or_copy, phase_randomize, write_data_yaml, write_yolo_label)
from src.data import (crop_times, freqs_from_cfg, laplacian, load_bandpassed_epochs, render_erd, save_image, smooth,
                      tfr_power)
from src.utils.coords import CLASSES_2, CLASSES_5, MI_CLASSES
from src.utils.io import load_config, resolve, setup_logging

log = logging.getLogger(__name__)

SPLITS = ["train", "val", "test"]
GROUP_KEY = ["subject", "session", "trial_idx"]  # trial_idx holds the group id (reuses autolabel tables)


# --------------------------------------------------------------------------- groups


def sample_groups(pool: np.ndarray, k: int, n: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
    """Up to `n` distinct random k-subsets of `pool` (all subsets if fewer than `n` exist)."""
    pool = np.sort(pool)
    if len(pool) < k:
        return []
    if math.comb(len(pool), k) <= n:
        return [tuple(int(t) for t in c) for c in itertools.combinations(pool, k)]
    seen: set[tuple[int, ...]] = set()
    out = []
    while len(out) < n:
        g = tuple(sorted(int(t) for t in rng.choice(pool, size=k, replace=False)))
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def make_groups(meta: pd.DataFrame, k: int, n_groups: dict[str, int], seed: int) -> pd.DataFrame:
    """One row per group: subject, session, split, class_id, class_name, trial_idx (= group id), trials."""
    rows = []
    for (subject, session), m in meta[meta["split"] != ""].groupby(["subject", "session"]):
        gid = 0
        for split in SPLITS:
            for c in range(len(MI_CLASSES)):
                pool = m[(m["split"] == split) & (m["class_id"] == c)]["trial_idx"].to_numpy()
                if len(pool) == 0:
                    continue
                rng = np.random.default_rng([seed, k, int(subject[1:]), int(session == "E"), SPLITS.index(split), c])
                for g in sample_groups(pool, k, n_groups[split], rng):
                    rows.append({"subject": subject, "session": session, "split": split, "class_id": c,
                                 "class_name": MI_CLASSES[c], "trial_idx": gid, "n_pool": len(pool),
                                 "trials": ";".join(map(str, g))})
                    gid += 1
    return pd.DataFrame(rows)


def overlap_stats(groups: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per split: groups, unique groups, possible combinations, pairwise trial overlap, uses per trial."""
    rows = []
    for (split, subject, c), g in groups.groupby(["split", "subject", "class_id"]):
        sets = [set(t.split(";")) for t in g["trials"]]
        n_pool = int(g["n_pool"].iloc[0])
        pair = [len(a & b) / k for a, b in itertools.combinations(sets, 2)]
        rows.append({"split": split, "n_groups": len(sets), "n_unique": len({frozenset(s) for s in sets}),
                     "n_pool": n_pool, "log10_possible": math.log10(math.comb(n_pool, k)),
                     "mean_pair_overlap": np.mean(pair) if pair else np.nan,
                     "max_pair_overlap": np.max(pair) if pair else np.nan,
                     "uses_per_trial": len(sets) * k / n_pool})
    d = pd.DataFrame(rows)
    return (d.groupby("split").agg(n_groups=("n_groups", "sum"), n_unique=("n_unique", "sum"),
                                   pool_min=("n_pool", "min"), pool_median=("n_pool", "median"),
                                   log10_possible_min=("log10_possible", "min"),
                                   mean_pair_overlap=("mean_pair_overlap", "mean"),
                                   max_pair_overlap=("max_pair_overlap", "max"),
                                   uses_per_trial_mean=("uses_per_trial", "mean"))
            .reindex(SPLITS).reset_index().round(3))


# --------------------------------------------------------------------------- power


def file_power(subject: str, session: str, pcfg: dict[str, Any], surrogate_rng: np.random.Generator | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """(trial_idx of kept trials, cropped power (n, 5, n_freqs, n_times)); optionally phase-randomized."""
    _, kept, epochs, ch_names = load_bandpassed_epochs(subject, session, pcfg)
    if surrogate_rng is not None:
        epochs = phase_randomize(epochs, surrogate_rng)
    lap = laplacian(epochs, ch_names, pcfg["laplacian"])
    power = np.concatenate([tfr_power(lap[i:i + pcfg["tfr_chunk"]], pcfg).astype(np.float32)
                            for i in range(0, len(lap), pcfg["tfr_chunk"])])
    return kept["trial_idx"].to_numpy(), power


def group_power(power: np.ndarray, trial_idx: np.ndarray, groups: pd.DataFrame) -> np.ndarray:
    """Mean power over the trials of each group -> (n_groups, 5, n_freqs, n_times)."""
    pos = {t: i for i, t in enumerate(trial_idx)}
    return np.stack([power[[pos[int(t)] for t in g.split(";")]].mean(axis=0) for g in groups["trials"]])


def session_z(gpower_smoothed: np.ndarray, times: np.ndarray, window_s: list[float]) -> np.ndarray:
    """session_db z-scores (see `src.autolabel.zscore_values`) of already smoothed group power.

    Smoothing is linear, so smoothing every trial once and then averaging equals smoothing the group
    mean; this avoids smoothing each group again.
    """
    v = 10.0 * np.log10(gpower_smoothed)
    mean, sd = baseline_stats(v, times, window_s)
    return ((v - mean[..., None]) / sd[..., None]).astype(np.float32)


def label_groups(gpower_smoothed: np.ndarray, groups: pd.DataFrame, acfg: dict[str, Any], pcfg: dict[str, Any]
                 ) -> list[dict[str, Any]]:
    """Location labels of the groups of one subject-session (z statistics from these groups)."""
    times, freqs = crop_times(pcfg), freqs_from_cfg(pcfg)
    rows, _ = label_values(session_z(gpower_smoothed, times, acfg["baseline_window_s"]), groups, acfg, freqs, times)
    return rows


# --------------------------------------------------------------------------- null test


def null_test(gcfg: dict[str, Any], acfg: dict[str, Any], pcfg: dict[str, Any], meta: pd.DataFrame) -> pd.DataFrame:
    """Real vs surrogate group labels for every k x z in the config (session T, train + val groups)."""
    ncfg = gcfg["null_test"]
    meta = meta[meta["session"].isin(ncfg["sessions"])]
    counts = []
    for subject in tqdm(sorted(meta["subject"].unique()), desc="null test"):
        m = meta[meta["subject"] == subject]
        tidx, real = file_power(subject, "T", pcfg)
        _, surr = file_power(subject, "T", pcfg, np.random.default_rng([ncfg["surrogate_seed"], int(subject[1:])]))
        real, surr = smooth(real, pcfg), smooth(surr, pcfg)
        for k in ncfg["k_values"]:
            groups = make_groups(m, k, gcfg["n_groups"], gcfg["seed"])
            for data, power in (("real", real), ("surrogate", surr)):
                gp = group_power(power, tidx, groups)
                for z in ncfg["z_values"]:
                    rows = label_groups(gp, groups, deep_update(acfg, {"zscore_k": z}), pcfg)
                    b = pd.DataFrame(rows, columns=BOX_COLUMNS)
                    b = b[b["status"] == "kept"]
                    per = b.groupby(["trial_idx", "class_name"]).size().unstack(fill_value=0)
                    per = per.reindex(index=groups["trial_idx"], columns=CLASSES_5, fill_value=0)
                    counts.append(per.assign(subject=subject, k=k, z=z, data=data).reset_index())
            del gp
        del real, surr
    df = pd.concat(counts, ignore_index=True)
    out = []
    for (k, z), g in df.groupby(["k", "z"]):
        row = {"k": k, "z": z}
        for data in ("real", "surrogate"):
            d = g[g["data"] == data]
            row[f"n_images_{data}"] = len(d)
            row[f"background_pct_{data}"] = 100 * (d[CLASSES_5].sum(axis=1) == 0).mean()
        for c in CLASSES_5:
            r, s = g[g["data"] == "real"][c], g[g["data"] == "surrogate"][c]
            row |= {f"{c}_per_img_real": r.mean(), f"{c}_per_img_surr": s.mean(),
                    f"{c}_ratio": r.mean() / s.mean() if s.mean() > 0 else np.inf,
                    f"{c}_img_with_pct_real": 100 * (r > 0).mean(), f"{c}_img_with_pct_surr": 100 * (s > 0).mean()}
        out.append(row)
    res = pd.DataFrame(out).round(3)
    res.to_csv(resolve("results/tables") / ncfg["output"], index=False)
    return res


# --------------------------------------------------------------------------- build (selected k, z)


def chance_cue_match(groups: pd.DataFrame, boxes: pd.DataFrame, n_perm: int = 20, seed: int = 0) -> pd.DataFrame:
    """Cue-match chance level: cues shuffled among groups within subject-session-split."""
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(n_perm):
        g = groups.copy()
        g["class_id"] = g.groupby(["subject", "session", "split"])["class_id"].transform(
            lambda s: rng.permutation(s.to_numpy()))
        res.append(cue_match_stats(g, boxes).set_index(["subject", "session"]))
    ch = sum(r[["any_match_pct", "strongest_match_pct"]] for r in res) / n_perm
    return ch.add_suffix("_chance").round(1).reset_index()


def example_grid(tiles: list[Image.Image], n_cols: int, path) -> None:
    w, h = tiles[0].size
    grid = Image.new("RGB", (w * n_cols, h * math.ceil(len(tiles) / n_cols)), "white")
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % n_cols) * w, (i // n_cols) * h))
    grid.resize((grid.width // 2, grid.height // 2), Image.LANCZOS).save(path)


def dataset_classes(drop: list[str]) -> tuple[list[str], list[str]]:
    """(5-class list, 2-class list) after dropping classes that failed the null test."""
    return [c for c in CLASSES_5 if c not in drop], [c for c in CLASSES_2 if c not in drop]


def write_labels(path, boxes: pd.DataFrame, classes: list[str]) -> None:
    """YOLO label file with class ids re-indexed to `classes` (boxes of other classes are skipped)."""
    b = boxes[boxes["class_name"].isin(classes)].assign(class_id=lambda d: d["class_name"].map(classes.index))
    write_yolo_label(path, b)


def qc_kit(groups: pd.DataFrame, boxes: pd.DataFrame, image_dir, cfg: dict[str, Any]) -> None:
    """Random train images with at least one box (seed fixed): overlays + an empty verdict sheet."""
    from src.utils.viz import draw_boxes

    out = resolve("results/qc")
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()
    kept = boxes[(boxes["status"] == "kept") & (boxes["split"] == "train")]
    keys = kept[GROUP_KEY].drop_duplicates().sample(n=cfg["n_images"], random_state=cfg["seed"])
    info = groups.set_index(GROUP_KEY)
    sheet = []
    for key in keys.sort_values(GROUP_KEY).itertuples(index=False):
        g = info.loc[tuple(key)]
        b = kept[(kept["subject"] == key.subject) & (kept["session"] == key.session)
                 & (kept["trial_idx"] == key.trial_idx)]
        img = np.asarray(Image.open(image_dir / "train" / f"{g.image}.png").convert("RGB"))
        draw_boxes(img, b, title=f"{g.image}  cue: {g.class_name}").save(out / f"{g.image}.png")
        for i, bb in enumerate(b.itertuples()):
            sheet.append({"image": f"{g.image}.png", "box_idx": i, "class": bb.class_name, "channel": bb.channel,
                          "t_on": bb.t_on, "t_off": bb.t_off, "f_low": bb.f_low, "f_high": bb.f_high,
                          "verdict": ""})
    pd.DataFrame(sheet).to_csv(out / "qc_sheet.csv", index=False)


def build(gcfg: dict[str, Any], acfg: dict[str, Any], pcfg: dict[str, Any], meta: pd.DataFrame) -> None:
    """Build the k-group YOLO datasets (5class, 2class) with z-map images, plus statistics and QC kit."""
    from src.utils.viz import draw_boxes

    k = gcfg["k"]
    acfg = deep_update(acfg, {"zscore_k": gcfg["zscore_k"]})
    rcfg = dict(pcfg, render=gcfg["render"])
    times, freqs = crop_times(pcfg), freqs_from_cfg(pcfg)
    cls5, cls2 = dataset_classes(gcfg.get("drop_classes", []))
    groups = make_groups(meta, k, gcfg["n_groups"], gcfg["seed"])
    groups["image"] = [f"{s}{ses}_group_{g:04d}"
                       for s, ses, g in zip(groups["subject"], groups["session"], groups["trial_idx"])]

    yolo = resolve(gcfg["yolo_dir"])
    for name in ("5class", "2class"):
        if (yolo / name).exists():
            shutil.rmtree(yolo / name)
    rows = []
    for (subject, session), g in tqdm(list(groups.groupby(["subject", "session"])), desc="build"):
        tidx, power = file_power(subject, session, pcfg)
        z = session_z(group_power(smooth(power, pcfg), tidx, g), times, acfg["baseline_window_s"])
        del power
        r, _ = label_values(z, g, acfg, freqs, times)
        rows += r
        b = pd.DataFrame(r, columns=BOX_COLUMNS)
        b = b[b["status"] == "kept"]
        for zi, gr in zip(z, g.itertuples()):
            bg = b[b["trial_idx"] == gr.trial_idx]
            img5 = yolo / "5class" / "images" / gr.split / f"{gr.image}.png"
            save_image(render_erd(zi, rcfg), img5)
            write_labels(yolo / "5class" / "labels" / gr.split / f"{gr.image}.txt", bg, cls5)
            if gr.class_id < len(CLASSES_2):
                link_or_copy(img5, yolo / "2class" / "images" / gr.split / f"{gr.image}.png")
                write_labels(yolo / "2class" / "labels" / gr.split / f"{gr.image}.txt", bg, cls2)
        del z
    write_data_yaml(resolve("configs/data_5class.yaml"), f"{gcfg['yolo_dir']}/5class", cls5)
    write_data_yaml(resolve("configs/data_2class.yaml"), f"{gcfg['yolo_dir']}/2class", cls2)

    boxes = pd.DataFrame(rows, columns=BOX_COLUMNS)
    boxes.loc[~boxes["class_name"].isin(cls5) & (boxes["status"] == "kept"), "status"] = "dropped_class"
    out = resolve(pcfg["paths"]["processed_dir"]) / "groups"
    out.mkdir(parents=True, exist_ok=True)
    groups.to_csv(out / f"groups_k{k}.csv", index=False)
    boxes.to_csv(out / f"boxes_k{k}.csv", index=False)

    prefix = gcfg["output_prefix"]
    tables = resolve("results/tables")
    label_stats(groups, boxes).to_csv(tables / f"{prefix}_label_stats.csv", index=False)
    overlap_stats(groups, k).to_csv(tables / f"{prefix}_group_overlap.csv", index=False)
    cm = cue_match_stats(groups, boxes).merge(chance_cue_match(groups, boxes), on=["subject", "session"])
    cm.to_csv(tables / f"{prefix}_cue_match_by_subject.csv", index=False)

    ex = gcfg["examples"]
    ex_groups = (groups[groups["split"] == ex["split"]].groupby("class_id")
                 .sample(n=ex["n_per_class"], random_state=ex["seed"]).sort_values(["class_id", "subject"]))
    kept = boxes[boxes["status"] == "kept"]
    tiles = []
    for gr in ex_groups.itertuples():
        img = np.asarray(Image.open(yolo / "5class" / "images" / gr.split / f"{gr.image}.png").convert("RGB"))
        bg = kept[(kept["subject"] == gr.subject) & (kept["session"] == gr.session)
                  & (kept["trial_idx"] == gr.trial_idx)]
        tiles.append(draw_boxes(img, bg, title=f"{gr.image} (k={k})  cue: {gr.class_name}"))
    example_grid(tiles, ex["n_per_class"], resolve(f"results/figures/{prefix}_examples.png"))
    qc_kit(groups, boxes, yolo / "5class" / "images", gcfg["qc"])
    log.info("built %d groups (k=%d, z=%.1f) into %s", len(groups), k, gcfg["zscore_k"], yolo)


# --------------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/groups.yaml")
    parser.add_argument("--null-test", action="store_true")
    parser.add_argument("--build", action="store_true", help="build the YOLO datasets with the selected k and z")
    args = parser.parse_args()

    setup_logging()
    gcfg = load_config(args.config)
    acfg = load_config(gcfg["autolabel_config"])
    pcfg = load_config(acfg["preprocess_config"])
    meta = pd.read_csv(resolve(pcfg["paths"]["processed_dir"]) / "metadata.csv", keep_default_na=False)
    meta = meta[meta["split"] != ""]
    if args.null_test:
        res = null_test(gcfg, acfg, pcfg, meta)
        log.info("null test:\n%s", res[["k", "z"] + [c for c in res.columns if c.endswith("_ratio")]].to_string())
    if args.build:
        build(gcfg, acfg, pcfg, meta)


if __name__ == "__main__":
    main()
