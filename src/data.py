"""GDF loading, preprocessing, time-frequency maps, and image rendering.

Pipeline per file (DATA §5): drop EOG -> bandpass 4-40 Hz -> epoch around cues ->
small Laplacian (5 channels) -> Morlet power -> crop -> baseline % change -> smoothing.

Usage:
    python -m src.data --config configs/preprocess.yaml [--subjects A01] [--sessions T]
    python -m src.data --config configs/preprocess.yaml --grand-average-only
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib
import mne
import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

from src.utils.coords import F_MAX, F_MIN, IMG, MI_CLASSES, PANEL_H, PANEL_ORDER, T_MAX, T_MIN
from src.utils.io import load_config, resolve, setup_logging

log = logging.getLogger(__name__)

# Standard 10-20 names of the 22 EEG channels, in GDF order (DATA §3).
EEG_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]
GDF_ANCHORS = {0: "EEG-Fz", 7: "EEG-C3", 9: "EEG-Cz", 11: "EEG-C4", 19: "EEG-Pz",
               22: "EOG-left", 23: "EOG-central", 24: "EOG-right"}
CUE_CODES = {"769": 0, "770": 1, "771": 2, "772": 3}  # -> class_id
UNKNOWN_CUE = "783"
REJECT_CODE = "1023"
META_COLUMNS = ["subject", "session", "trial_idx", "cue_onset_s", "class_id", "class_name", "rejected", "split"]


# --------------------------------------------------------------------------- loading


def load_raw(path: Path, sfreq: float) -> mne.io.BaseRaw:
    """Read a GDF file, verify the channel layout, drop EOG, and rename to 10-20 names."""
    raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
    for idx, name in GDF_ANCHORS.items():
        if raw.ch_names[idx] != name:
            raise ValueError(f"{path.name}: expected {name} at index {idx}, got {raw.ch_names[idx]}")
    if raw.info["sfreq"] != sfreq:
        raise ValueError(f"{path.name}: sfreq {raw.info['sfreq']} != {sfreq}")
    if raw.first_samp != 0:
        raise ValueError(f"{path.name}: unexpected first_samp {raw.first_samp}")
    raw.drop_channels(raw.ch_names[22:])
    raw.rename_channels(dict(zip(raw.ch_names, EEG_NAMES)))
    raw.set_channel_types({ch: "eeg" for ch in EEG_NAMES})
    return raw


def load_classlabel(labels_dir: Path, name: str) -> np.ndarray:
    """True labels from `<name>.mat` as class_id (0-3)."""
    return sio.loadmat(labels_dir / f"{name}.mat")["classlabel"].ravel().astype(int) - 1


def find_trials(raw: mne.io.BaseRaw, session: str, class_ids: np.ndarray,
                reject_window: tuple[float, float]) -> pd.DataFrame:
    """One row per cue: onset, class, and rejection flag (1023 within the window around the cue)."""
    desc = np.asarray(raw.annotations.description)
    onset = np.asarray(raw.annotations.onset)
    if session == "T":
        mask = np.isin(desc, list(CUE_CODES))
        cue_class = np.array([CUE_CODES[d] for d in desc[mask]])
    else:
        mask = desc == UNKNOWN_CUE
        cue_class = None
    cues = onset[mask]
    if len(cues) != len(class_ids):
        raise ValueError(f"{len(cues)} cues but {len(class_ids)} labels")
    if cue_class is not None and not np.array_equal(cue_class, class_ids):
        raise ValueError("session T cue codes do not match classlabel")
    rejects = onset[desc == REJECT_CODE]
    lo, hi = reject_window
    rejected = np.array([np.any((rejects >= c + lo) & (rejects <= c + hi)) for c in cues])
    return pd.DataFrame({
        "trial_idx": np.arange(len(cues)),
        "cue_onset_s": cues,
        "class_id": class_ids,
        "class_name": [MI_CLASSES[c] for c in class_ids],
        "rejected": rejected,
    })


# --------------------------------------------------------------------------- signal processing


def extract_epochs(data: np.ndarray, sfreq: float, cue_onsets: np.ndarray,
                   tmin: float, tmax: float) -> np.ndarray:
    """Slice (n_trials, n_channels, n_times) windows [cue + tmin, cue + tmax] from continuous data."""
    start = np.round((cue_onsets + tmin) * sfreq).astype(int)
    n = int(round((tmax - tmin) * sfreq)) + 1
    if start.min() < 0 or start.max() + n > data.shape[1]:
        raise ValueError("epoch window exceeds recording")
    return np.stack([data[:, s:s + n] for s in start])


def add_white_noise(epochs: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian white noise per epoch and channel at a given SNR (CONCEPT §7.3).

    P_noise = P_signal / 10^(SNR / 10), with P_signal the mean power of that epoch and channel.
    """
    p_signal = (epochs ** 2).mean(axis=-1, keepdims=True)
    sd = np.sqrt(p_signal / 10 ** (snr_db / 10))
    return epochs + rng.standard_normal(epochs.shape) * sd


def effective_snr_db(epochs: np.ndarray, ch_names: list[str], cfg: dict[str, Any], snr_db: float, seed: int = 0
                     ) -> float:
    """SNR after the small Laplacian, with the added white noise restricted to the band-pass range (E4 diagnostic).

    The Laplacian removes signal shared by neighbouring channels but sums their independent noise, so the
    effective SNR of the analysed signal is lower than the nominal per-channel SNR.
    """
    from scipy.signal import butter, sosfiltfilt

    noise = add_white_noise(epochs, snr_db, np.random.default_rng(seed)) - epochs
    sos = butter(4, [cfg["bandpass"]["l_freq"], cfg["bandpass"]["h_freq"]], btype="band", fs=cfg["sfreq"], output="sos")
    s = (laplacian(epochs, ch_names, cfg["laplacian"]) ** 2).mean()
    n = (sosfiltfilt(sos, laplacian(noise, ch_names, cfg["laplacian"]), axis=-1) ** 2).mean()
    return float(10 * np.log10(s / n))


def laplacian(epochs: np.ndarray, ch_names: list[str], neighbors: dict[str, list[str]]) -> np.ndarray:
    """Small Laplacian x_c - mean(x_neighbors) for PANEL_ORDER channels -> (n, 5, n_times)."""
    idx = {ch: i for i, ch in enumerate(ch_names)}
    out = [epochs[:, idx[ch]] - epochs[:, [idx[n] for n in neighbors[ch]]].mean(axis=1) for ch in PANEL_ORDER]
    return np.stack(out, axis=1)


def freqs_from_cfg(cfg: dict[str, Any]) -> np.ndarray:
    t = cfg["tfr"]
    return np.arange(t["fmin"], t["fmax"] + t["fstep"] / 2, t["fstep"], dtype=float)


def crop_times(cfg: dict[str, Any]) -> np.ndarray:
    """Sample times (s) of the cropped window [crop.tmin, crop.tmax]."""
    c_tmin, c_tmax = cfg["crop"]["tmin"], cfg["crop"]["tmax"]
    return c_tmin + np.arange(int(round((c_tmax - c_tmin) * cfg["sfreq"])) + 1) / cfg["sfreq"]


def tfr_power(lap: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Morlet power of Laplacian epochs, cropped to [crop.tmin, crop.tmax] -> (n, 5, n_freqs, n_times)."""
    sfreq = cfg["sfreq"]
    freqs = freqs_from_cfg(cfg)
    i0 = int(round((cfg["crop"]["tmin"] - cfg["epoch"]["tmin"]) * sfreq))
    power = mne.time_frequency.tfr_array_morlet(
        lap, sfreq=sfreq, freqs=freqs, n_cycles=freqs / cfg["tfr"]["n_cycles_divisor"], output="power", verbose="ERROR"
    )
    return power[..., i0:i0 + len(crop_times(cfg))]


def percent_change(power: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """ERD/ERS (%) relative to the mean power in the baseline window, along the last axis."""
    t = crop_times(cfg)
    b = (t >= cfg["baseline"]["tmin"]) & (t <= cfg["baseline"]["tmax"])
    base = power[..., b].mean(axis=-1, keepdims=True)
    return (power - base) / base * 100.0


def smooth(pct: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Gaussian smoothing over (freq, time) of (..., n_freqs, n_times) maps."""
    s = cfg["smoothing"]
    sigma = (0,) * (pct.ndim - 2) + (s["sigma_hz"] / cfg["tfr"]["fstep"], s["sigma_s"] * cfg["sfreq"])
    return gaussian_filter(pct, sigma=sigma, mode="nearest")


def erd_maps(lap: np.ndarray, cfg: dict[str, Any], class_ids: np.ndarray | None = None
             ) -> tuple[np.ndarray, np.ndarray | None]:
    """Laplacian epochs -> smoothed single-trial ERD/ERS % maps, float32 (n, 5, n_freqs, n_times).

    If `class_ids` is given, also returns the per-class mean power (4, 5, n_freqs, n_times),
    used for the classical (trial-averaged power) grand average.
    """
    maps, sums = [], None
    step = cfg["tfr_chunk"]
    for s in range(0, len(lap), step):
        power = tfr_power(lap[s:s + step], cfg)
        maps.append(smooth(percent_change(power, cfg), cfg).astype(np.float32))
        if class_ids is not None:
            if sums is None:
                sums = np.zeros((len(MI_CLASSES),) + power.shape[1:])
            for c in range(len(MI_CLASSES)):
                sums[c] += power[class_ids[s:s + step] == c].sum(axis=0)
    class_power = None
    if class_ids is not None:
        counts = np.bincount(class_ids, minlength=len(MI_CLASSES))
        class_power = (sums / counts[:, None, None, None]).astype(np.float32)
    return np.concatenate(maps), class_power


def load_laplacian_epochs(subject: str, session: str, cfg: dict[str, Any], max_trials: int | None = None
                          ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """GDF -> (all cue rows, kept rows, Laplacian epochs of kept trials (n, 5, n_times))."""
    meta, kept, epochs, ch_names = load_bandpassed_epochs(subject, session, cfg, max_trials)
    return meta, kept, laplacian(epochs, ch_names, cfg["laplacian"])


def load_bandpassed_epochs(subject: str, session: str, cfg: dict[str, Any], max_trials: int | None = None
                           ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    """GDF -> (all cue rows, kept rows, band-passed 22-channel epochs of kept trials, channel names)."""
    name = f"{subject}{session}"
    raw = load_raw(resolve(cfg["paths"]["gdf_dir"]) / f"{name}.gdf", cfg["sfreq"])
    meta = find_trials(raw, session, load_classlabel(resolve(cfg["paths"]["labels_dir"]), name),
                       tuple(cfg["rejection_window_s"]))
    raw.filter(cfg["bandpass"]["l_freq"], cfg["bandpass"]["h_freq"], fir_design="firwin", verbose="ERROR")
    data = raw.get_data()
    if np.isnan(data).any():
        raise ValueError(f"{name}: NaN in filtered signal")
    kept = meta[~meta["rejected"]]
    if max_trials is not None:
        kept = kept.head(max_trials)
    epochs = extract_epochs(data, cfg["sfreq"], kept["cue_onset_s"].to_numpy(), cfg["epoch"]["tmin"], cfg["epoch"]["tmax"])
    return meta, kept, epochs, raw.ch_names


def process_file(subject: str, session: str, cfg: dict[str, Any], max_trials: int | None = None) -> pd.DataFrame:
    """Process one GDF file: save one ERD map per kept trial and return its metadata rows."""
    name = f"{subject}{session}"
    meta, kept, lap = load_laplacian_epochs(subject, session, cfg, max_trials)
    maps, class_power = erd_maps(lap, cfg, kept["class_id"].to_numpy())

    processed = resolve(cfg["paths"]["processed_dir"])
    out_dir = processed / "erd" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, m in zip(kept["trial_idx"], maps):
        np.save(out_dir / f"trial_{idx:03d}.npy", m)
    if max_trials is None:
        (processed / "grand_average").mkdir(parents=True, exist_ok=True)
        np.save(processed / "grand_average" / f"class_power_{name}.npy", class_power)

    meta.insert(0, "session", session)
    meta.insert(0, "subject", subject)
    meta["split"] = ""
    log.info("%s: %d cues, %d rejected, %d saved", name, len(meta), meta["rejected"].sum(), len(kept))
    return meta[META_COLUMNS]


def erd_path(cfg: dict[str, Any], subject: str, session: str, trial_idx: int) -> Path:
    return resolve(cfg["paths"]["processed_dir"]) / "erd" / f"{subject}{session}" / f"trial_{trial_idx:03d}.npy"


# --------------------------------------------------------------------------- rendering


def render_erd(erd: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """ERD map (5, n_freqs, n_times) -> RGB uint8 (IMG, IMG, 3) image (DATA §7).

    Each pixel samples the map at its center with bilinear interpolation, so the
    coordinate mapping in `src.utils.coords` holds exactly.
    """
    sfreq, fstep = cfg["sfreq"], cfg["tfr"]["fstep"]
    r = cfg["render"]
    t_px = T_MIN + (np.arange(IMG) + 0.5) / IMG * (T_MAX - T_MIN)
    f_px = F_MIN + (1.0 - (np.arange(PANEL_H) + 0.5) / PANEL_H) * (F_MAX - F_MIN)
    rows, cols = np.meshgrid((f_px - F_MIN) / fstep, (t_px - T_MIN) * sfreq, indexing="ij")
    panels = [map_coordinates(p, [rows, cols], order=1, mode="nearest") for p in erd]
    values = np.clip(np.concatenate(panels, axis=0), r["vmin"], r["vmax"])
    norm = (values - r["vmin"]) / (r["vmax"] - r["vmin"])
    rgba = matplotlib.colormaps[r["cmap"]](norm)
    return (rgba[..., :3] * 255).round().astype(np.uint8)


def save_image(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


# --------------------------------------------------------------------------- grand average


def grand_average(cfg: dict[str, Any], session: str = "T") -> np.ndarray:
    """Classical grand-average ERD/ERS per MI class -> (4, 5, n_freqs, n_times).

    Power is averaged over trials per subject and class, converted to % change versus the
    baseline window, then averaged over subjects. Averaging single-trial % maps instead is
    biased upward (noisy per-trial baseline in the denominator), so it is not used here.
    """
    ga_dir = resolve(cfg["paths"]["processed_dir"]) / "grand_average"
    per_subject = [percent_change(np.load(ga_dir / f"class_power_{s}{session}.npy").astype(np.float64), cfg)
                   for s in cfg["subjects"]]
    ga = np.mean(per_subject, axis=0).astype(np.float32)
    np.save(ga_dir / f"grand_average_{session}.npy", ga)
    return ga


def plot_grand_average(ga: np.ndarray, out_path: Path, vlim: float = 30.0) -> None:
    """Grid figure: rows = MI classes, columns = panels (C5..C6)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(MI_CLASSES), len(PANEL_ORDER), figsize=(16, 10), sharex=True, sharey=True,
                             constrained_layout=True)
    for i, cls in enumerate(MI_CLASSES):
        for j, ch in enumerate(PANEL_ORDER):
            ax = axes[i, j]
            im = ax.imshow(ga[i, j], origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                           extent=[T_MIN, T_MAX, F_MIN - 0.5, F_MAX + 0.5])
            for t in (0.0, 4.0):
                ax.axvline(t, color="k", lw=0.6, ls="--")
            if i == 0:
                ax.set_title(ch)
            if j == 0:
                ax.set_ylabel(f"{cls}\nfreq (Hz)")
            if i == len(MI_CLASSES) - 1:
                ax.set_xlabel("time from cue (s)")
    fig.colorbar(im, ax=axes, shrink=0.6, label="ERD/ERS (%)")
    fig.suptitle("Grand-average ERD/ERS per class (session T, trial-averaged power, mean of 9 subjects)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- CLI


def update_metadata(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    """Replace rows of the processed (subject, session) pairs in metadata.csv."""
    if path.exists():
        old = pd.read_csv(path, dtype={"split": str}, keep_default_na=False)
        keys = set(zip(new["subject"], new["session"]))
        old = old[[k not in keys for k in zip(old["subject"], old["session"])]]
        new = pd.concat([old, new], ignore_index=True)
    order = new["session"].map({"T": 0, "E": 1})
    new = new.assign(_order=order).sort_values(["subject", "_order", "trial_idx"]).drop(columns="_order")
    new.to_csv(path, index=False)
    return new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/preprocess.yaml")
    parser.add_argument("--subjects", nargs="*", help="default: all from config")
    parser.add_argument("--sessions", nargs="*", help="default: all from config")
    parser.add_argument("--max-trials", type=int, help="smoke test: limit kept trials per file")
    parser.add_argument("--grand-average-only", action="store_true")
    parser.add_argument("--no-grand-average", action="store_true")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    meta_path = resolve(cfg["paths"]["processed_dir"]) / "metadata.csv"
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.grand_average_only:
        rows = [process_file(s, ses, cfg, args.max_trials)
                for s in (args.subjects or cfg["subjects"]) for ses in (args.sessions or cfg["sessions"])]
        update_metadata(meta_path, pd.concat(rows, ignore_index=True))

    if not args.no_grand_average and args.max_trials is None:
        ga = grand_average(cfg, "T")
        plot_grand_average(ga, resolve("results/figures/grand_average_erd.png"))
        log.info("grand average saved")


if __name__ == "__main__":
    main()
