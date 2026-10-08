"""Common evaluator: detection, domain, and decoding metrics from one prediction format.

Every method (YOLO, Faster R-CNN, threshold baseline) writes results/predictions/<run_id>/<split>.json
(CONVENTIONS §4, boxes in absolute 640 x 640 pixels); this module only reads that format.

Usage:
    python -m src.evaluate --run e1_baseline --split val --predict   # predict with best weights, then evaluate
    python -m src.evaluate --run threshold --split val               # evaluate existing predictions
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.coords import MI_CLASSES, xyxy_to_box, yolo_to_xyxy
from src.utils.io import load_config, resolve, setup_logging

log = logging.getLogger(__name__)

CONF_THR, IOU_THR = 0.25, 0.5  # CONCEPT §7.1 (precision / recall / F1, confusion, domain, decoding)
ALL_RUNS_COLUMNS = ["run_id", "experiment", "model", "n_classes", "split", "seed", "epochs_trained", "map50",
                    "map50_95", "precision", "recall", "f1", "onset_err_ms", "offset_err_ms", "flow_err_hz",
                    "fhigh_err_hz", "train_time_min", "notes"]

Preds = dict[str, dict[str, np.ndarray]]  # image stem -> {"boxes": (n, 4), "scores": (n,), "labels": (n,)}


# --------------------------------------------------------------------------- ground truth and predictions


def split_images(data_yaml: str | Path, split: str, limit: int | None = None) -> list[Path]:
    data = load_config(data_yaml)
    return sorted((resolve(data["path"]) / data[split]).glob("*.png"))[:limit]


def ground_truth(data_yaml: str | Path, split: str, limit: int | None = None) -> Preds:
    """YOLO label files of a split -> absolute xyxy boxes and class ids (no scores)."""
    out = {}
    for img in split_images(data_yaml, split, limit):
        lbl = Path(str(img.parent).replace("images", "labels")) / f"{img.stem}.txt"
        rows = [line.split() for line in lbl.read_text().splitlines() if line.strip()]
        out[img.stem] = {"boxes": np.array([yolo_to_xyxy(*map(float, r[1:])) for r in rows], dtype=float).reshape(-1, 4),
                         "labels": np.array([int(r[0]) for r in rows], dtype=int)}
    return out


def group_info() -> pd.DataFrame:
    """Group table (subject, session, group id, cue) indexed by image stem."""
    gcfg = load_config("configs/groups.yaml")
    pcfg = load_config(load_config(gcfg["autolabel_config"])["preprocess_config"])
    path = resolve(pcfg["paths"]["processed_dir"]) / "groups" / f"groups_k{gcfg['k']}.csv"
    return pd.read_csv(path).set_index("image")


def save_predictions(run_id: str, split: str, preds: Preds, classes: list[str]) -> Path:
    info = group_info()
    images = []
    for stem, p in sorted(preds.items()):
        g = info.loc[stem]
        images.append({"image": f"{stem}.png", "subject": g.subject, "session": g.session,
                       "trial_idx": int(g.trial_idx), "true_class": int(g.class_id),
                       "boxes": np.round(p["boxes"], 2).tolist(), "scores": np.round(p["scores"], 5).tolist(),
                       "labels": p["labels"].astype(int).tolist()})
    path = resolve("results/predictions") / run_id / f"{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_id": run_id, "split": split, "classes": classes, "images": images}),
                    encoding="utf-8")
    return path


def load_predictions(run_id: str, split: str) -> tuple[Preds, dict[str, Any]]:
    d = json.loads((resolve("results/predictions") / run_id / f"{split}.json").read_text(encoding="utf-8"))
    preds = {Path(i["image"]).stem: {"boxes": np.array(i["boxes"], dtype=float).reshape(-1, 4),
                                     "scores": np.array(i["scores"], dtype=float),
                                     "labels": np.array(i["labels"], dtype=int)} for i in d["images"]}
    return preds, d


def predict_yolo(weights: str | Path, data_yaml: str | Path, split: str, imgsz: int, limit: int | None = None) -> Preds:
    """YOLO predictions with a low confidence floor (0.001) so mAP sees the full ranking."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = {}
    paths = [str(p) for p in split_images(data_yaml, split, limit)]
    for r in model.predict(source=paths, imgsz=imgsz, conf=0.001, iou=0.7, max_det=100, batch=16, stream=True,
                           verbose=False):
        b = r.boxes
        out[Path(r.path).stem] = {"boxes": b.xyxy.cpu().numpy().astype(float), "scores": b.conf.cpu().numpy(),
                                  "labels": b.cls.cpu().numpy().astype(int)}
    return out


def predict_frcnn_model(model, data_yaml: str | Path, split: str, device, limit: int | None = None) -> Preds:
    """Faster R-CNN predictions (labels shifted back to 0-based)."""
    import torch
    from torch.utils.data import DataLoader

    from src.train import YoloDetectionDataset, collate

    model.eval()
    out = {}
    loader = DataLoader(YoloDetectionDataset(data_yaml, split, limit), batch_size=4, collate_fn=collate)
    with torch.no_grad():
        for imgs, _, stems in loader:
            for stem, d in zip(stems, model([i.to(device) for i in imgs])):
                out[stem] = {"boxes": d["boxes"].cpu().numpy().astype(float), "scores": d["scores"].cpu().numpy(),
                             "labels": d["labels"].cpu().numpy().astype(int) - 1}
    return out


def predict_frcnn(ckpt: str | Path, data_yaml: str | Path, split: str) -> Preds:
    import torch

    from src.train import build_frcnn

    state = torch.load(ckpt, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_frcnn(len(state["names"]), state["imgsz"], pretrained=False)
    model.load_state_dict(state["model"])
    return predict_frcnn_model(model.to(device), data_yaml, split, device)


# --------------------------------------------------------------------------- detection metrics


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def match(gt: dict[str, np.ndarray], pred: dict[str, np.ndarray], conf: float = CONF_THR, iou: float = IOU_THR,
          class_aware: bool = True) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """Greedy matching by descending score: (matched (pred, gt) index pairs, unmatched preds, unmatched gt)."""
    keep = np.flatnonzero(pred["scores"] >= conf)
    keep = keep[np.argsort(-pred["scores"][keep], kind="stable")]
    ious = iou_matrix(pred["boxes"], gt["boxes"])
    used, pairs, fp = set(), [], []
    for i in keep:
        cand = [(ious[i, j], j) for j in range(len(gt["labels"]))
                if j not in used and ious[i, j] >= iou and (not class_aware or gt["labels"][j] == pred["labels"][i])]
        if cand:
            j = max(cand)[1]
            used.add(j)
            pairs.append((int(i), j))
        else:
            fp.append(int(i))
    return pairs, np.array(fp, dtype=int), np.array([j for j in range(len(gt["labels"])) if j not in used], dtype=int)


def detection_metrics(gt: Preds, preds: Preds, n_classes: int) -> dict[str, Any]:
    """mAP (torchmetrics, COCO IoUs), per-class AP, P / R / F1 at CONF_THR and IOU_THR, confusion matrix."""
    import torch
    from torchmetrics.detection import MeanAveragePrecision

    empty = {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, dtype=int)}
    stems = sorted(gt)
    tm_pred = [{"boxes": torch.tensor(preds.get(s, empty)["boxes"], dtype=torch.float32).reshape(-1, 4),
                "scores": torch.tensor(preds.get(s, empty)["scores"], dtype=torch.float32),
                "labels": torch.tensor(preds.get(s, empty)["labels"], dtype=torch.int64)} for s in stems]
    tm_gt = [{"boxes": torch.tensor(gt[s]["boxes"], dtype=torch.float32).reshape(-1, 4),
              "labels": torch.tensor(gt[s]["labels"], dtype=torch.int64)} for s in stems]
    full = MeanAveragePrecision(iou_type="bbox", class_metrics=True, max_detection_thresholds=[1, 10, 100])
    full.update(tm_pred, tm_gt)
    r = full.compute()
    at50 = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5], class_metrics=True)
    at50.update(tm_pred, tm_gt)
    r50 = at50.compute()

    def per_class(res) -> dict[int, float]:
        classes = res["classes"].reshape(-1).tolist()
        vals = res["map_per_class"].reshape(-1).tolist()
        return {int(c): float(v) for c, v in zip(classes, vals)}

    ap50_95, ap50 = per_class(r), per_class(r50)
    tp, fp, fn = np.zeros(n_classes), np.zeros(n_classes), np.zeros(n_classes)
    cm = np.zeros((n_classes + 1, n_classes + 1), dtype=int)  # rows: true (last = background), cols: predicted
    for s in stems:
        p, g = preds.get(s, empty), gt[s]
        pairs, fps, fns = match(g, p)
        for i, j in pairs:
            tp[g["labels"][j]] += 1
        for i in fps:
            fp[p["labels"][i]] += 1
        for j in fns:
            fn[g["labels"][j]] += 1
        pairs, fps, fns = match(g, p, class_aware=False)
        for i, j in pairs:
            cm[g["labels"][j], p["labels"][i]] += 1
        for i in fps:
            cm[n_classes, p["labels"][i]] += 1
        for j in fns:
            cm[g["labels"][j], n_classes] += 1
    prec = lambda a, b: float(a / (a + b)) if a + b else 0.0
    P, R = prec(tp.sum(), fp.sum()), prec(tp.sum(), fn.sum())
    return {"map50": float(r["map_50"]), "map50_95": float(r["map"]),
            "precision": P, "recall": R, "f1": 2 * P * R / (P + R) if P + R else 0.0,
            "per_class": [{"class_id": c, "ap50": ap50.get(c, float("nan")), "ap50_95": ap50_95.get(c, float("nan")),
                           "precision": prec(tp[c], fp[c]), "recall": prec(tp[c], fn[c]),
                           "tp": int(tp[c]), "fp": int(fp[c]), "fn": int(fn[c])} for c in range(n_classes)],
            "confusion": cm}


def domain_metrics(gt: Preds, preds: Preds) -> dict[str, float]:
    """Mean absolute onset / offset (ms) and lower / upper frequency (Hz) errors of matched boxes."""
    err = []
    for s, g in gt.items():
        p = preds.get(s)
        if p is None:
            continue
        for i, j in match(g, p)[0]:
            _, pt0, pt1, pf0, pf1 = xyxy_to_box(*p["boxes"][i])
            _, gt0, gt1, gf0, gf1 = xyxy_to_box(*g["boxes"][j])
            err.append((abs(pt0 - gt0) * 1000, abs(pt1 - gt1) * 1000, abs(pf0 - gf0), abs(pf1 - gf1)))
    if not err:
        return {k: float("nan") for k in ("onset_err_ms", "offset_err_ms", "flow_err_hz", "fhigh_err_hz")}
    e = np.mean(err, axis=0)
    return {"onset_err_ms": e[0], "offset_err_ms": e[1], "flow_err_hz": e[2], "fhigh_err_hz": e[3], "n_matched": len(err)}


# --------------------------------------------------------------------------- decoding (E5)


def decode_groups(preds: Preds, conf: float = CONF_THR) -> pd.DataFrame:
    """Per image: predicted cue class = ERD class (0-3) with the highest summed confidence (CONCEPT §6.4)."""
    info = group_info()
    rows = []
    for stem, p in preds.items():
        sel = (p["scores"] >= conf) & (p["labels"] < len(MI_CLASSES))
        sums = np.bincount(p["labels"][sel], weights=p["scores"][sel], minlength=len(MI_CLASSES))
        g = info.loc[stem]
        rows.append({"image": stem, "subject": g.subject, "true_class": int(g.class_id),
                     "pred_class": int(np.argmax(sums)) if sel.any() else -1})
    return pd.DataFrame(rows)


def decoding_metrics(dec: pd.DataFrame) -> pd.DataFrame:
    """Accuracy, Cohen's kappa ((acc - 0.25) / 0.75), and no-decision rate, per subject and overall.

    `dec` needs subject, true_class, pred_class (-1 = no decision, counted as wrong).
    """
    def agg(d: pd.DataFrame) -> pd.Series:
        acc = float((d["pred_class"] == d["true_class"]).mean())
        return pd.Series({"n_groups": len(d), "accuracy": acc, "kappa": (acc - 0.25) / 0.75,
                          "no_decision_pct": 100 * float((d["pred_class"] < 0).mean())})
    per = dec.groupby("subject").apply(agg, include_groups=False)
    per.loc["all"] = agg(dec)
    return per.reset_index()


# --------------------------------------------------------------------------- tables


def upsert(path: Path, row: dict[str, Any], keys: list[str], columns: list[str] | None = None) -> None:
    """Insert or replace the row(s) identified by `keys` in a CSV table."""
    new = pd.DataFrame([row] if isinstance(row, dict) else row)
    if path.exists():
        old = pd.read_csv(path)
        mask = np.ones(len(old), dtype=bool)
        for k in keys:
            mask &= old[k].astype(str).isin(new[k].astype(str).unique())
        new = pd.concat([old[~mask], new], ignore_index=True)
    if columns:
        new = new.reindex(columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)


def epochs_trained(run_dir: Path) -> float:
    if (run_dir / "results.csv").exists():
        return float(len(pd.read_csv(run_dir / "results.csv")))
    if (run_dir / "log.csv").exists():
        return float(len(pd.read_csv(run_dir / "log.csv")))
    return float("nan")


def evaluate_run(run_id: str, split: str, predict: bool, run_cfg: dict[str, Any], notes: str = "") -> dict[str, Any]:
    """Predict (optional), evaluate, and update results/tables (all_runs, per_class_ap, confusion, decoding)."""
    data_yaml = resolve(run_cfg["data"])
    classes = list(load_config(data_yaml)["names"].values())
    run_dir = resolve("results/runs") / run_id
    if predict:
        if run_cfg["model"] == "frcnn":
            preds = predict_frcnn(run_dir / "best.pt", data_yaml, split)
        else:
            preds = predict_yolo(run_dir / "weights" / "best.pt", data_yaml, split, run_cfg["train"]["imgsz"])
        save_predictions(run_id, split, preds, classes)
    preds, _ = load_predictions(run_id, split)
    gt = ground_truth(data_yaml, split)
    det = detection_metrics(gt, preds, len(classes))
    dom = domain_metrics(gt, preds)
    tables = resolve("results/tables")
    ttime = run_dir / "train_time_min.txt"
    row = {"run_id": run_id, "experiment": run_cfg.get("experiment", ""), "model": run_cfg["model"],
           "n_classes": len(classes), "split": split, "seed": run_cfg.get("seed", ""),
           "epochs_trained": epochs_trained(run_dir),
           **{k: det[k] for k in ("map50", "map50_95", "precision", "recall", "f1")},
           **{k: dom[k] for k in ("onset_err_ms", "offset_err_ms", "flow_err_hz", "fhigh_err_hz")},
           "train_time_min": float(ttime.read_text()) if ttime.exists() else float("nan"), "notes": notes}
    pc = [{"run_id": run_id, "split": split, "class_name": classes[c["class_id"]], **c} for c in det["per_class"]]
    labels = classes + ["background"]
    cm = pd.DataFrame(det["confusion"], index=pd.Index(labels, name="true"), columns=labels)
    dec = None
    if len(classes) == len(MI_CLASSES) + 1:  # 5-class setting: decoding per group
        dec = decoding_metrics(decode_groups(preds))
        dec.insert(0, "split", split)
        dec.insert(0, "run_id", run_id)
    log.info("%s / %s: mAP50 %.3f, mAP50-95 %.3f, P %.3f, R %.3f", run_id, split, det["map50"], det["map50_95"],
             det["precision"], det["recall"])
    if run_id.startswith("smoke_"):  # smoke runs are never reported (CONVENTIONS §5)
        log.info("smoke run: results tables not updated\n%s\n%s", pd.DataFrame(pc).to_string(), cm.to_string())
        return {**row, "per_class": det["per_class"], "confusion": cm, "decoding": dec}
    upsert(tables / "all_runs.csv", row, ["run_id", "split"], ALL_RUNS_COLUMNS)
    upsert(tables / "per_class_ap.csv", pc, ["run_id", "split"])
    (tables / "confusion").mkdir(parents=True, exist_ok=True)
    cm.to_csv(tables / "confusion" / f"{run_id}_{split}.csv")
    if dec is not None:
        upsert(tables / "decoding_by_subject.csv", dec, ["run_id", "split"])
    return {**row, "per_class": det["per_class"], "confusion": cm, "decoding": dec}


E1_RUNS = ["e1_baseline", "e1_lr1e-3", "e1_batch32", "e1_imgsz512", "e1_ep50", "e1_adamw"]


def e1_table() -> pd.DataFrame:
    """E1 table (validation) from all_runs.csv; best run = highest mAP@0.5, ties broken by mAP@0.5:0.95."""
    runs = pd.read_csv(resolve("results/tables/all_runs.csv"))
    runs = runs[(runs["split"] == "val") & runs["run_id"].isin(E1_RUNS)].set_index("run_id")
    missing = [r for r in E1_RUNS if r not in runs.index]
    if missing:
        log.warning("E1 runs not evaluated yet: %s", missing)
    t = runs.reindex([r for r in E1_RUNS if r in runs.index])[
        ["map50", "map50_95", "precision", "recall", "f1", "epochs_trained", "train_time_min", "notes"]]
    t.insert(0, "change", [load_config(f"configs/experiments/{r}.yaml").get("change", "") for r in t.index])
    best = t.sort_values(["map50", "map50_95"], ascending=False).index[0]
    t["selected"] = t.index == best
    t = t.reset_index()
    t.to_csv(resolve("results/tables/e1_hyperparams.csv"), index=False)
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="run_id (configs/experiments/<run_id>.yaml)")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--predict", action="store_true", help="run inference with the run's best weights first")
    parser.add_argument("--config", help="experiment config (default: configs/experiments/<run>.yaml)")
    parser.add_argument("--notes", default="")
    parser.add_argument("--table", choices=["e1"], help="derive an experiment table from all_runs.csv")
    args = parser.parse_args()

    setup_logging()
    if args.run:
        cfg = load_config(args.config or f"configs/experiments/{args.run}.yaml")
        evaluate_run(args.run, args.split, args.predict, cfg, args.notes)
    if args.table == "e1":
        log.info("E1 (val):\n%s", e1_table().to_string(index=False))


if __name__ == "__main__":
    main()
