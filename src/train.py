"""Training entry point for YOLO11 and Faster R-CNN runs.

Usage:
    python -m src.train --config configs/experiments/e1_baseline.yaml
    python -m src.train --config configs/experiments/e1_baseline.yaml --smoke   # 1 epoch, small fraction

Outputs go to results/runs/<run_id>/ (smoke runs: results/runs/smoke_<run_id>/).
"""

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.utils.io import load_config, resolve, seed_everything, setup_logging

log = logging.getLogger(__name__)

RUNS_DIR = "results/runs"

# Every Ultralytics augmentation is switched off (CONCEPT §6.2): pixel position and color carry physical meaning.
YOLO_NO_AUG = dict(
    fliplr=0.0, flipud=0.0, mosaic=0.0, close_mosaic=0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0.0, scale=0.0, degrees=0.0, shear=0.0, perspective=0.0,
    erasing=0.0, bgr=0.0,
)
YOLO_WEIGHTS = {"yolo11n": "yolo11n.pt", "yolo11s": "yolo11s.pt"}


def resolved_data_yaml(data_cfg: str, out_dir: Path) -> Path:
    """Copy a data YAML with its dataset `path` made absolute (Ultralytics resolves relative paths elsewhere)."""
    data = load_config(data_cfg)
    data["path"] = str(resolve(data["path"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def run_name(cfg: dict[str, Any], smoke: bool) -> str:
    return f"smoke_{cfg['run_id']}" if smoke else cfg["run_id"]


# --------------------------------------------------------------------------- YOLO


def train_yolo(cfg: dict[str, Any], smoke: bool, workers: int) -> Path:
    """Train a YOLO11 model; returns the run directory."""
    from ultralytics import YOLO

    name = run_name(cfg, smoke)
    run_dir = resolve(RUNS_DIR) / name
    if run_dir.exists() and any(run_dir.glob("weights/*.pt")):
        raise FileExistsError(f"{run_dir} already has weights; remove it or pick another run_id")
    args = dict(cfg["train"])
    if smoke:
        args.update(epochs=1, fraction=0.05, batch=8, patience=0)
    model = YOLO(YOLO_WEIGHTS[cfg["model"]])
    model.train(data=str(resolved_data_yaml(cfg["data"], run_dir)), seed=cfg["seed"], deterministic=True,
                workers=workers, project=str(resolve(RUNS_DIR)), name=name, exist_ok=True, plots=True,
                **YOLO_NO_AUG, **args)
    return run_dir


# --------------------------------------------------------------------------- Faster R-CNN


class YoloDetectionDataset:
    """Images + YOLO labels -> (float image tensor [3, H, W], target dict with absolute xyxy boxes, labels 1..C)."""

    def __init__(self, data_yaml: Path, split: str, limit: int | None = None):
        data = load_config(data_yaml)
        root = Path(data["path"])
        self.images = sorted((root / data[split]).glob("*.png"))[:limit]
        self.label_dir = root / data[split].replace("images", "labels")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        import torch
        from PIL import Image

        from src.utils.coords import yolo_to_xyxy

        img = np.asarray(Image.open(self.images[i]).convert("RGB"), dtype=np.float32) / 255.0
        rows = [line.split() for line in (self.label_dir / f"{self.images[i].stem}.txt").read_text().splitlines()
                if line.strip()]
        boxes = [yolo_to_xyxy(*map(float, r[1:])) for r in rows]
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.tensor([int(r[0]) + 1 for r in rows], dtype=torch.int64)}
        return torch.from_numpy(img).permute(2, 0, 1), target, self.images[i].stem


def collate(batch):
    return tuple(zip(*batch))


def build_frcnn(num_classes: int, imgsz: int, pretrained: bool = True):
    """torchvision Faster R-CNN ResNet50-FPN v2 (COCO weights) with a new box predictor; no resize beyond imgsz.

    v2 (improved torchvision recipe) is used instead of v1 because its COCO weights were already cached
    (user decision, Phase 3).
    """
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None, min_size=imgsz, max_size=imgsz,
                                    box_score_thresh=0.001, box_detections_per_img=100)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def train_frcnn(cfg: dict[str, Any], smoke: bool, workers: int) -> Path:
    """Train Faster R-CNN; keeps the checkpoint with the best val mAP@0.5 (common evaluator)."""
    import torch
    from torch.utils.data import DataLoader

    from src.evaluate import detection_metrics, ground_truth, predict_frcnn_model

    name = run_name(cfg, smoke)
    run_dir = resolve(RUNS_DIR) / name
    if (run_dir / "best.pt").exists():
        raise FileExistsError(f"{run_dir} already has best.pt; remove it or pick another run_id")
    data_yaml = resolved_data_yaml(cfg["data"], run_dir)
    names = load_config(data_yaml)["names"]
    tr = dict(cfg["train"])
    limit = 50 if smoke else None
    epochs = 1 if smoke else tr["epochs"]
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = YoloDetectionDataset(data_yaml, "train", limit)
    loader = DataLoader(train_ds, batch_size=tr["batch"], shuffle=True, num_workers=workers, collate_fn=collate,
                        generator=torch.Generator().manual_seed(cfg["seed"]))
    model = build_frcnn(len(names), tr["imgsz"]).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=tr["lr"], momentum=tr["momentum"], weight_decay=tr["weight_decay"])
    gt_val = ground_truth(data_yaml, "val", limit=limit)

    best, log_rows = -1.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for imgs, targets, _ in loader:
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(imgs, targets).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "time_s": round(time.time() - t0, 1)}
        if epoch % tr.get("eval_every", 1) == 0 or epoch == epochs:
            preds = predict_frcnn_model(model, data_yaml, "val", device, limit=limit)
            m = detection_metrics(gt_val, preds, len(names))
            row.update(val_map50=m["map50"], val_map50_95=m["map50_95"])
            if m["map50"] > best:
                best = m["map50"]
                torch.save({"model": model.state_dict(), "names": names, "imgsz": tr["imgsz"], "epoch": epoch},
                           run_dir / "best.pt")
        torch.save({"model": model.state_dict(), "names": names, "imgsz": tr["imgsz"], "epoch": epoch},
                   run_dir / "last.pt")
        log_rows.append(row)
        log.info("epoch %d: %s", epoch, row)
        with (run_dir / "log.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "time_s", "val_map50", "val_map50_95"])
            w.writeheader()
            w.writerows(log_rows)
    return run_dir


# --------------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true", help="1 epoch on a small subset, run name smoke_<run_id>")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    t0 = time.time()
    if cfg["model"] in YOLO_WEIGHTS:
        run_dir = train_yolo(cfg, args.smoke, args.workers)
    elif cfg["model"] == "frcnn":
        run_dir = train_frcnn(cfg, args.smoke, args.workers)
    else:
        raise ValueError(f"unknown model {cfg['model']}")
    minutes = (time.time() - t0) / 60
    (run_dir / "train_time_min.txt").write_text(f"{minutes:.1f}\n", encoding="utf-8")
    log.info("finished %s in %.1f min -> %s", cfg["run_id"], minutes, run_dir)


if __name__ == "__main__":
    main()
