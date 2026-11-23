"""
Fine-tune YOLOv8 on a pothole dataset.

Usage (from the training/ folder):
    pip install -r requirements.txt
    python train_yolo.py --data pothole_dataset/data.yaml --epochs 80

On a laptop without an NVIDIA GPU this is very slow. Use Google Colab (free GPU) -
see dataset_setup.md for copy-paste steps.

When training finishes, the best weights are copied to ../detection/best.pt so
that detect_image.py / detect_pothole.py can use them straight away.
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 for pothole detection.")
    p.add_argument("--data", required=True, help="Path to the dataset's data.yaml")
    p.add_argument("--model", default="yolov8n.pt",
                   help="Starting checkpoint (yolov8n.pt = smallest/fastest; yolov8s.pt = more accurate)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="Lower this (e.g. 8) if you run out of GPU memory")
    p.add_argument("--device", default=None, help="'0' for first GPU, 'cpu' to force CPU. Default: auto")
    p.add_argument("--name", default="pothole", help="Run name (results go to runs/detect/<name>)")
    p.add_argument("--export-to", default="../detection/best.pt",
                   help="Where to copy the best weights when training finishes")
    args = p.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"data.yaml not found: {args.data}\nSee dataset_setup.md to get a dataset first.")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=20,        # stop early if validation stops improving
    )

    best = Path(model.trainer.best)
    print(f"\nBest weights: {best}")

    metrics = model.val()
    print(f"Validation mAP50: {metrics.box.map50:.3f}   mAP50-95: {metrics.box.map:.3f}")

    dest = Path(args.export_to)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, dest)
    print(f"Copied weights to {dest.resolve()}")


if __name__ == "__main__":
    main()