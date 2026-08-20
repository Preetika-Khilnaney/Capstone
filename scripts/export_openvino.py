"""Export the YOLO weights to an OpenVINO IR for fast CPU/edge inference.

Produces `<stem>_openvino_model/` (INT8, default) or `<stem>_openvino_model/` (FP16)
next to the weights. INT8 gives ~2x speedup over FP16 on CPU with comparable recall.
Run once, or after swapping the .pt weights.

Usage:
    python scripts/export_openvino.py            # INT8 (recommended)
    python scripts/export_openvino.py --fp16     # FP16 precision
    python scripts/export_openvino.py --fp32     # full precision
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from app.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO to OpenVINO IR")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--int8", action="store_true", default=True,
                       help="INT8 quantized (default, best speed+recall)")
    group.add_argument("--fp16", action="store_true",
                       help="FP16 half precision")
    group.add_argument("--fp32", action="store_true",
                       help="FP32 full precision")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="export input size")
    args = parser.parse_args()

    weights = settings.paths.base_dir / settings.yolo.model_name
    if not weights.exists():
        print(f"Weights not found: {weights}")
        sys.exit(1)

    if args.fp16:
        fmt, half, int8 = "FP16", True, False
    elif args.fp32:
        fmt, half, int8 = "FP32", False, False
    else:
        fmt, half, int8 = "INT8", False, True

    print(f"Exporting {weights} to OpenVINO ({fmt}, imgsz={args.imgsz})...")
    export_kwargs = {"format": "openvino", "imgsz": args.imgsz}
    if half:
        export_kwargs["half"] = True
    if int8:
        export_kwargs["int8"] = True
    out = YOLO(str(weights)).export(**export_kwargs)
    print(f"Done: {out}")
    print(f"Set yolo.backend='openvino' (the default) to use it. Expected dir: "
          f"{settings.yolo.openvino_model_dir}")


if __name__ == "__main__":
    main()
