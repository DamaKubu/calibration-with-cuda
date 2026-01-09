"""Legacy wrapper for camera intrinsic calibration.

This file used to run a one-off fisheye calibration. It now delegates to the
new CLI in `calibrate_camera.py` while keeping compatible defaults.
"""

from __future__ import annotations

import argparse
import sys

from calibrate_camera import run_calibration_cli


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate camera intrinsics from chessboard images (legacy wrapper)."
    )
    parser.add_argument(
        "--images",
        default="calibration_images/*.jpg",
        help="Glob for input images (default: calibration_images/*.jpg)",
    )
    parser.add_argument(
        "--pattern",
        default="8x5",
        help="Chessboard inner corners as COLSxROWS (default: 8x5)",
    )
    parser.add_argument(
        "--square-mm",
        type=float,
        default=65.0,
        help="Square size in millimeters (default: 65)",
    )
    parser.add_argument(
        "--quality",
        choices=["fast", "best"],
        default="best",
        help="Detection/calibration tradeoff (default: best)",
    )
    parser.add_argument(
        "--model",
        choices=["standard", "fisheye"],
        default="standard",
        help="Lens model (default: standard for barrel distortion)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use CUDA preprocessing if available (corner detection remains CPU)",
    )
    parser.add_argument(
        "--out",
        default="cam_intrinsics.yaml",
        help="Output YAML path (default: cam_intrinsics.yaml)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return run_calibration_cli(
        images_glob=args.images,
        pattern=args.pattern,
        square_mm=args.square_mm,
        quality=args.quality,
        model=args.model,
        use_gpu=args.use_gpu,
        out_path=args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
