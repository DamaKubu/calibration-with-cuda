"""Legacy wrapper for camera intrinsic calibration.

This file used to run a one-off fisheye calibration. It now delegates to the
new CLI in `calibrate_camera.py` while keeping compatible defaults.
"""

import argparse
import sys

from calibrate_camera import run_calibration_cli


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate camera (legacy wrapper).")
    parser.add_argument("--images", default="calibration_images/*.png")
    parser.add_argument("--pattern", default="8x5", help="e.g. 8x5")
    parser.add_argument("--square-mm", type=float, default=65.0)
    parser.add_argument("--quality", choices=["fast", "best"], default="best")
    parser.add_argument("--model", choices=["standard", "fisheye"], default="standard")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-remove", type=int, default=10)
    parser.add_argument("--max-per-image-error", type=float, default=1.5)
    args = parser.parse_args(argv)
    return run_calibration_cli(
        images_glob=args.images,
        pattern=args.pattern,
        square_mm=args.square_mm,
        quality=args.quality,
        model=args.model,
        use_gpu=args.use_gpu,
        output=args.output,
        max_remove=args.max_remove,
        max_per_image_error=args.max_per_image_error,
    )


if __name__ == "__main__":
    raise SystemExit(main())
