from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CaptureConfig:
    camera_id: int
    out_dir: str
    count: int
    pattern_cols: int
    pattern_rows: int
    quality: str
    use_gpu: bool
    min_interval_ms: int
    detect_scale: float
    auto: bool

    @property
    def pattern_size(self) -> tuple[int, int]:
        return (self.pattern_cols, self.pattern_rows)


def _parse_pattern(pattern: str) -> tuple[int, int]:
    s = pattern.lower().replace(" ", "")
    if "x" not in s:
        raise ValueError("--pattern must look like 8x5")
    a, b = s.split("x", 1)
    return int(a), int(b)


def _cuda_available() -> bool:
    try:
        return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


def _to_gray(img_bgr: np.ndarray, use_gpu: bool) -> np.ndarray:
    if not use_gpu or not _cuda_available():
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    try:
        gpu = cv2.cuda_GpuMat()
        gpu.upload(img_bgr)
        gpu_gray = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2GRAY)
        return gpu_gray.download()
    except Exception:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _resize_gray(gray: np.ndarray, scale: float, use_gpu: bool) -> np.ndarray:
    if scale == 1.0:
        return gray

    if not use_gpu or not _cuda_available():
        return cv2.resize(gray, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    try:
        gpu = cv2.cuda_GpuMat()
        gpu.upload(gray)
        gpu_resized = cv2.cuda.resize(gpu, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return gpu_resized.download()
    except Exception:
        return cv2.resize(gray, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _find_corners(gray: np.ndarray, pattern_size: tuple[int, int], quality: str) -> tuple[bool, np.ndarray | None]:
    cols, rows = pattern_size
    if quality == "best" and hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
            flags |= cv2.CALIB_CB_EXHAUSTIVE
        if hasattr(cv2, "CALIB_CB_ACCURACY"):
            flags |= cv2.CALIB_CB_ACCURACY
        ret, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags)
        if ret:
            return True, corners.astype(np.float32)
        return False, None

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not ret:
        return False, None
    return True, corners.astype(np.float32)


def _should_accept(prev_corners: np.ndarray | None, new_corners: np.ndarray, min_mean_shift_px: float) -> bool:
    if prev_corners is None:
        return True
    a = prev_corners.reshape(-1, 2)
    b = new_corners.reshape(-1, 2)
    mean_shift = float(np.mean(np.linalg.norm(a - b, axis=1)))
    return mean_shift >= float(min_mean_shift_px)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture calibration images from a camera.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for OpenCV VideoCapture")
    parser.add_argument("--out", default="calibration_images", help="Output directory")
    parser.add_argument("--count", type=int, default=30, help="Number of images to capture")
    parser.add_argument("--pattern", default="8x5", help="Chessboard inner corners COLSxROWS")
    parser.add_argument("--quality", choices=["fast", "best"], default="best")
    parser.add_argument("--use-gpu", action="store_true", help="Use CUDA preprocessing if available")
    parser.add_argument("--auto", action="store_true", help="Auto-capture when a good pattern is seen")
    parser.add_argument("--interval-ms", type=int, default=800, help="Minimum time between auto captures")
    args = parser.parse_args(argv)

    cols, rows = _parse_pattern(args.pattern)
    detect_scale = 0.5 if args.quality == "fast" else 1.0

    cfg = CaptureConfig(
        camera_id=args.camera,
        out_dir=args.out,
        count=args.count,
        pattern_cols=cols,
        pattern_rows=rows,
        quality=args.quality,
        use_gpu=args.use_gpu,
        min_interval_ms=args.interval_ms,
        detect_scale=detect_scale,
        auto=bool(args.auto),
    )

    os.makedirs(cfg.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(cfg.camera_id)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open camera {cfg.camera_id}")

    print("Controls: [Space]=save   [A]=toggle auto   [Esc]=quit")

    captured = 0
    last_save_t = 0.0
    prev_corners_full: np.ndarray | None = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        gray_full = _to_gray(frame, use_gpu=cfg.use_gpu)
        gray_det = _resize_gray(gray_full, scale=cfg.detect_scale, use_gpu=cfg.use_gpu)

        found, corners = _find_corners(gray_det, cfg.pattern_size, cfg.quality)
        corners_full = None
        if found and corners is not None:
            corners_full = corners / float(cfg.detect_scale) if cfg.detect_scale != 1.0 else corners
            corners_full = corners_full.reshape(-1, 1, 2).astype(np.float32)

            # Draw for feedback
            cv2.drawChessboardCorners(frame, cfg.pattern_size, corners_full, True)

        # UI overlay
        label = f"{captured}/{cfg.count} | quality={cfg.quality} | auto={cfg.auto} | gpu_pre={bool(cfg.use_gpu and _cuda_available())}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Capture Calibration Images", frame)

        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        if key == 27:  # ESC
            break
        if key in (ord("a"), ord("A")):
            cfg.auto = not cfg.auto
        if key == 32:  # SPACE
            if corners_full is not None:
                accept = _should_accept(prev_corners_full, corners_full, min_mean_shift_px=12.0)
                if accept:
                    out_path = os.path.join(cfg.out_dir, f"calib_{captured:03d}.jpg")
                    cv2.imwrite(out_path, frame)
                    prev_corners_full = corners_full
                    captured += 1
                    last_save_t = now
                    print(f"Saved {out_path}")
            else:
                print("No board detected; not saved")

        if cfg.auto and corners_full is not None:
            if (now - last_save_t) * 1000.0 >= float(cfg.min_interval_ms):
                accept = _should_accept(prev_corners_full, corners_full, min_mean_shift_px=14.0)
                if accept:
                    out_path = os.path.join(cfg.out_dir, f"calib_{captured:03d}.jpg")
                    cv2.imwrite(out_path, frame)
                    prev_corners_full = corners_full
                    captured += 1
                    last_save_t = now
                    print(f"Auto-saved {out_path}")

        if captured >= cfg.count:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Captured {captured} images into {cfg.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
