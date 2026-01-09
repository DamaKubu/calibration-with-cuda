from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import yaml


Quality = Literal["fast", "best"]
Model = Literal["standard", "fisheye"]


@dataclass(frozen=True)
class PatternSpec:
    cols: int
    rows: int
    square_mm: float

    @property
    def size(self) -> tuple[int, int]:
        return (self.cols, self.rows)

    def object_points(self) -> np.ndarray:
        objp = np.zeros((self.cols * self.rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2)
        objp *= float(self.square_mm)
        return objp


def _parse_pattern(pattern: str) -> tuple[int, int]:
    s = pattern.lower().replace(" ", "")
    if "x" not in s:
        raise ValueError("--pattern must look like 8x5")
    a, b = s.split("x", 1)
    cols = int(a)
    rows = int(b)
    if cols <= 1 or rows <= 1:
        raise ValueError("--pattern must be at least 2x2")
    return cols, rows


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
    if scale <= 0 or scale > 1.0:
        raise ValueError("scale must be in (0, 1]")

    if not use_gpu or not _cuda_available():
        return cv2.resize(gray, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    try:
        gpu = cv2.cuda_GpuMat()
        gpu.upload(gray)
        gpu_resized = cv2.cuda.resize(gpu, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return gpu_resized.download()
    except Exception:
        return cv2.resize(gray, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _find_corners(gray: np.ndarray, pattern_size: tuple[int, int], quality: Quality) -> tuple[bool, np.ndarray | None]:
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


def _refine_corners(gray_full: np.ndarray, corners_full_res: np.ndarray, quality: Quality) -> np.ndarray:
    win = (11, 11) if quality == "fast" else (15, 15)
    iters = 30 if quality == "fast" else 60
    eps = 1e-6 if quality == "fast" else 1e-8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, eps)
    return cv2.cornerSubPix(gray_full, corners_full_res, winSize=win, zeroZone=(-1, -1), criteria=criteria)


def _detect_on_image(
    img_bgr: np.ndarray,
    pattern: PatternSpec,
    quality: Quality,
    use_gpu: bool,
    detect_scale: float,
) -> tuple[bool, np.ndarray | None, tuple[int, int]]:
    gray_full = _to_gray(img_bgr, use_gpu=use_gpu)
    h, w = gray_full.shape[:2]

    gray_det = _resize_gray(gray_full, scale=detect_scale, use_gpu=use_gpu)

    ok, corners = _find_corners(gray_det, pattern.size, quality)
    if not ok or corners is None:
        return False, None, (w, h)

    # Map corners back to full resolution if detection was scaled.
    if detect_scale != 1.0:
        corners = corners / float(detect_scale)

    corners = corners.reshape(-1, 1, 2).astype(np.float32)
    corners_refined = _refine_corners(gray_full, corners, quality)
    return True, corners_refined, (w, h)


def _standard_reprojection_errors(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
) -> list[float]:
    errors: list[float] = []
    for i, objp in enumerate(objpoints):
        imgp = imgpoints[i]
        rvec = rvecs[i]
        tvec = tvecs[i]
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        diff = (imgp.reshape(-1, 2) - proj.reshape(-1, 2))
        err = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
        errors.append(err)
    return errors


def _fisheye_reprojection_errors(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    K: np.ndarray,
    D: np.ndarray,
) -> list[float]:
    errors: list[float] = []
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs, strict=False):
        proj, _ = cv2.fisheye.projectPoints(objp, rvec, tvec, K, D)
        diff = (imgp.reshape(-1, 2) - proj.reshape(-1, 2))
        err = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
        errors.append(err)
    return errors


def _save_yaml(out_path: str, *, model: Model, image_size: tuple[int, int], K: np.ndarray, D: np.ndarray) -> None:
    data = {
        "model": str(model),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "K": K.tolist(),
        "D": D.ravel().tolist(),
    }
    with open(out_path, "w") as f:
        yaml.dump(data, f)


def _calibrate_standard(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    image_size: tuple[int, int],
    quality: Quality,
) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    flags = 0
    if quality == "best":
        flags |= cv2.CALIB_RATIONAL_MODEL

    K = np.eye(3, dtype=np.float64)
    dist = np.zeros((8, 1), dtype=np.float64) if quality == "best" else np.zeros((5, 1), dtype=np.float64)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200 if quality == "best" else 100, 1e-10)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        K,
        dist,
        flags=flags,
        criteria=criteria,
    )
    return float(ret), K, dist, rvecs, tvecs


def _calibrate_fisheye(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    image_size: tuple[int, int],
    quality: Quality,
) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    K = np.zeros((3, 3), dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    rvecs: list[np.ndarray] = []
    tvecs: list[np.ndarray] = []

    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
    if quality == "best" and hasattr(cv2.fisheye, "CALIB_FIX_PRINCIPAL_POINT"):
        # Typically you *don't* want to fix principal point; leave it free by default.
        pass

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200 if quality == "best" else 100, 1e-10)

    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        image_size,
        K,
        D,
        rvecs,
        tvecs,
        flags,
        criteria,
    )
    return float(ret), K, D, rvecs, tvecs


def _iteratively_reject_outliers(
    *,
    model: Model,
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    image_size: tuple[int, int],
    quality: Quality,
    max_remove: int,
    target_rmse_px: float,
) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[float], list[int]]:
    keep_idx = list(range(len(objpoints)))

    def _calibrate_current() -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[float]]:
        cur_obj = [objpoints[i] for i in keep_idx]
        cur_img = [imgpoints[i] for i in keep_idx]
        if model == "standard":
            rms, K, D, rvecs, tvecs = _calibrate_standard(cur_obj, cur_img, image_size, quality)
            per_view = _standard_reprojection_errors(cur_obj, cur_img, rvecs, tvecs, K, D)
        else:
            rms, K, D, rvecs, tvecs = _calibrate_fisheye(cur_obj, cur_img, image_size, quality)
            per_view = _fisheye_reprojection_errors(cur_obj, cur_img, rvecs, tvecs, K, D)
        return rms, K, D, rvecs, tvecs, per_view

    removed: list[int] = []
    rms, K, D, rvecs, tvecs, per_view = _calibrate_current()

    # Remove worst views until RMSE is acceptable or we hit max_remove.
    for _ in range(max_remove):
        if len(keep_idx) <= 8:
            break
        if rms <= target_rmse_px:
            break

        worst_local = int(np.argmax(np.array(per_view)))
        worst_global = keep_idx[worst_local]
        removed.append(worst_global)
        keep_idx.pop(worst_local)

        rms, K, D, rvecs, tvecs, per_view = _calibrate_current()

    return rms, K, D, rvecs, tvecs, per_view, removed


def run_calibration_cli(
    *,
    images_glob: str,
    pattern: str,
    square_mm: float,
    quality: Quality,
    model: Model,
    use_gpu: bool,
    out_path: str,
) -> int:
    cols, rows = _parse_pattern(pattern)
    pat = PatternSpec(cols=cols, rows=rows, square_mm=square_mm)

    paths = sorted(glob.glob(images_glob))
    if not paths:
        raise SystemExit(f"No images found for glob: {images_glob}")

    objp = pat.object_points()
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []

    image_size: tuple[int, int] | None = None

    detect_scale = 0.5 if quality == "fast" else 1.0

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue

        ok, corners, size_wh = _detect_on_image(
            img,
            pat,
            quality=quality,
            use_gpu=use_gpu,
            detect_scale=detect_scale,
        )
        if image_size is None:
            image_size = size_wh

        if ok and corners is not None:
            objpoints.append(objp)
            imgpoints.append(corners)

    if image_size is None:
        raise SystemExit("Failed to read any images.")

    if len(objpoints) < 8:
        raise SystemExit(f"Not enough usable images with detected corners: {len(objpoints)} (need at least 8)")

    # Outlier rejection helps a lot for best-possible calibration.
    max_remove = 0 if quality == "fast" else min(6, max(0, len(objpoints) - 10))
    target_rmse_px = 1.2 if quality == "fast" else 0.6

    rms, K, D, _rvecs, _tvecs, per_view, removed = _iteratively_reject_outliers(
        model=model,
        objpoints=objpoints,
        imgpoints=imgpoints,
        image_size=image_size,
        quality=quality,
        max_remove=max_remove,
        target_rmse_px=target_rmse_px,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    _save_yaml(out_path, model=model, image_size=image_size, K=K, D=D)

    kept = len(objpoints) - len(removed)
    print(f"Images: {len(paths)} total, {len(objpoints)} detected, {kept} kept, {len(removed)} removed")
    print(f"Model: {model} | Quality: {quality} | GPU preprocess: {bool(use_gpu and _cuda_available())}")
    print(f"RMS reprojection error: {rms:.4f} px")
    if per_view:
        print(f"Per-view RMSE: min={min(per_view):.4f}  median={float(np.median(per_view)):.4f}  max={max(per_view):.4f}")
    print(f"Saved: {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics from chessboard images.")
    parser.add_argument("--images", default="calibration_images/*.jpg", help="Input glob (default: calibration_images/*.jpg)")
    parser.add_argument("--pattern", default="8x5", help="Chessboard inner corners COLSxROWS (default: 8x5)")
    parser.add_argument("--square-mm", type=float, default=65.0, help="Square size in millimeters")
    parser.add_argument("--quality", choices=["fast", "best"], default="best")
    parser.add_argument("--model", choices=["standard", "fisheye"], default="standard")
    parser.add_argument("--use-gpu", action="store_true", help="Use CUDA preprocessing if available")
    parser.add_argument("--out", default="cam_intrinsics.yaml", help="Output YAML path")
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
    raise SystemExit(main())
