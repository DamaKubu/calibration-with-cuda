import cv2
import numpy as np
import glob
import argparse
import os
import yaml
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class Detection:
    path: str
    corners: np.ndarray  # (N,1,2) float32
    gray_shape: Tuple[int, int]  # (h,w)


def _reprojection_errors_fisheye(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    K: np.ndarray,
    D: np.ndarray,
) -> List[float]:
    errors: List[float] = []
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.fisheye.projectPoints(objp, rvec, tvec, K, D)
        diff = projected.reshape(-1, 2) - imgp.reshape(-1, 2)
        err = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
        errors.append(err)
    return errors


def _reprojection_errors_standard(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
) -> List[float]:
    errors: List[float] = []
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        diff = projected.reshape(-1, 2) - imgp.reshape(-1, 2)
        err = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
        errors.append(err)
    return errors

def calibrate_camera(
    images_glob,
    pattern_size,
    square_mm,
    quality,
    use_gpu,
    model,
    output,
    max_remove,
    max_per_image_error,
):
    # Prepare object points
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_mm)

    detections: List[Detection] = []
    images = sorted(glob.glob(images_glob))
    
    if not images:
        raise SystemExit(f"No images found: {images_glob}")

    has_cuda = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
    gpu_ok = use_gpu and has_cuda

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue

        # GPU preprocessing if requested
        if gpu_ok:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(img)
                gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
                gray = gpu_gray.download()
            except:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find corners (use better detector for 'best' quality)
        if quality == "best" and hasattr(cv2, 'findChessboardCornersSB'):
            ret, corners = cv2.findChessboardCornersSB(
                gray, pattern_size, 
                cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            )
        else:
            ret, corners = cv2.findChessboardCorners(
                gray, pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            )

        if ret:
            # Refine corners
            win = (15, 15) if quality == "best" else (11, 11)
            iters = 80 if quality == "best" else 40
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, 1e-12)
            corners2 = cv2.cornerSubPix(gray, corners, winSize=win, zeroZone=(-1, -1), criteria=criteria)
            detections.append(Detection(path=fname, corners=corners2, gray_shape=gray.shape))

    if len(detections) < 8:
        raise SystemExit(f"Not enough images with corners: {len(detections)} (need >= 8 for stable best-quality)")

    # Ensure all images are the same resolution
    shapes = {d.gray_shape for d in detections}
    if len(shapes) != 1:
        raise SystemExit(f"Images have different sizes: {sorted(list(shapes))}. Capture all at one fixed resolution.")
    h, w = detections[0].gray_shape
    img_size = (w, h)

    def build_points(idxs: List[int]):
        objpoints = [objp.reshape(-1, 1, 3).astype(np.float32) for _ in idxs]
        imgpoints = [detections[i].corners.astype(np.float32) for i in idxs]
        paths = [detections[i].path for i in idxs]
        return objpoints, imgpoints, paths

    kept = list(range(len(detections)))

    def run_once(idxs: List[int]):
        objpoints, imgpoints, paths = build_points(idxs)

        if model == "fisheye":
            K = np.zeros((3, 3))
            D = np.zeros((4, 1))
            rvecs, tvecs = [], []
            flags = (
                cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                | cv2.fisheye.CALIB_CHECK_COND
                | cv2.fisheye.CALIB_FIX_SKEW
            )
            iters = 500 if quality == "best" else 150
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, 1e-12)
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints, imgpoints, img_size, K, D, rvecs, tvecs, flags, criteria
            )
            per = _reprojection_errors_fisheye(objpoints, imgpoints, rvecs, tvecs, K, D)
            return rms, K, D, rvecs, tvecs, per, paths

        # Standard model
        K = np.eye(3)
        dist = np.zeros((8, 1)) if quality == "best" else np.zeros((5, 1))
        flags = cv2.CALIB_RATIONAL_MODEL if quality == "best" else 0
        iters = 500 if quality == "best" else 150
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, 1e-12)
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_size, K, dist, flags=flags, criteria=criteria
        )
        per = _reprojection_errors_standard(objpoints, imgpoints, rvecs, tvecs, K, dist)
        return rms, K, dist, rvecs, tvecs, per, paths

    rejected: List[Dict[str, Any]] = []

    # Iterative outlier rejection by per-image reprojection error
    for _ in range(int(max_remove) + 1):
        rms, K, D, rvecs, tvecs, per_errors, paths = run_once(kept)
        worst_i = int(np.argmax(per_errors))
        worst_err = float(per_errors[worst_i])
        worst_path = paths[worst_i]

        if max_per_image_error is not None and worst_err > float(max_per_image_error) and len(kept) > 8:
            rejected.append({"path": worst_path, "reproj_error_px": worst_err})
            del kept[worst_i]
            continue
        break

    # Recompute final per-image errors for reporting
    objpoints, imgpoints, paths = build_points(kept)
    if model == "fisheye":
        per_errors = _reprojection_errors_fisheye(objpoints, imgpoints, rvecs, tvecs, K, D)
    else:
        per_errors = _reprojection_errors_standard(objpoints, imgpoints, rvecs, tvecs, K, D)

    print(f"\nCalibrated with {len(kept)} images (rejected {len(rejected)})")
    print(f"RMS error (solver): {rms:.4f} px")
    print(f"Mean per-image reproj: {float(np.mean(per_errors)):.4f} px")
    print(f"Max  per-image reproj: {float(np.max(per_errors)):.4f} px")
    print(f"GPU: {gpu_ok} | Quality: {quality} | Model: {model}")
    print(f"Image size: {img_size[0]}x{img_size[1]}")

    # Save results
    if output is None:
        output = f"intrinsics_{model}.yml"

    payload: Dict[str, Any] = {
        "model": model,
        "quality": quality,
        "use_gpu": bool(gpu_ok),
        "pattern": {"cols": int(cols), "rows": int(rows)},
        "square_mm": float(square_mm),
        "image_size": {"width": int(img_size[0]), "height": int(img_size[1])},
        "rms_error_px": float(rms),
        "camera_matrix": K.tolist(),
        "distortion_coefficients": D.reshape(-1).tolist(),
        "kept_images": paths,
        "per_image_reproj_error_px": {p: float(e) for p, e in zip(paths, per_errors)},
        "rejected": rejected,
    }

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    print(f"\nSaved calibration: {output}")
    return 0


def run_calibration_cli(*, images_glob, pattern, square_mm, quality, model, use_gpu, output, max_remove, max_per_image_error):
    cols, rows = map(int, pattern.lower().replace(" ", "").split("x"))
    return calibrate_camera(
        images_glob,
        (cols, rows),
        square_mm,
        quality,
        use_gpu,
        model,
        output,
        max_remove,
        max_per_image_error,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics.")
    parser.add_argument("--images", default="calibration_images/*.png")
    parser.add_argument("--pattern", default="8x5", help="e.g. 8x5")
    parser.add_argument("--square-mm", type=float, default=65.0)
    parser.add_argument("--quality", choices=["fast", "best"], default="best")
    parser.add_argument("--model", choices=["standard", "fisheye"], default="fisheye")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--output", default=None, help="YAML output path")
    parser.add_argument("--max-remove", type=int, default=10, help="Max outlier images to remove")
    parser.add_argument("--max-per-image-error", type=float, default=1.5, help="Reject images above this reprojection error (px)")
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
