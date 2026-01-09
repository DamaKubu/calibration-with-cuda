import cv2
import numpy as np
import glob
import argparse

def calibrate_camera(images_glob, pattern_size, square_mm, quality, use_gpu, model):
    # Prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_mm

    objpoints = []
    imgpoints = []
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
            objpoints.append(objp)
            # Refine corners
            win = (15, 15) if quality == "best" else (11, 11)
            iters = 60 if quality == "best" else 30
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, 1e-8)
            corners2 = cv2.cornerSubPix(gray, corners, winSize=win, zeroZone=(-1, -1), criteria=criteria)
            imgpoints.append(corners2)

    if len(objpoints) < 5:
        raise SystemExit(f"Not enough images with corners: {len(objpoints)}")

    img_size = gray.shape[::-1]

    # Calibrate
    if model == "fisheye":
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs, tvecs = [], []
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
        iters = 200 if quality == "best" else 100
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, 1e-10)
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints, imgpoints, img_size, K, D, rvecs, tvecs, flags, criteria
        )
    else:
        # Standard (barrel distortion) model
        K = np.eye(3)
        dist = np.zeros((8, 1)) if quality == "best" else np.zeros((5, 1))
        flags = cv2.CALIB_RATIONAL_MODEL if quality == "best" else 0
        iters = 200 if quality == "best" else 100
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, 1e-10)
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_size, K, dist, flags=flags, criteria=criteria
        )
        D = dist

    # Print results
    print(f"\nCalibrated with {len(objpoints)} images")
    print(f"RMS error: {rms:.4f} px")
    print(f"GPU: {gpu_ok} | Quality: {quality} | Model: {model}")
    print(f"\nCamera Matrix (K):")
    print(K)
    print(f"\nDistortion Coefficients (D):")
    print(D.ravel())
    print(f"\nImage size: {img_size[0]}x{img_size[1]}")
    return 0


def run_calibration_cli(*, images_glob, pattern, square_mm, quality, model, use_gpu):
    cols, rows = map(int, pattern.lower().replace(" ", "").split("x"))
    return calibrate_camera(images_glob, (cols, rows), square_mm, quality, use_gpu, model)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics.")
    parser.add_argument("--images", default="calibration_images/*.jpg")
    parser.add_argument("--pattern", default="8x5", help="e.g. 8x5")
    parser.add_argument("--square-mm", type=float, default=65.0)
    parser.add_argument("--quality", choices=["fast", "best"], default="best")
    parser.add_argument("--model", choices=["standard", "fisheye"], default="standard")
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args(argv)
    return run_calibration_cli(
        images_glob=args.images,
        pattern=args.pattern,
        square_mm=args.square_mm,
        quality=args.quality,
        model=args.model,
        use_gpu=args.use_gpu,
    )


if __name__ == "__main__":
    raise SystemExit(main())
