import cv2
import numpy as np
import argparse
import os
import time
from typing import Optional, Tuple

def has_cuda():
    try:
        return hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except:
        return False

def to_gray(img, use_gpu):
    if not use_gpu or not has_cuda():
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
        return gpu_gray.download()
    except:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def find_chessboard(gray, pattern_size, quality):
    cols, rows = pattern_size
    if quality == "best" and hasattr(cv2, 'findChessboardCornersSB'):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        ret, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags)
    else:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    
    if ret:
        return True, corners.reshape(-1, 1, 2).astype(np.float32)
    return False, None

def corners_changed_enough(prev_corners, new_corners, min_shift_px=14.0):
    if prev_corners is None:
        return True
    diff = prev_corners.reshape(-1, 2) - new_corners.reshape(-1, 2)
    mean_shift = np.mean(np.linalg.norm(diff, axis=1))
    return mean_shift >= min_shift_px


def blur_score_variance_of_laplacian(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def try_set_max_resolution(cap: cv2.VideoCapture) -> Tuple[int, int]:
    # OpenCV can't reliably query "max"; we try common resolutions top-down.
    candidates = [
        (3840, 2160),
        (2560, 1440),
        (1920, 1080),
        (1600, 1200),
        (1280, 720),
        (1024, 768),
        (800, 600),
        (640, 480),
    ]
    best_w, best_h = 0, 0
    for w, h in candidates:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
        # Grab a frame to let the driver apply settings.
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            continue
        got_h, got_w = frame.shape[:2]
        if got_w >= best_w and got_h >= best_h:
            best_w, best_h = got_w, got_h
        # If we got exactly what we asked for, stop early.
        if got_w == w and got_h == h:
            return got_w, got_h
    if best_w > 0 and best_h > 0:
        return best_w, best_h
    # Fallback
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return w, h

def main(argv=None):
    parser = argparse.ArgumentParser(description="Capture calibration images from camera.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--out", default="calibration_images")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--pattern", default="8x5", help="e.g. 8x5")
    parser.add_argument("--quality", choices=["fast", "best"], default="best")
    parser.add_argument("--use-gpu", action="store_true", default=True)
    parser.add_argument("--auto", action="store_true", default=True)
    parser.add_argument("--interval-ms", type=int, default=800)
    parser.add_argument("--try-max-res", action="store_true", default=True, help="Try to capture at highest camera resolution")
    parser.add_argument("--min-blur", type=float, default=120.0, help="Variance of Laplacian threshold; higher = sharper")
    parser.add_argument("--ext", choices=["png", "jpg"], default="png")
    parser.add_argument("--png-compression", type=int, default=0, help="0-9 (0 is fastest, still lossless)")
    parser.add_argument("--min-shift-px", type=float, default=14.0, help="Minimum chessboard pose change between saves")
    parser.add_argument("--preview", action="store_true", default=True, help="Show preview window")
    args = parser.parse_args(argv)

    cols, rows = map(int, args.pattern.replace(" ", "").split("x"))
    pattern_size = (cols, rows)
    
    os.makedirs(args.out, exist_ok=True)
    
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    # Prefer MJPG for higher throughput at high resolutions (many webcams)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    if args.try_max_res:
        w, h = try_set_max_resolution(cap)
    else:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    print(f"Controls: [Space]=manual save  [A]=toggle auto  [Esc]=quit")
    print(f"Auto-capture: {args.auto} | GPU: {args.use_gpu and has_cuda()} | Quality: {args.quality}")
    print(f"Capture format: {args.ext.upper()} | Resolution: {w}x{h} | min_blur: {args.min_blur}")

    captured = 0
    last_save_time = 0.0
    prev_corners = None
    auto_enabled = args.auto

    while captured < args.count:
        ret, frame = cap.read()
        if not ret:
            continue

        # Keep a raw copy to save (do NOT save overlays)
        frame_raw = frame.copy()

        gray = to_gray(frame, args.use_gpu)
        blur = blur_score_variance_of_laplacian(gray)
        found, corners = find_chessboard(gray, pattern_size, args.quality)

        if found:
            cv2.drawChessboardCorners(frame, pattern_size, corners, True)

        # Status overlay
        status = f"{captured}/{args.count} | auto={auto_enabled} | blur={blur:.0f}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if args.preview:
            cv2.imshow("Capture Calibration", frame)

        key = cv2.waitKey(1) & 0xFF if args.preview else 255
        now = time.time()

        # Controls
        if key == 27:  # ESC
            break
        if key in (ord('a'), ord('A')):
            auto_enabled = not auto_enabled
            print(f"Auto-capture: {auto_enabled}")

        # Manual save (spacebar)
        if key == 32 and found:
            if blur < args.min_blur:
                print(f"Skip (blur {blur:.0f} < {args.min_blur})")
            elif corners_changed_enough(prev_corners, corners, min_shift_px=float(args.min_shift_px)):
                path = os.path.join(args.out, f"calib_{captured:04d}.{args.ext}")
                if args.ext == "png":
                    cv2.imwrite(path, frame_raw, [cv2.IMWRITE_PNG_COMPRESSION, int(args.png_compression)])
                else:
                    cv2.imwrite(path, frame_raw, [cv2.IMWRITE_JPEG_QUALITY, 100])
                prev_corners = corners
                captured += 1
                last_save_time = now
                print(f"Saved {path}")

        # Auto save
        if auto_enabled and found:
            elapsed_ms = (now - last_save_time) * 1000
            if elapsed_ms >= args.interval_ms:
                if blur < args.min_blur:
                    continue
                if corners_changed_enough(prev_corners, corners, min_shift_px=float(args.min_shift_px)):
                    path = os.path.join(args.out, f"calib_{captured:04d}.{args.ext}")
                    if args.ext == "png":
                        cv2.imwrite(path, frame_raw, [cv2.IMWRITE_PNG_COMPRESSION, int(args.png_compression)])
                    else:
                        cv2.imwrite(path, frame_raw, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    prev_corners = corners
                    captured += 1
                    last_save_time = now
                    print(f"Auto-saved {path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Captured {captured} images in {args.out}/")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
