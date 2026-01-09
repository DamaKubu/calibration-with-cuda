import cv2
import numpy as np
import argparse
import os
import time

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
    args = parser.parse_args(argv)

    cols, rows = map(int, args.pattern.replace(" ", "").split("x"))
    pattern_size = (cols, rows)
    
    os.makedirs(args.out, exist_ok=True)
    
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    print(f"Controls: [Space]=manual save  [A]=toggle auto  [Esc]=quit")
    print(f"Auto-capture: {args.auto} | GPU: {args.use_gpu and has_cuda()} | Quality: {args.quality}")

    captured = 0
    last_save_time = 0.0
    prev_corners = None
    auto_enabled = args.auto

    while captured < args.count:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = to_gray(frame, args.use_gpu)
        found, corners = find_chessboard(gray, pattern_size, args.quality)

        if found:
            cv2.drawChessboardCorners(frame, pattern_size, corners, True)

        # Status overlay
        status = f"{captured}/{args.count} | auto={auto_enabled}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Capture Calibration", frame)

        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        # Controls
        if key == 27:  # ESC
            break
        if key in (ord('a'), ord('A')):
            auto_enabled = not auto_enabled
            print(f"Auto-capture: {auto_enabled}")

        # Manual save (spacebar)
        if key == 32 and found:
            if corners_changed_enough(prev_corners, corners, min_shift_px=12.0):
                path = os.path.join(args.out, f"calib_{captured:03d}.jpg")
                cv2.imwrite(path, frame)
                prev_corners = corners
                captured += 1
                last_save_time = now
                print(f"Saved {path}")

        # Auto save
        if auto_enabled and found:
            elapsed_ms = (now - last_save_time) * 1000
            if elapsed_ms >= args.interval_ms:
                if corners_changed_enough(prev_corners, corners):
                    path = os.path.join(args.out, f"calib_{captured:03d}.jpg")
                    cv2.imwrite(path, frame)
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
