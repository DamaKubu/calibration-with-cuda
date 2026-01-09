from __future__ import annotations

import argparse

import cv2
import numpy as np


def calibrate_camera(
    square_mm: float = 65.0,
    camera_id: int = 1,
    samples: int = 15,
    pattern_cols: int = 8,
    pattern_rows: int = 5,
    roi_frac: float = 1.0,
) -> int:
    # Prepare object points scaled by square size in millimeters.
    objp = np.zeros((pattern_cols * pattern_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
    objp *= float(square_mm)

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []

    cap = cv2.VideoCapture(int(camera_id))
    if not cap.isOpened():
        print(f"Failed to open camera {camera_id}")
        return 2

    print("Collecting chessboard views... Press ESC to quit early.")
    last_corners: np.ndarray | None = None
    last_size_wh: tuple[int, int] | None = None

    while len(objpoints) < int(samples):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        last_size_wh = (w, h)

        # Optionally crop to the central region when you want to lock onto a smaller sub-grid
        # inside a larger printed board. The crop keeps the aspect ratio of the frame.
        if roi_frac <= 0 or roi_frac > 1:
            print("--roi-frac must be in (0, 1].")
            break
        if roi_frac < 1.0:
            side = int(min(w, h) * roi_frac)
            x0 = (w - side) // 2
            y0 = (h - side) // 2
            crop = gray[y0 : y0 + side, x0 : x0 + side]
            gray_det = crop
            offset = (x0, y0)
        else:
            gray_det = gray
            offset = (0, 0)

        ret, corners = cv2.findChessboardCorners(
            gray_det,
            (pattern_cols, pattern_rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if ret and corners is not None:
            # Refine to subpixel for better accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            corners = cv2.cornerSubPix(gray_det, corners, (11, 11), (-1, -1), criteria)

            # Shift corners back to full-frame coordinates if we cropped.
            if offset != (0, 0):
                corners = corners + np.array([[offset]], dtype=np.float32)

            # Simple diversity filter: avoid accepting nearly identical views
            accept = True
            if last_corners is not None:
                a = last_corners.reshape(-1, 2)
                b = corners.reshape(-1, 2)
                mean_shift = float(np.mean(np.linalg.norm(a - b, axis=1)))
                accept = mean_shift >= 8.0

            if accept:
                objpoints.append(objp.copy())
                imgpoints.append(corners)
                last_corners = corners.copy()

            # Draw for feedback
            vis = frame.copy()
            cv2.drawChessboardCorners(vis, (pattern_cols, pattern_rows), corners, True)
        else:
            vis = frame

        label = f"views={len(objpoints)}/{samples}  pattern={pattern_cols}x{pattern_rows}  square={square_mm:.1f} mm  roi={roi_frac:.2f}"
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Minimalistic Calibration", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 8:
        print(f"Not enough views: {len(objpoints)} (need at least 8)")
        return 3

    # Calibrate intrinsics
    if last_size_wh is None:
        print("No frames read from camera.")
        return 4
    image_size = last_size_wh
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    print("Calibration complete.")
    print(f"RMS reprojection error: {rms:.4f} px")
    print(f"Camera matrix (K):\n{K}")
    print(f"Distortion coefficients (k1,k2,p1,p2,k3,...):\n{dist.ravel()}")
    print(f"fx={fx:.3f}  fy={fy:.3f}  cx={cx:.3f}  cy={cy:.3f}")
    print(f"Chessboard square size: {square_mm:.1f} mm")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Minimal camera calibration with configurable chessboard size.")
    parser.add_argument("--square-mm", type=float, default=65.0, help="Chessboard square size in millimeters (default: 65)")
    parser.add_argument("--camera", type=int, default=1, help="OpenCV camera index (default: 1)")
    parser.add_argument("--samples", type=int, default=15, help="Number of views to collect (default: 15)")
    parser.add_argument("--pattern-cols", type=int, default=8, help="Chessboard inner corners (columns), default 8")
    parser.add_argument("--pattern-rows", type=int, default=5, help="Chessboard inner corners (rows), default 5")
    parser.add_argument(
        "--roi-frac",
        type=float,
        default=1.0,
        help="Fraction of the shorter image side to center-crop before detection; use <1.0 to target a smaller sub-grid inside a larger board",
    )
    args = parser.parse_args(argv)

    return calibrate_camera(
        square_mm=args.square_mm,
        camera_id=args.camera,
        samples=args.samples,
        pattern_cols=args.pattern_cols,
        pattern_rows=args.pattern_rows,
        roi_frac=args.roi_frac,
    )


if __name__ == "__main__":
    raise SystemExit(main())
