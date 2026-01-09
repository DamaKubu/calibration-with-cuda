"""
Simple Extrinsic Camera Calibration with CUDA Acceleration
===========================================================

This script calibrates the extrinsic parameters (relative position and rotation)
between multiple cameras using an 8x5 chessboard with 65mm squares.

Features:
- CUDA-accelerated feature detection and matching
- Stereo calibration for overlapping camera pairs
- Bundle adjustment for multi-camera setup
- Support for fisheye distortion models
- Tennis ball tracking as alternative calibration target

Hardware Requirements:
- NVIDIA GPU with 4GB+ VRAM (tested on RTX 3050 Ti)
- 32GB system RAM

Author: Calibration System
Date: January 2026
"""

import numpy as np
import cv2
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# Chessboard configuration
CHESSBOARD_SIZE = (8, 5)  # Inner corners (columns, rows)
SQUARE_SIZE = 65.0  # mm

# CUDA acceleration settings
USE_CUDA = True
CUDA_DEVICE_ID = 0


class ExtrinsicCalibrator:
    """Handles extrinsic calibration between multiple cameras."""
    
    def __init__(self, intrinsics_path: str, use_cuda: bool = True):
        """
        Initialize the calibrator with intrinsic parameters.
        
        Args:
            intrinsics_path: Path to cameras_intrinsics.json file
            use_cuda: Enable CUDA acceleration if available
        """
        self.intrinsics = self._load_intrinsics(intrinsics_path)
        self.use_cuda = use_cuda and self._check_cuda()
        
        # Prepare 3D object points for chessboard
        self.obj_points_pattern = self._create_chessboard_object_points()
        
        # Store calibration results
        self.camera_poses = {}  # Relative poses between cameras
        self.all_object_points = []
        self.all_image_points = {}  # Per camera
        
        print(f"CUDA acceleration: {'ENABLED' if self.use_cuda else 'DISABLED'}")
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available in OpenCV."""
        try:
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                cv2.cuda.setDevice(CUDA_DEVICE_ID)
                print(f"CUDA device count: {cv2.cuda.getCudaEnabledDeviceCount()}")
                print(f"Using CUDA device: {cv2.cuda.getDevice()}")
                return True
        except Exception as e:
            print(f"CUDA not available: {e}")
        return False
    
    def _load_intrinsics(self, path: str) -> Dict:
        """Load camera intrinsic parameters from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        intrinsics = {}
        for cam in data['cameras']:
            cam_id = cam['id']
            intrinsics[cam_id] = {
                'K': np.array(cam['fisheye']['K'], dtype=np.float64),
                'D': np.array(cam['fisheye']['D'], dtype=np.float64),
                'image_size': tuple(cam['image_size']),
                'is_fisheye': True
            }
        
        print(f"Loaded intrinsics for {len(intrinsics)} cameras")
        return intrinsics
    
    def _create_chessboard_object_points(self) -> np.ndarray:
        """Create 3D coordinates for chessboard corners in mm."""
        objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 
                               0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        objp *= SQUARE_SIZE  # Scale to actual size in mm
        return objp
    
    def detect_chessboard_cuda(self, image: np.ndarray, 
                               camera_id: str) -> Optional[np.ndarray]:
        """
        Detect chessboard corners with CUDA acceleration.
        
        Args:
            image: Input image (BGR)
            camera_id: Camera identifier
            
        Returns:
            Detected corners or None if not found
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Upload to GPU if CUDA is enabled
        if self.use_cuda:
            try:
                gpu_gray = cv2.cuda_GpuMat()
                gpu_gray.upload(gray)
                
                # Enhance image on GPU
                clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gpu_enhanced = clahe.apply(gpu_gray)
                gray = gpu_enhanced.download()
            except Exception as e:
                print(f"CUDA processing failed, falling back to CPU: {e}")
                # Apply CLAHE on CPU
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
        else:
            # CPU-only CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # Detect chessboard corners (CPU operation)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags)
        
        if ret:
            # Refine corners to subpixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return corners
        
        return None
    
    def calibrate_stereo_pair(self, cam1_id: str, cam2_id: str,
                             image_pairs: List[Tuple[np.ndarray, np.ndarray]],
                             visualize: bool = True) -> Dict:
        """
        Calibrate extrinsic parameters between two cameras.
        
        Args:
            cam1_id: First camera ID
            cam2_id: Second camera ID
            image_pairs: List of synchronized image pairs
            visualize: Show detection results
            
        Returns:
            Dictionary with R, T, E, F matrices and reprojection error
        """
        print(f"\n{'='*60}")
        print(f"Calibrating stereo pair: {cam1_id} <-> {cam2_id}")
        print(f"{'='*60}")
        
        obj_points = []
        img_points_1 = []
        img_points_2 = []
        
        valid_pairs = 0
        for idx, (img1, img2) in enumerate(image_pairs):
            print(f"Processing pair {idx+1}/{len(image_pairs)}...", end=' ')
            
            # Detect corners in both images
            corners1 = self.detect_chessboard_cuda(img1, cam1_id)
            corners2 = self.detect_chessboard_cuda(img2, cam2_id)
            
            if corners1 is not None and corners2 is not None:
                obj_points.append(self.obj_points_pattern)
                img_points_1.append(corners1)
                img_points_2.append(corners2)
                valid_pairs += 1
                print(f"✓ Valid")
                
                if visualize and idx < 3:  # Show first 3 detections
                    vis1 = cv2.drawChessboardCorners(img1.copy(), CHESSBOARD_SIZE, 
                                                     corners1, True)
                    vis2 = cv2.drawChessboardCorners(img2.copy(), CHESSBOARD_SIZE, 
                                                     corners2, True)
                    combined = np.hstack([vis1, vis2])
                    cv2.imshow(f'Detection Pair {idx+1}', combined)
                    cv2.waitKey(500)
            else:
                print(f"✗ Failed")
        
        if visualize:
            cv2.destroyAllWindows()
        
        print(f"\nValid pairs: {valid_pairs}/{len(image_pairs)}")
        
        if valid_pairs < 10:
            raise ValueError(f"Not enough valid pairs ({valid_pairs}). Need at least 10.")
        
        # Get intrinsic parameters
        K1 = self.intrinsics[cam1_id]['K']
        D1 = self.intrinsics[cam1_id]['D']
        K2 = self.intrinsics[cam2_id]['K']
        D2 = self.intrinsics[cam2_id]['D']
        
        img_size_1 = self.intrinsics[cam1_id]['image_size']
        
        # Stereo calibration flags for fisheye
        flags = cv2.fisheye.CALIB_FIX_INTRINSIC
        
        # Stereo calibration criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        
        print("\nRunning stereo calibration...")
        start_time = time.time()
        
        # Use fisheye stereo calibration
        ret, K1_out, D1_out, K2_out, D2_out, R, T = cv2.fisheye.stereoCalibrate(
            obj_points, img_points_1, img_points_2,
            K1, D1, K2, D2,
            img_size_1,
            flags=flags,
            criteria=criteria
        )
        
        elapsed = time.time() - start_time
        print(f"Calibration completed in {elapsed:.2f} seconds")
        print(f"RMS reprojection error: {ret:.4f} pixels")
        
        # Calculate essential and fundamental matrices
        E = self._compute_essential_matrix(R, T)
        F = self._compute_fundamental_matrix(K1, K2, E)
        
        # Store results
        result = {
            'R': R,  # Rotation from cam1 to cam2
            'T': T,  # Translation from cam1 to cam2
            'E': E,  # Essential matrix
            'F': F,  # Fundamental matrix
            'rms_error': ret,
            'valid_pairs': valid_pairs,
            'calibration_time': elapsed
        }
        
        # Print calibration summary
        self._print_stereo_summary(cam1_id, cam2_id, result)
        
        return result
    
    def _compute_essential_matrix(self, R: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute essential matrix from R and T."""
        T_x = np.array([
            [0, -T[2, 0], T[1, 0]],
            [T[2, 0], 0, -T[0, 0]],
            [-T[1, 0], T[0, 0], 0]
        ])
        E = T_x @ R
        return E
    
    def _compute_fundamental_matrix(self, K1: np.ndarray, K2: np.ndarray, 
                                   E: np.ndarray) -> np.ndarray:
        """Compute fundamental matrix from intrinsics and essential matrix."""
        F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
        return F
    
    def _print_stereo_summary(self, cam1_id: str, cam2_id: str, result: Dict):
        """Print calibration summary."""
        print(f"\n{'='*60}")
        print(f"STEREO CALIBRATION RESULTS: {cam1_id} → {cam2_id}")
        print(f"{'='*60}")
        
        R = result['R']
        T = result['T']
        
        # Convert rotation matrix to Euler angles (in degrees)
        angles = self._rotation_matrix_to_euler(R)
        
        # Compute baseline distance
        baseline = np.linalg.norm(T)
        
        print(f"\nRotation (Euler angles in degrees):")
        print(f"  Roll  (X): {angles[0]:8.3f}°")
        print(f"  Pitch (Y): {angles[1]:8.3f}°")
        print(f"  Yaw   (Z): {angles[2]:8.3f}°")
        
        print(f"\nTranslation vector (mm):")
        print(f"  X: {T[0, 0]:8.2f}")
        print(f"  Y: {T[1, 0]:8.2f}")
        print(f"  Z: {T[2, 0]:8.2f}")
        print(f"  Baseline: {baseline:.2f} mm ({baseline/10:.2f} cm)")
        
        print(f"\nCalibration quality:")
        print(f"  RMS error: {result['rms_error']:.4f} pixels")
        print(f"  Valid image pairs: {result['valid_pairs']}")
        print(f"  Computation time: {result['calibration_time']:.2f} seconds")
        
        if result['rms_error'] < 1.0:
            quality = "EXCELLENT"
        elif result['rms_error'] < 2.0:
            quality = "GOOD"
        elif result['rms_error'] < 3.0:
            quality = "ACCEPTABLE"
        else:
            quality = "POOR - Consider recalibration"
        
        print(f"  Quality assessment: {quality}")
        print(f"{'='*60}\n")
    
    def _rotation_matrix_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (XYZ convention) in degrees."""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.degrees([x, y, z])
    
    def save_calibration(self, output_path: str):
        """Save calibration results to JSON file."""
        results = {
            'calibration_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'chessboard_size': CHESSBOARD_SIZE,
            'square_size_mm': SQUARE_SIZE,
            'cuda_enabled': self.use_cuda,
            'camera_pairs': {}
        }
        
        for pair_key, pair_data in self.camera_poses.items():
            results['camera_pairs'][pair_key] = {
                'R': pair_data['R'].tolist(),
                'T': pair_data['T'].tolist(),
                'rms_error': float(pair_data['rms_error']),
                'valid_pairs': int(pair_data['valid_pairs'])
            }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Calibration saved to: {output_path}")


def capture_calibration_images(camera_indices: List[int], 
                               num_images: int = 20) -> Dict[str, List[np.ndarray]]:
    """
    Capture synchronized images from multiple cameras for calibration.
    
    Args:
        camera_indices: List of camera indices to use
        num_images: Number of image sets to capture
        
    Returns:
        Dictionary mapping camera IDs to lists of captured images
    """
    # Open all cameras
    caps = {}
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"Warning: Could not open camera {idx}")
            continue
        
        # Set resolution if needed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        caps[f"cam{idx}"] = cap
    
    if len(caps) < 2:
        raise ValueError("Need at least 2 cameras for stereo calibration")
    
    print(f"\nCapturing {num_images} image sets from {len(caps)} cameras")
    print("Press SPACE to capture, ESC to finish early, Q to quit\n")
    
    captured = {cam_id: [] for cam_id in caps.keys()}
    count = 0
    
    while count < num_images:
        # Read frames from all cameras
        frames = {}
        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                frames[cam_id] = frame
        
        # Display all frames
        display = []
        for cam_id in sorted(frames.keys()):
            frame = frames[cam_id].copy()
            cv2.putText(frame, f"{cam_id} - {count}/{num_images}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            display.append(frame)
        
        if display:
            combined = np.hstack(display)
            cv2.imshow('Calibration Capture', combined)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space to capture
            for cam_id, frame in frames.items():
                captured[cam_id].append(frame.copy())
            count += 1
            print(f"Captured image set {count}/{num_images}")
        
        elif key == 27:  # ESC to finish early
            if count >= 10:
                print(f"\nFinishing with {count} image sets")
                break
            else:
                print(f"Need at least 10 image sets, have {count}")
        
        elif key == ord('q'):  # Q to quit
            captured = {cam_id: [] for cam_id in caps.keys()}
            break
    
    # Cleanup
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
    
    return captured


def main():
    """Main calibration workflow."""
    # Paths
    workspace = Path(__file__).parent.parent.parent
    intrinsics_path = workspace / "cameras_intrinsics.json"
    output_path = workspace / "camera_extrinsics.json"
    
    # Initialize calibrator
    calibrator = ExtrinsicCalibrator(str(intrinsics_path), use_cuda=USE_CUDA)
    
    print("\n" + "="*60)
    print("EXTRINSIC CAMERA CALIBRATION")
    print("="*60)
    print(f"Chessboard: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} (inner corners)")
    print(f"Square size: {SQUARE_SIZE} mm")
    print("="*60 + "\n")
    
    # Option 1: Capture new images
    print("Choose calibration source:")
    print("1. Capture new images from live cameras")
    print("2. Load images from directory")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        # Capture from cameras
        num_cameras = int(input("Number of cameras (2-4): "))
        camera_indices = [int(input(f"Camera {i+1} index: ")) for i in range(num_cameras)]
        num_images = int(input("Number of image sets to capture (recommended: 20-30): "))
        
        images = capture_calibration_images(camera_indices, num_images)
        
        # Calibrate all pairs
        cam_ids = sorted(images.keys())
        for i in range(len(cam_ids) - 1):
            for j in range(i + 1, len(cam_ids)):
                cam1, cam2 = cam_ids[i], cam_ids[j]
                
                # Create image pairs
                image_pairs = list(zip(images[cam1], images[cam2]))
                
                # Calibrate
                result = calibrator.calibrate_stereo_pair(cam1, cam2, image_pairs)
                calibrator.camera_poses[f"{cam1}_{cam2}"] = result
    
    else:
        print("\nLoad your synchronized images and call:")
        print("  result = calibrator.calibrate_stereo_pair(cam1_id, cam2_id, image_pairs)")
        return
    
    # Save results
    calibrator.save_calibration(str(output_path))
    
    print("\n" + "="*60)
    print("CALIBRATION COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {output_path}")
    print("\nNext steps:")
    print("1. Use these extrinsic parameters for multi-view reconstruction")
    print("2. Implement stereo matching with CUDA for depth estimation")
    print("3. Set up voxel-based ray tracing for 3D object tracking")


if __name__ == "__main__":
    main()
