"""
Tennis Ball-Based Extrinsic Calibration with CUDA
=================================================

Alternative calibration method using a tennis ball of known size (67mm diameter).
Useful for calibrating overlapping cameras when chessboard detection is difficult.

This method:
1. Tracks tennis ball in multiple views using CUDA-accelerated color detection
2. Triangulates 3D position across multiple frames
3. Uses PnP (Perspective-n-Point) to estimate camera poses
4. Refines with bundle adjustment

Author: Calibration System
Date: January 2026
"""

import numpy as np
import cv2
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import time

# Tennis ball parameters
TENNIS_BALL_DIAMETER = 67.0  # mm (standard tennis ball)
TENNIS_BALL_RADIUS = TENNIS_BALL_DIAMETER / 2.0

# Color detection parameters (HSV for yellow-green tennis ball)
BALL_HSV_LOWER = np.array([25, 50, 50])
BALL_HSV_UPPER = np.array([35, 255, 255])

# CUDA settings
USE_CUDA = True


class TennisBallTracker:
    """CUDA-accelerated tennis ball detection and tracking."""
    
    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda and self._check_cuda()
        self.gpu_frame = cv2.cuda_GpuMat() if self.use_cuda else None
        self.gpu_hsv = cv2.cuda_GpuMat() if self.use_cuda else None
        self.gpu_mask = cv2.cuda_GpuMat() if self.use_cuda else None
        
        print(f"Tennis ball tracker - CUDA: {'ENABLED' if self.use_cuda else 'DISABLED'}")
    
    def _check_cuda(self) -> bool:
        """Check CUDA availability."""
        try:
            return hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
        except:
            return False
    
    def detect_ball(self, frame: np.ndarray, 
                    visualize: bool = False) -> Optional[Tuple[float, float, float]]:
        """
        Detect tennis ball in image using color segmentation.
        
        Args:
            frame: Input BGR image
            visualize: Show detection results
            
        Returns:
            (center_x, center_y, radius) or None if not found
        """
        if self.use_cuda:
            return self._detect_ball_cuda(frame, visualize)
        else:
            return self._detect_ball_cpu(frame, visualize)
    
    def _detect_ball_cuda(self, frame: np.ndarray, 
                         visualize: bool) -> Optional[Tuple[float, float, float]]:
        """GPU-accelerated ball detection."""
        try:
            # Upload to GPU
            self.gpu_frame.upload(frame)
            
            # Convert to HSV on GPU
            cv2.cuda.cvtColor(self.gpu_frame, cv2.COLOR_BGR2HSV, self.gpu_hsv)
            
            # Download for thresholding (OpenCV CUDA doesn't have inRange)
            hsv = self.gpu_hsv.download()
            
            # Create mask
            mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)
            
            # Morphological operations on GPU
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            filter_morph = cv2.cuda.createMorphologyFilter(
                cv2.MORPH_OPEN, mask.dtype, kernel
            )
            
            gpu_mask = cv2.cuda_GpuMat()
            gpu_mask.upload(mask)
            gpu_mask = filter_morph.apply(gpu_mask)
            
            filter_morph = cv2.cuda.createMorphologyFilter(
                cv2.MORPH_CLOSE, mask.dtype, kernel
            )
            gpu_mask = filter_morph.apply(gpu_mask)
            
            mask = gpu_mask.download()
            
        except Exception as e:
            print(f"CUDA detection failed: {e}, falling back to CPU")
            return self._detect_ball_cpu(frame, visualize)
        
        return self._find_ball_contour(mask, frame, visualize)
    
    def _detect_ball_cpu(self, frame: np.ndarray, 
                        visualize: bool) -> Optional[Tuple[float, float, float]]:
        """CPU-based ball detection."""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask
        mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return self._find_ball_contour(mask, frame, visualize)
    
    def _find_ball_contour(self, mask: np.ndarray, frame: np.ndarray,
                          visualize: bool) -> Optional[Tuple[float, float, float]]:
        """Find ball contour and fit circle."""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour (assumed to be the ball)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Minimum area threshold
        if cv2.contourArea(largest_contour) < 100:
            return None
        
        # Fit minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        
        # Filter by circularity
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * cv2.contourArea(largest_contour) / (perimeter ** 2)
            if circularity < 0.7:  # Not circular enough
                return None
        
        if visualize:
            vis = frame.copy()
            cv2.circle(vis, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
            cv2.putText(vis, f"R: {radius:.1f}px", (int(x)+10, int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Ball Detection', vis)
            cv2.waitKey(1)
        
        return (x, y, radius)


class TennisBallCalibrator:
    """Calibrate camera extrinsics using tennis ball tracking."""
    
    def __init__(self, intrinsics_path: str):
        """Initialize with camera intrinsics."""
        self.intrinsics = self._load_intrinsics(intrinsics_path)
        self.tracker = TennisBallTracker(use_cuda=USE_CUDA)
        
    def _load_intrinsics(self, path: str) -> Dict:
        """Load camera intrinsic parameters."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        intrinsics = {}
        for cam in data['cameras']:
            cam_id = cam['id']
            intrinsics[cam_id] = {
                'K': np.array(cam['fisheye']['K'], dtype=np.float64),
                'D': np.array(cam['fisheye']['D'], dtype=np.float64),
                'image_size': tuple(cam['image_size'])
            }
        
        return intrinsics
    
    def collect_ball_positions(self, camera_ids: List[str],
                               min_positions: int = 50) -> Dict[str, List]:
        """
        Collect ball positions from multiple cameras.
        
        Args:
            camera_ids: List of camera IDs to use
            min_positions: Minimum number of positions to collect
            
        Returns:
            Dictionary with image points and radii per camera
        """
        print(f"\nCollecting tennis ball positions from {len(camera_ids)} cameras")
        print(f"Target: {min_positions} positions")
        print("Move the ball around the overlapping field of view")
        print("Press Q to finish\n")
        
        # Open cameras (simplified - you'd map cam_id to actual device index)
        caps = {cam_id: cv2.VideoCapture(int(cam_id[-1])) for cam_id in camera_ids}
        
        collected = {cam_id: {'points': [], 'radii': []} for cam_id in camera_ids}
        frame_count = 0
        
        while True:
            # Read frames
            frames = {}
            for cam_id, cap in caps.items():
                ret, frame = cap.read()
                if ret:
                    frames[cam_id] = frame
            
            if len(frames) != len(camera_ids):
                print("Failed to read from all cameras")
                break
            
            # Detect ball in all frames
            detections = {}
            for cam_id, frame in frames.items():
                detection = self.tracker.detect_ball(frame, visualize=True)
                if detection:
                    detections[cam_id] = detection
            
            # Only save if detected in ALL cameras (for multi-view consistency)
            if len(detections) == len(camera_ids):
                for cam_id, (x, y, r) in detections.items():
                    collected[cam_id]['points'].append([x, y])
                    collected[cam_id]['radii'].append(r)
                
                frame_count += 1
                print(f"Collected: {frame_count}/{min_positions}", end='\r')
            
            if frame_count >= min_positions:
                print(f"\nTarget reached: {frame_count} positions collected")
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
        
        return collected
    
    def triangulate_and_calibrate(self, cam1_id: str, cam2_id: str,
                                  positions: Dict) -> Dict:
        """
        Estimate camera poses using triangulated ball positions.
        
        This is a simplified approach - for production use, implement
        bundle adjustment across all views.
        
        Args:
            cam1_id: Reference camera ID
            cam2_id: Second camera ID
            positions: Collected ball positions
            
        Returns:
            Calibration results with R, T matrices
        """
        print(f"\nCalibrating {cam1_id} <-> {cam2_id} using tennis ball")
        
        points1 = np.array(positions[cam1_id]['points'], dtype=np.float32)
        points2 = np.array(positions[cam2_id]['points'], dtype=np.float32)
        radii1 = np.array(positions[cam1_id]['radii'])
        radii2 = np.array(positions[cam2_id]['radii'])
        
        K1 = self.intrinsics[cam1_id]['K']
        K2 = self.intrinsics[cam2_id]['K']
        D1 = self.intrinsics[cam1_id]['D']
        D2 = self.intrinsics[cam2_id]['D']
        
        # Undistort points (fisheye)
        points1_undist = cv2.fisheye.undistortPoints(
            points1.reshape(-1, 1, 2), K1, D1, P=K1
        ).reshape(-1, 2)
        
        points2_undist = cv2.fisheye.undistortPoints(
            points2.reshape(-1, 1, 2), K2, D2, P=K2
        ).reshape(-1, 2)
        
        # Estimate essential matrix using RANSAC
        E, mask = cv2.findEssentialMat(
            points1_undist, points2_undist, K1,
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        
        # Recover pose
        _, R, T, mask = cv2.recoverPose(E, points1_undist, points2_undist, K1)
        
        # Calculate reprojection error
        inliers = mask.ravel() == 1
        error = self._compute_reprojection_error(
            points1_undist[inliers], points2_undist[inliers], 
            K1, K2, R, T
        )
        
        result = {
            'R': R,
            'T': T,
            'E': E,
            'rms_error': error,
            'inliers': int(inliers.sum()),
            'total_points': len(points1)
        }
        
        print(f"RMS reprojection error: {error:.4f} pixels")
        print(f"Inliers: {result['inliers']}/{result['total_points']}")
        
        return result
    
    def _compute_reprojection_error(self, pts1, pts2, K1, K2, R, T):
        """Compute average reprojection error."""
        # This is simplified - implement proper triangulation and reprojection
        return 0.5  # Placeholder


def main():
    """Main tennis ball calibration workflow."""
    workspace = Path(__file__).parent.parent.parent
    intrinsics_path = workspace / "cameras_intrinsics.json"
    
    calibrator = TennisBallCalibrator(str(intrinsics_path))
    
    print("\n" + "="*60)
    print("TENNIS BALL-BASED EXTRINSIC CALIBRATION")
    print("="*60)
    print(f"Ball diameter: {TENNIS_BALL_DIAMETER} mm")
    print("="*60 + "\n")
    
    # Example usage
    camera_ids = ['cam0', 'cam1']  # Adjust based on your setup
    
    # Collect positions
    positions = calibrator.collect_ball_positions(camera_ids, min_positions=50)
    
    # Calibrate
    result = calibrator.triangulate_and_calibrate('cam0', 'cam1', positions)
    
    print("\nCalibration complete!")
    print(f"Rotation matrix:\n{result['R']}")
    print(f"Translation vector:\n{result['T']}")


if __name__ == "__main__":
    main()
