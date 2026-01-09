"""
Calibration Quality Assessment Tool
===================================

Validates and analyzes extrinsic calibration quality.
Provides diagnostics and recommendations for improvement.

Author: Calibration System
Date: January 2026
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple


class CalibrationValidator:
    """Validate and assess calibration quality."""
    
    def __init__(self, intrinsics_path: str, extrinsics_path: str):
        """Initialize validator with calibration data."""
        self.intrinsics = self._load_json(intrinsics_path)
        self.extrinsics = self._load_json(extrinsics_path)
        
    def _load_json(self, path: str) -> Dict:
        """Load JSON file."""
        with open(path) as f:
            return json.load(f)
    
    def rotation_matrix_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles in degrees."""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        if sy > 1e-6:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return tuple(np.degrees([x, y, z]))
    
    def assess_geometry(self, R: np.ndarray, T: np.ndarray) -> Dict:
        """Assess geometric configuration."""
        baseline = np.linalg.norm(T)
        angles = self.rotation_matrix_to_euler(R)
        
        # Calculate vergence angle (angle between optical axes)
        z1 = np.array([0, 0, 1])  # Optical axis of cam1
        z2 = R @ z1  # Optical axis of cam2 in cam1 frame
        vergence_angle = np.degrees(np.arccos(np.clip(z1 @ z2, -1, 1)))
        
        return {
            'baseline_mm': baseline,
            'baseline_cm': baseline / 10,
            'euler_angles': angles,
            'vergence_angle': vergence_angle
        }
    
    def estimate_depth_precision(self, baseline: float, focal_length: float,
                                 distance: float, pixel_error: float = 0.5) -> float:
        """
        Estimate depth measurement precision.
        
        Formula: δZ = (Z² / (f × B)) × δd
        where:
            Z = object distance
            f = focal length
            B = baseline
            δd = disparity error (in pixels)
        """
        depth_error = (distance ** 2) / (focal_length * baseline) * pixel_error
        return depth_error
    
    def check_calibration_quality(self):
        """Perform comprehensive quality check."""
        print("\n" + "="*70)
        print("CALIBRATION QUALITY ASSESSMENT")
        print("="*70 + "\n")
        
        # Check each camera pair
        for pair_key, pair_data in self.extrinsics.get('camera_pairs', {}).items():
            cam1_id, cam2_id = pair_key.split('_')
            
            print(f"\n{'─'*70}")
            print(f"Camera Pair: {cam1_id} ↔ {cam2_id}")
            print(f"{'─'*70}")
            
            R = np.array(pair_data['R'])
            T = np.array(pair_data['T'])
            rms = pair_data['rms_error']
            
            # Geometric assessment
            geom = self.assess_geometry(R, T)
            
            print(f"\n📐 Geometric Configuration:")
            print(f"   Baseline: {geom['baseline_cm']:.1f} cm ({geom['baseline_mm']:.1f} mm)")
            print(f"   Vergence angle: {geom['vergence_angle']:.2f}°")
            print(f"\n🔄 Rotation (Euler angles):")
            print(f"   Roll  (X): {geom['euler_angles'][0]:7.2f}°")
            print(f"   Pitch (Y): {geom['euler_angles'][1]:7.2f}°")
            print(f"   Yaw   (Z): {geom['euler_angles'][2]:7.2f}°")
            
            # RMS error assessment
            print(f"\n📊 Calibration Error:")
            print(f"   RMS reprojection: {rms:.3f} pixels")
            
            if rms < 1.0:
                quality = "🟢 EXCELLENT"
                recommendation = "Calibration is production-ready"
            elif rms < 2.0:
                quality = "🟡 GOOD"
                recommendation = "Suitable for most applications"
            elif rms < 3.0:
                quality = "🟠 ACCEPTABLE"
                recommendation = "Consider recalibration for critical applications"
            else:
                quality = "🔴 POOR"
                recommendation = "Recalibration strongly recommended"
            
            print(f"   Quality: {quality}")
            print(f"   Recommendation: {recommendation}")
            
            # Get focal lengths for depth precision
            cam1_intrinsics = next(c for c in self.intrinsics['cameras'] if c['id'] == cam1_id)
            K = np.array(cam1_intrinsics['fisheye']['K'])
            focal_length = (K[0, 0] + K[1, 1]) / 2  # Average focal length
            
            # Depth precision at various distances
            print(f"\n📏 Depth Measurement Precision (at {rms:.1f}px error):")
            print(f"   Distance | Depth Error | Relative Error")
            print(f"   ─────────────────────────────────────────")
            
            for distance in [1000, 2000, 5000, 10000]:  # mm
                error = self.estimate_depth_precision(
                    geom['baseline_mm'], focal_length, distance, rms
                )
                relative = (error / distance) * 100
                
                print(f"   {distance/1000:4.0f}m    | "
                      f"{error:7.1f} mm  | "
                      f"{relative:5.2f}%")
            
            # Baseline recommendations
            print(f"\n💡 Optimization Suggestions:")
            
            if geom['baseline_cm'] < 10:
                print(f"   ⚠️  Baseline ({geom['baseline_cm']:.1f} cm) is quite small")
                print(f"       → Consider increasing camera separation for far objects")
            elif geom['baseline_cm'] > 100:
                print(f"   ⚠️  Large baseline ({geom['baseline_cm']:.1f} cm)")
                print(f"       → Good for distant objects, may struggle with close range")
            else:
                print(f"   ✓ Baseline ({geom['baseline_cm']:.1f} cm) is reasonable")
            
            if abs(geom['vergence_angle']) > 10:
                print(f"   ⚠️  High vergence angle ({geom['vergence_angle']:.1f}°)")
                print(f"       → Consider more parallel camera alignment")
            else:
                print(f"   ✓ Vergence angle ({geom['vergence_angle']:.1f}°) is good")
            
            if abs(geom['euler_angles'][0]) > 5 or abs(geom['euler_angles'][1]) > 5:
                print(f"   ⚠️  Significant roll/pitch misalignment")
                print(f"       → Check camera mounting rigidity")
            
        # Overall summary
        print(f"\n{'═'*70}")
        print("OVERALL ASSESSMENT")
        print(f"{'═'*70}\n")
        
        avg_rms = np.mean([p['rms_error'] for p in self.extrinsics['camera_pairs'].values()])
        
        print(f"Average RMS error: {avg_rms:.3f} pixels")
        
        if avg_rms < 1.5:
            print("\n✅ Your calibration is EXCELLENT!")
            print("   Ready for high-precision 3D tracking and reconstruction")
        elif avg_rms < 2.5:
            print("\n✅ Your calibration is GOOD!")
            print("   Suitable for most 3D vision applications")
        else:
            print("\n⚠️  Calibration could be improved")
            print("\n   Recommendations:")
            print("   1. Capture more calibration images (30+ recommended)")
            print("   2. Cover larger volume of overlap space")
            print("   3. Ensure good lighting and sharp images")
            print("   4. Check camera synchronization")
            print("   5. Verify intrinsic calibration quality")
        
        print()
    
    def simulate_tracking_accuracy(self, target_distance: float = 10000.0):
        """
        Simulate tracking accuracy at a target distance.
        
        Args:
            target_distance: Distance to object in mm
        """
        print(f"\n{'═'*70}")
        print(f"TRACKING ACCURACY SIMULATION @ {target_distance/1000:.0f}m")
        print(f"{'═'*70}\n")
        
        for pair_key, pair_data in self.extrinsics['camera_pairs'].items():
            cam1_id, cam2_id = pair_key.split('_')
            
            R = np.array(pair_data['R'])
            T = np.array(pair_data['T'])
            rms = pair_data['rms_error']
            
            geom = self.assess_geometry(R, T)
            
            cam1_intrinsics = next(c for c in self.intrinsics['cameras'] if c['id'] == cam1_id)
            K = np.array(cam1_intrinsics['fisheye']['K'])
            focal_length = (K[0, 0] + K[1, 1]) / 2
            
            # Position error
            depth_error = self.estimate_depth_precision(
                geom['baseline_mm'], focal_length, target_distance, rms
            )
            
            # Angular error (simplified)
            angular_error = np.degrees(depth_error / target_distance)
            
            # Lateral error (from rotation calibration error, assume 0.1° error)
            rot_error_deg = 0.1  # Typical rotation calibration error
            lateral_error = target_distance * np.tan(np.radians(rot_error_deg))
            
            print(f"{cam1_id} ↔ {cam2_id}:")
            print(f"   Depth error:   ±{depth_error:.0f} mm ({depth_error/10:.1f} cm)")
            print(f"   Lateral error: ±{lateral_error:.0f} mm ({lateral_error/10:.1f} cm)")
            print(f"   Angular error: ±{angular_error:.3f}°")
            
            # Overall 3D position error
            total_3d_error = np.sqrt(depth_error**2 + lateral_error**2)
            print(f"   3D position:   ±{total_3d_error:.0f} mm ({total_3d_error/10:.1f} cm)")
            print()
        
        print("Note: These are theoretical estimates. Actual accuracy depends on:")
        print("  • Detection accuracy in each camera view")
        print("  • Synchronization between cameras")
        print("  • Environmental factors (lighting, motion blur)")
        print("  • Lens quality and aberrations")


def main():
    """Main validation workflow."""
    workspace = Path(__file__).parent.parent.parent
    intrinsics_path = workspace / "cameras_intrinsics.json"
    extrinsics_path = workspace / "camera_extrinsics.json"
    
    # Check files exist
    if not intrinsics_path.exists():
        print("❌ ERROR: cameras_intrinsics.json not found!")
        return
    
    if not extrinsics_path.exists():
        print("❌ ERROR: camera_extrinsics.json not found!")
        print("\nPlease run extrinsic calibration first:")
        print("  python main/extrinsic/simplest.py")
        return
    
    # Run validation
    validator = CalibrationValidator(str(intrinsics_path), str(extrinsics_path))
    validator.check_calibration_quality()
    
    # Simulate tracking at user's target distance
    print("\n" + "─"*70)
    target = input("\nEnter target tracking distance in meters (e.g., 10): ").strip()
    
    try:
        target_m = float(target)
        validator.simulate_tracking_accuracy(target_m * 1000)  # Convert to mm
    except ValueError:
        print("Invalid distance, skipping simulation")
    
    print("\n" + "="*70)
    print("Validation complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
