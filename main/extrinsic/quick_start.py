"""
Quick Start - Extrinsic Calibration Wizard
==========================================

Interactive script to guide you through extrinsic calibration.
Helps choose the right method and validates your setup.

Author: Calibration System
Date: January 2026
"""

import json
import sys
from pathlib import Path

import numpy as np


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")


def check_intrinsic_calibration():
    """Verify intrinsic calibration quality."""
    intrinsics_path = Path(__file__).parent.parent.parent / "cameras_intrinsics.json"
    
    if not intrinsics_path.exists():
        print("❌ ERROR: cameras_intrinsics.json not found!")
        print("\nYou need to complete intrinsic calibration first.")
        print("Run one of these scripts:")
        print("  - main/intrinsic/calibrate_camera.py")
        print("  - main/intrinsic/intrinsic_calibration_cuda.py")
        return False
    
    with open(intrinsics_path) as f:
        data = json.load(f)
    
    print("✓ Intrinsic calibration found\n")
    print("Camera Summary:")
    print("-" * 70)
    
    good_cams = []
    for cam in data['cameras']:
        cam_id = cam['id']
        rms = cam.get('rms_reprojection_error', 999)
        img_size = cam.get('image_size', [0, 0])
        
        status = "✓ GOOD" if rms < 2.0 else "⚠ ACCEPTABLE" if rms < 3.0 else "❌ POOR"
        
        print(f"{cam_id:8s} | {img_size[0]:4d}×{img_size[1]:<4d} | RMS: {rms:5.2f}px | {status}")
        
        if rms < 3.0:
            good_cams.append(cam_id)
    
    print("-" * 70)
    
    if len(good_cams) < 2:
        print("\n❌ Need at least 2 well-calibrated cameras for extrinsic calibration")
        print("Please improve intrinsic calibration first")
        return False
    
    print(f"\n✓ {len(good_cams)} cameras ready for extrinsic calibration")
    return True


def get_calibration_requirements():
    """Ask user about their calibration requirements."""
    print_header("Calibration Requirements")
    
    print("Please answer a few questions to determine the best method:\n")
    
    # Question 1: Typical object distance
    print("1. What distance will you typically track objects at?")
    print("   a) Close range (< 2 meters)")
    print("   b) Medium range (2-5 meters)")
    print("   c) Far range (> 5 meters)")
    distance = input("\n   Your choice (a/b/c): ").strip().lower()
    
    # Question 2: Camera overlap
    print("\n2. Do your cameras have significant overlap?")
    print("   a) Yes, they cover mostly the same area")
    print("   b) Some overlap at the edges")
    print("   c) Minimal or no overlap")
    overlap = input("\n   Your choice (a/b/c): ").strip().lower()
    
    # Question 3: Lighting conditions
    print("\n3. What are your lighting conditions?")
    print("   a) Controlled indoor lighting")
    print("   b) Variable indoor/outdoor")
    print("   c) Difficult lighting (low light, outdoor sun)")
    lighting = input("\n   Your choice (a/b/c): ").strip().lower()
    
    # Question 4: Available equipment
    print("\n4. What calibration equipment do you have?")
    print("   a) Printed chessboard (8×5, 65mm squares)")
    print("   b) Tennis ball or similar trackable object")
    print("   c) Both")
    print("   d) Neither yet")
    equipment = input("\n   Your choice (a/b/c/d): ").strip().lower()
    
    return {
        'distance': distance,
        'overlap': overlap,
        'lighting': lighting,
        'equipment': equipment
    }


def recommend_method(requirements):
    """Recommend calibration method based on requirements."""
    print_header("Calibration Method Recommendation")
    
    score_chessboard = 0
    score_tennis = 0
    
    # Analyze distance
    if requirements['distance'] == 'a':
        score_chessboard += 3
        score_tennis += 1
    elif requirements['distance'] == 'b':
        score_chessboard += 2
        score_tennis += 2
    else:  # Far range
        score_chessboard += 0
        score_tennis += 3
    
    # Analyze overlap
    if requirements['overlap'] == 'a':
        score_chessboard += 3
        score_tennis += 2
    elif requirements['overlap'] == 'b':
        score_chessboard += 2
        score_tennis += 3
    else:
        score_chessboard += 0
        score_tennis += 2
    
    # Analyze lighting
    if requirements['lighting'] == 'a':
        score_chessboard += 3
        score_tennis += 2
    elif requirements['lighting'] == 'b':
        score_chessboard += 1
        score_tennis += 2
    else:
        score_chessboard += 0
        score_tennis += 3
    
    # Determine recommendation
    if score_chessboard > score_tennis + 1:
        method = 'chessboard'
    elif score_tennis > score_chessboard + 1:
        method = 'tennis'
    else:
        method = 'both'
    
    # Print recommendation
    print("Based on your requirements:\n")
    
    if method == 'chessboard':
        print("🎯 RECOMMENDED: Chessboard-Based Calibration")
        print("\nReasons:")
        print("  ✓ Close-range tracking")
        print("  ✓ Good camera overlap")
        print("  ✓ Controlled lighting")
        print("  ✓ Highest accuracy possible (~0.5-1.0 pixel RMS)")
        
        print("\n📋 What you need:")
        print("  1. 8×5 chessboard with 65mm squares")
        print("  2. Rigid mount (foam board, acrylic, or wood)")
        print("  3. Good lighting in overlap area")
        
        print("\n🚀 To start:")
        print("  python main/extrinsic/simplest.py")
        
    elif method == 'tennis':
        print("🎾 RECOMMENDED: Tennis Ball Tracking Calibration")
        print("\nReasons:")
        print("  ✓ Far-range tracking capability")
        print("  ✓ Works with limited overlap")
        print("  ✓ Robust to lighting variations")
        print("  ✓ Easy to move through 3D space")
        
        print("\n📋 What you need:")
        print("  1. Standard tennis ball (67mm diameter)")
        print("  2. Good yellow-green color (not faded)")
        print("  3. Ability to move ball through overlap volume")
        
        print("\n🚀 To start:")
        print("  python main/extrinsic/tennis_ball_calibration.py")
        
    else:
        print("🎯 RECOMMENDED: Hybrid Approach")
        print("\nWhy both methods:")
        print("  ✓ Chessboard for high-accuracy near-field calibration")
        print("  ✓ Tennis ball to validate at target distance")
        print("  ✓ Cross-validation improves confidence")
        
        print("\n📋 What you need:")
        print("  1. 8×5 chessboard with 65mm squares")
        print("  2. Standard tennis ball (67mm diameter)")
        
        print("\n🚀 Recommended workflow:")
        print("  1. python main/extrinsic/simplest.py              (chessboard)")
        print("  2. python main/extrinsic/tennis_ball_calibration.py  (validation)")
        print("  3. Compare results and use best RMS error")
    
    print("\n" + "-"*70)
    
    return method


def print_optimization_tips():
    """Print CUDA optimization tips."""
    print_header("CUDA Optimization Tips")
    
    print("Your hardware: NVIDIA RTX 3050 Ti (4GB VRAM)")
    print("\nOptimization settings for best performance:\n")
    
    print("✓ CUDA acceleration is ENABLED by default")
    print("✓ Expected speedup: 3-5x for image processing")
    print("✓ Voxel operations: 10-20x faster\n")
    
    print("Memory usage estimates:")
    print("  • Chessboard calibration: ~500 MB VRAM")
    print("  • Tennis ball tracking: ~300 MB VRAM")
    print("  • Voxel space (2×2×5m, 20mm res): ~1.5 GB VRAM")
    print("\n✓ All operations fit comfortably in 4GB VRAM\n")
    
    print("If you experience issues:")
    print("  1. Close other GPU-intensive applications")
    print("  2. Reduce image resolution if needed")
    print("  3. Scripts auto-fallback to CPU if CUDA unavailable")


def print_next_steps():
    """Print what to do after calibration."""
    print_header("After Calibration: Next Steps")
    
    print("Once extrinsic calibration is complete, you'll have:\n")
    print("  📄 camera_extrinsics.json - Camera poses and angles\n")
    
    print("Use this for:\n")
    
    print("1️⃣  3D Object Tracking:")
    print("   python main/extrinsic/multi_camera_3d_tracking.py")
    print("   • Triangulate object positions from multiple views")
    print("   • Track motion in 3D space")
    print("   • Compute parallax for depth estimation\n")
    
    print("2️⃣  Voxel-Based Reconstruction:")
    print("   • Dense 3D scene reconstruction")
    print("   • Space carving from silhouettes")
    print("   • Ray tracing with CUDA acceleration\n")
    
    print("3️⃣  Far-Object Tracking (your use case):")
    print("   • Optimized for parallax-based depth")
    print("   • Accurate angle calibration critical")
    print("   • Works with distant objects (10m+)\n")
    
    print("📖 See main/extrinsic/README.md for detailed guide")


def main():
    """Main wizard workflow."""
    print_header("Extrinsic Camera Calibration Wizard")
    
    print("This wizard will help you:")
    print("  • Verify your intrinsic calibration")
    print("  • Choose the best extrinsic calibration method")
    print("  • Get started with CUDA-accelerated calibration")
    
    input("\nPress ENTER to continue...")
    
    # Step 1: Check intrinsic calibration
    print_header("Step 1: Verify Intrinsic Calibration")
    
    if not check_intrinsic_calibration():
        sys.exit(1)
    
    input("\nPress ENTER to continue...")
    
    # Step 2: Get requirements
    requirements = get_calibration_requirements()
    
    # Step 3: Recommend method
    method = recommend_method(requirements)
    
    # Step 4: Equipment check
    if requirements['equipment'] == 'd':
        print("\n⚠️  You'll need to acquire calibration equipment first")
        print("\nWhere to get it:")
        print("  • Chessboard: Print from https://calib.io/pages/camera-calibration-pattern-generator")
        print("  • Tennis ball: Any sporting goods store (~$3)")
    
    input("\nPress ENTER to continue...")
    
    # Step 5: Optimization tips
    print_optimization_tips()
    
    input("\nPress ENTER to continue...")
    
    # Step 6: Next steps
    print_next_steps()
    
    # Final summary
    print_header("Summary")
    
    print("You're ready to calibrate! 🚀\n")
    
    if method == 'chessboard' or method == 'both':
        print("👉 Run: python main/extrinsic/simplest.py")
    if method == 'tennis' or method == 'both':
        print("👉 Run: python main/extrinsic/tennis_ball_calibration.py")
    
    print("\n📖 Full documentation: main/extrinsic/README.md")
    print("\nGood luck with your calibration! 🎯\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWizard cancelled by user")
        sys.exit(0)
