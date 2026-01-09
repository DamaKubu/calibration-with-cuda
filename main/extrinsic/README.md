# Extrinsic Camera Calibration Guide

## Overview

This guide covers extrinsic calibration for multi-camera setups with **CUDA acceleration** for optimal performance on your NVIDIA RTX 3050 Ti (4GB VRAM).

## Hardware Setup
- **GPU**: NVIDIA RTX 3050 Ti (4GB VRAM)
- **System RAM**: 32GB
- **Cameras**: Multiple overlapping cameras (fisheye model supported)

## Calibration Methods

### Method 1: Chessboard-Based Calibration (Recommended)

**Equipment**: 8x5 chessboard with 65mm squares

**Pros**:
- Highest accuracy (RMS error < 1 pixel achievable)
- Well-established method
- Robust corner detection

**Cons**:
- Requires good lighting
- Chessboard must be visible in all cameras simultaneously
- Limited to near-field calibration

**Steps**:
```bash
python main/extrinsic/simplest.py
```

### Method 2: Tennis Ball Tracking (Alternative)

**Equipment**: Standard tennis ball (67mm diameter)

**Pros**:
- Works at various distances
- Good for far-field calibration
- Easy to move around overlap region
- CUDA-accelerated color detection

**Cons**:
- Slightly lower accuracy than chessboard
- Requires consistent lighting
- Need ~50+ positions for good results

**Steps**:
```bash
python main/extrinsic/tennis_ball_calibration.py
```

## Calibration Workflow

### Step 1: Verify Intrinsic Calibration

Ensure you have `cameras_intrinsics.json` with:
- Camera matrix (K)
- Distortion coefficients (D)
- Fisheye model parameters

Check your current calibration:
```python
import json
with open('cameras_intrinsics.json') as f:
    data = json.load(f)
    for cam in data['cameras']:
        print(f"{cam['id']}: RMS error = {cam['rms_reprojection_error']:.3f}")
```

**Good intrinsic calibration**: RMS < 2.0 pixels

### Step 2: Prepare Calibration Target

#### Option A: Print Chessboard
1. Download pattern: https://calib.io/pages/camera-calibration-pattern-generator
2. Configure: 8 columns × 5 rows, 65mm squares
3. Print on rigid surface (foam board or acrylic)
4. Verify square size with ruler

#### Option B: Use Tennis Ball
1. Get standard tennis ball (Wilson, Penn, etc.)
2. Verify diameter: ~67mm
3. Ensure good yellow-green color (not faded)

### Step 3: Capture Calibration Data

#### Chessboard Method:
```python
# Run simplest.py and follow prompts
python main/extrinsic/simplest.py

# When capturing:
# - Hold chessboard in overlapping field of view
# - Vary distance: 0.5m to 3m
# - Vary angles: -45° to +45°
# - Capture 20-30 positions
# - Ensure good lighting
```

**Tips for best results**:
- Keep chessboard flat and rigid
- Cover the entire overlapping volume
- Include positions at different depths
- Avoid motion blur

#### Tennis Ball Method:
```python
python main/extrinsic/tennis_ball_calibration.py

# When tracking:
# - Move ball slowly through overlap region
# - Vary depth: near to far
# - Keep ball visible in ALL cameras
# - Collect 50+ positions
```

### Step 4: Review Calibration Results

Output file: `camera_extrinsics.json`

**Key metrics**:
- **RMS error**: < 1.0 = Excellent, < 2.0 = Good, < 3.0 = Acceptable
- **Baseline distance**: Physical distance between cameras (verify!)
- **Rotation angles**: Should match physical setup

**Example output**:
```
Rotation (Euler angles):
  Roll  (X):    2.345°  ← Small rotation around optical axis
  Pitch (Y):   -1.234°  ← Vertical angle difference
  Yaw   (Z):   45.678°  ← Horizontal angle difference

Translation vector (mm):
  X:   150.00  ← Horizontal separation
  Y:    10.00  ← Vertical offset
  Z:     5.00  ← Depth offset
  Baseline: 150.33 mm (15.03 cm)

RMS error: 0.847 pixels ← EXCELLENT!
```

### Step 5: Validate Calibration

Test calibration accuracy:
```python
from main.extrinsic.multi_camera_3d_tracking import MultiCameraTracker

tracker = MultiCameraTracker(
    'cameras_intrinsics.json',
    'camera_extrinsics.json'
)

# Place object at known position and measure reprojection error
# Should be < 2 pixels for good calibration
```

## CUDA Optimization

### Performance Tips

**Enable CUDA** (automatic detection):
```python
USE_CUDA = True  # Set in scripts
```

**Expected speedups**:
- Image preprocessing: 3-5x faster
- Feature detection enhancement: 2-3x faster
- Voxel operations: 10-20x faster

**Memory usage** (4GB VRAM):
- Calibration: ~500MB
- Voxel tracking (2m³): ~1-2GB
- Safe for multiple cameras

**Troubleshooting**:
```python
import cv2
print("CUDA available:", hasattr(cv2, 'cuda'))
print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())
```

If CUDA not available:
1. Check OpenCV build: `cv2.getBuildInformation()`
2. Install CUDA-enabled OpenCV:
   ```bash
   pip uninstall opencv-python
   pip install opencv-contrib-python
   ```
3. Or build from source with CUDA support

## Multi-Camera 3D Tracking

Once calibrated, use for 3D object tracking:

```python
from main.extrinsic.multi_camera_3d_tracking import MultiCameraTracker

tracker = MultiCameraTracker(
    'cameras_intrinsics.json',
    'camera_extrinsics.json',
    voxel_bounds=(-1000, 1000, -1000, 1000, 0, 5000),  # mm
    voxel_resolution=20.0  # mm
)

# Triangulate point from multiple views
observations = {
    'cam0': np.array([x0, y0]),
    'cam1': np.array([x1, y1]),
    # ... more cameras
}

point_3d = tracker.triangulate_point(observations)
print(f"3D position: {point_3d} mm")
```

### Voxel-Based Reconstruction

For dense 3D tracking:

```python
# Create binary masks for object in each view
detections = {
    'cam0': mask0,  # Binary image (255 = object)
    'cam1': mask1,
    # ...
}

# Carve voxel space
occupied_voxels = tracker.voxel_carving(detections, threshold=2)

# Convert to point cloud
points_3d = [tracker.voxel_space.voxel_to_world(v) for v in occupied_voxels]
```

## Parallax and Far-Object Tracking

For accurate far-object tracking, you need:

1. **Wide baseline**: Maximize camera separation
   - Formula: `depth_error = distance² / (baseline × focal_length)`
   - Example: 150mm baseline, 1000mm focal, 10m distance → ~67mm error
   - Increase baseline for far objects!

2. **Accurate angles**: Critical for ray intersection
   - 1° rotation error at 10m → ~175mm position error
   - This is why good extrinsic calibration matters

3. **Optimal configuration**:
   ```python
   # For 10m tracking distance:
   # - Baseline: 300-500mm (wider is better)
   # - Overlap angle: 20-40°
   # - Resolution: 1920×1080 minimum
   # - Focal length: 1000+ pixels
   ```

## Recommended Workflow for Your Setup

Based on your requirements (far-object tracking with parallax):

1. **Use chessboard** for initial calibration
   - Best accuracy for establishing camera geometry
   
2. **Verify with tennis ball** at target distance
   - Capture ball at 5m, 10m, 15m, etc.
   - Check triangulation accuracy
   
3. **Optimize camera placement**:
   - Position cameras with 30-40° overlap
   - Maximize baseline within mechanical constraints
   - Ensure parallel optical axes (minimize vergence)
   
4. **Implement tracking pipeline**:
   ```python
   # 1. Detect object in all views
   # 2. Triangulate 3D position
   # 3. Apply Kalman filter for smoothing
   # 4. Use voxel carving for shape estimation
   ```

## Troubleshooting

**Problem**: High RMS error (> 3 pixels)
- **Solution**: 
  - Increase number of calibration images
  - Improve lighting
  - Check for camera synchronization
  - Verify intrinsic calibration quality

**Problem**: Unrealistic baseline distance
- **Solution**:
  - Measure actual camera separation
  - Check camera assignment (may be swapped)
  - Recalibrate with better coverage

**Problem**: CUDA out of memory
- **Solution**:
  - Reduce voxel resolution
  - Process smaller workspace
  - Use CPU fallback (slower but works)

**Problem**: Poor triangulation at distance
- **Solution**:
  - Increase camera baseline
  - Improve lens quality (reduce aberrations)
  - Use higher resolution sensors
  - Recalibrate at target distance

## Files Created

1. **simplest.py**: Main chessboard-based calibration
2. **tennis_ball_calibration.py**: Alternative ball-tracking calibration
3. **multi_camera_3d_tracking.py**: 3D tracking and voxel reconstruction

## Next Steps

After successful calibration:

1. Test triangulation accuracy at target distances
2. Implement real-time object detection
3. Set up Kalman filtering for smooth tracking
4. Optimize voxel resolution for your use case
5. Consider bundle adjustment for > 2 cameras

## References

- OpenCV Fisheye Calibration: https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html
- CUDA in OpenCV: https://docs.opencv.org/4.x/d1/dfb/intro.html
- Multi-View Geometry: Hartley & Zisserman (2003)

## Support

For issues or questions, check:
- `camera_calibrations.txt` for notes
- `readme.txt` for system-specific info
- OpenCV documentation for API details
