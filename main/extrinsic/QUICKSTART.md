# Extrinsic Calibration - Quick Reference

## 🎯 Your Setup
- **Chessboard**: 8×5 (inner corners), 65mm squares
- **Hardware**: NVIDIA RTX 3050 Ti (4GB), 32GB RAM
- **Cameras**: Multiple overlapping cameras with barrel distortion
- **Goal**: Track far-away objects using parallax and ray tracing

## 🚀 Quick Start (3 Steps)

### Step 1: Run the Wizard
```bash
python main/extrinsic/quick_start.py
```
This will guide you through choosing the right calibration method.

### Step 2: Calibrate
Choose one based on wizard recommendation:

**Option A - Chessboard (Best for accuracy):**
```bash
python main/extrinsic/simplest.py
```

**Option B - Tennis Ball (Best for far objects):**
```bash
python main/extrinsic/tennis_ball_calibration.py
```

### Step 3: Validate
```bash
python main/extrinsic/validate_calibration.py
```

## 📁 Files Created

| File | Purpose | When to Use |
|------|---------|-------------|
| `simplest.py` | Chessboard calibration | Main calibration method |
| `tennis_ball_calibration.py` | Ball tracking | Alternative/validation |
| `multi_camera_3d_tracking.py` | 3D tracking & voxels | After calibration |
| `validate_calibration.py` | Quality check | After calibration |
| `quick_start.py` | Interactive wizard | First time setup |
| `README.md` | Full documentation | Reference |

## 🎓 What Each Method Does

### Chessboard Calibration
- **Accuracy**: ★★★★★ (0.5-1.0 pixel RMS)
- **Range**: Close to medium (0.5-3m)
- **Difficulty**: Easy
- **Time**: 5-10 minutes
- **Best for**: Highest precision baseline calibration

### Tennis Ball Calibration
- **Accuracy**: ★★★★☆ (1.0-2.0 pixel RMS)
- **Range**: Near to far (1-15m)
- **Difficulty**: Medium
- **Time**: 10-15 minutes
- **Best for**: Validating calibration at target distance

## 💡 For Your Use Case (Far Object Tracking)

Since you want to track far-away objects with parallax:

1. **Start with chessboard** - establishes accurate camera geometry
2. **Validate with tennis ball** at your target distances (5m, 10m, etc.)
3. **Key metric**: Rotation angles must be < 0.5° error for accurate parallax

### Depth Precision Formula
```
Depth Error = (Distance²) / (Baseline × Focal_Length) × Pixel_Error
```

**Example** (10m distance, 150mm baseline, 1000px focal, 1px error):
- Depth error ≈ 67mm at 10m
- To improve: Increase baseline or reduce pixel error

## ⚡ CUDA Performance

All scripts use CUDA by default. Expected performance:

| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Image preprocessing | 50ms | 15ms | 3× |
| Corner detection | 100ms | 30ms | 3× |
| Voxel operations | 2000ms | 100ms | 20× |

Your 4GB VRAM is sufficient for all operations.

## 📊 Quality Targets

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| RMS Error | < 1.0px | < 2.0px | < 3.0px | > 3.0px |
| Valid Images | > 25 | > 15 | > 10 | < 10 |
| Coverage | Full volume | Most volume | Partial | Limited |

## 🔧 Troubleshooting

**Problem**: Can't detect chessboard
- ✅ Improve lighting (diffuse, no shadows)
- ✅ Ensure chessboard is flat and rigid
- ✅ Check focus (no motion blur)
- ✅ Try different distances and angles

**Problem**: High RMS error (> 3px)
- ✅ Capture more images (30+ recommended)
- ✅ Cover entire overlap volume
- ✅ Check camera synchronization
- ✅ Verify intrinsic calibration quality

**Problem**: Unrealistic baseline distance
- ✅ Measure actual camera separation
- ✅ Check camera IDs (may be swapped)
- ✅ Recalibrate with better data

**Problem**: Poor far-field accuracy
- ✅ Increase camera baseline (wider separation)
- ✅ Validate calibration at target distance
- ✅ Check rotation angle accuracy
- ✅ Use higher resolution cameras

## 🎯 Next Steps After Calibration

1. **Test triangulation**:
   ```python
   from main.extrinsic.multi_camera_3d_tracking import MultiCameraTracker
   
   tracker = MultiCameraTracker(
       'cameras_intrinsics.json',
       'camera_extrinsics.json'
   )
   ```

2. **Implement object detection** in each camera view

3. **Triangulate 3D positions** from 2D detections

4. **Add temporal filtering** (Kalman filter)

5. **Set up voxel-based reconstruction** for dense tracking

## 📖 Full Documentation

See [README.md](README.md) for:
- Detailed workflow explanation
- Mathematical background
- Advanced optimization tips
- Multi-camera bundle adjustment
- Voxel ray tracing details

## 🆘 Need Help?

1. Check [README.md](README.md) for detailed explanations
2. Run `validate_calibration.py` to diagnose issues
3. Review `camera_calibrations.txt` for project notes
4. Check OpenCV docs: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

## 🎬 Typical Workflow

```
1. Quick Start Wizard
   ↓
2. Chessboard Calibration → camera_extrinsics.json
   ↓
3. Quality Validation → Check RMS error
   ↓
4. (Optional) Tennis Ball Validation → Verify at target distance
   ↓
5. 3D Tracking Implementation → Use multi_camera_3d_tracking.py
```

Good luck with your calibration! 🚀
