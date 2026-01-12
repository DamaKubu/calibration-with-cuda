#!/usr/bin/env python
import sys
print(f"Python executable: {sys.executable}")

try:
    import cv2
    print(f"✓ OpenCV is installed!")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Create a test image
    import numpy as np
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Draw shapes
    cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), 3)  # Blue rectangle
    cv2.circle(img, (350, 100), 50, (0, 255, 0), 3)           # Green circle
    cv2.circle(img, (250, 400), 75, (0, 0, 255), -1)          # Red filled circle
    
    # Save the image
    if cv2.imwrite("opencv_test_python.png", img):
        print("✓ Image saved as opencv_test_python.png")
        print(f"Image shape: {img.shape}")
    else:
        print("✗ Failed to save image")
        
except ImportError as e:
    print(f"✗ OpenCV is NOT installed: {e}")
    print("\nTo install OpenCV, run:")
    print("pip install opencv-python")
