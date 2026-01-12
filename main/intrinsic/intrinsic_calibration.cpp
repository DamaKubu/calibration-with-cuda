#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <sstream>
#include <chrono>

namespace fs = std::filesystem;

// Quality check: Measure if frame provides good diversity
float computeFrameQuality(const std::vector<cv::Point2f>& corners, cv::Size frameSize) {
    if (corners.empty()) return 0.0f;
    
    float minX = corners[0].x, maxX = corners[0].x;
    float minY = corners[0].y, maxY = corners[0].y;
    
    for (const auto& c : corners) {
        minX = std::min(minX, c.x);
        maxX = std::max(maxX, c.x);
        minY = std::min(minY, c.y);
        maxY = std::max(maxY, c.y);
    }
    
    float spreadX = (maxX - minX) / frameSize.width;
    float spreadY = (maxY - minY) / frameSize.height;
    return (spreadX + spreadY) / 2.0f;
}

int main(int argc, char** argv) {
    std::cout << "=== INTRINSIC CAMERA CALIBRATION (AUTO-CAPTURE) ===\n\n";

    int camera_id = 0;
    if (argc > 1) camera_id = std::stoi(argv[1]);

    cv::Size patternSize(8, 5);
    float squareSize = 65.0f;
    cv::VideoCapture cap(camera_id);

    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open camera " << camera_id << "\n";
        return 1;
    }

    std::cout << "Camera " << camera_id << " opened\n";
    std::cout << "AUTO-CAPTURE MODE: Program collects 30 high-quality frames automatically\n";
    std::cout << "Controls:\n";
    std::cout << "  [SPACE] - Force calibration early (min 10 images)\n";
    std::cout << "  [ESC] - Cancel\n\n";

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<cv::Point3f> objPts;

    for (int y = 0; y < patternSize.height; y++) {
        for (int x = 0; x < patternSize.width; x++) {
            objPts.push_back(cv::Point3f(x * squareSize, y * squareSize, 0.0f));
        }
    }

    cv::Mat frame, gray;
    int captured = 0;
    int frame_count = 0;
    auto last_capture = std::chrono::steady_clock::now();
    const int capture_interval_ms = 500;

    while (captured < 30) {
        cap >> frame;
        if (frame.empty()) break;

        frame_count++;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, patternSize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            // High-precision corner refinement for best quality
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 50, 0.0001));
            cv::drawChessboardCorners(frame, patternSize, corners, found);

            // Check if enough time has passed since last capture
            auto now = std::chrono::steady_clock::now();
            int elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_capture).count();

            if (elapsed_ms >= capture_interval_ms) {
                // Quality check: ensure good frame diversity
                float quality = computeFrameQuality(corners, gray.size());
                
                if (quality > 0.3f) {
                    objectPoints.push_back(objPts);
                    imagePoints.push_back(corners);
                    captured++;
                    last_capture = now;
                    std::cout << "AUTO-CAPTURED: " << captured << "/30 (quality: " 
                              << quality << ")\n";
                }
            }
        }

        cv::putText(frame, "AUTO-CAPTURE: " + std::to_string(captured) + "/30", 
            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        if (found) {
            cv::putText(frame, "PATTERN FOUND", cv::Point(10, 80),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::putText(frame, "Searching for chessboard...", cv::Point(10, 80),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }

        cv::putText(frame, "(Frames: " + std::to_string(frame_count) + ")", 
            cv::Point(10, 130), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);

        cv::imshow("Intrinsic Calibration - Camera " + std::to_string(camera_id), frame);

        int key = cv::waitKey(30) & 0xFF;
        if (key == 27) {
            std::cout << "Cancelled by user\n";
            cap.release();
            cv::destroyAllWindows();
            return 1;
        }
        else if ((key == 32 || key == 's' || key == 'S') && captured >= 10) {
            std::cout << "Early calibration triggered\n";
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    if (captured < 10) {
        std::cerr << "ERROR: Collected only " << captured << " images. Need at least 10.\n";
        return 1;
    }

    std::cout << "\n=== CALIBRATING WITH " << captured << " IMAGES ===\n";

    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    std::vector<cv::Mat> rvecs, tvecs;

    double rms = cv::calibrateCamera(objectPoints, imagePoints, gray.size(),
        cameraMatrix, distCoeffs, rvecs, tvecs, 
        cv::CALIB_FIX_PRINCIPAL_POINT | cv::CALIB_FIX_ASPECT_RATIO);

    std::cout << "\n=== CALIBRATION RESULTS ===\n";
    std::cout << "RMS Reprojection Error: " << rms << " pixels\n";
    if (rms < 0.5) {
        std::cout << "Quality: EXCELLENT\n";
    } else if (rms < 1.0) {
        std::cout << "Quality: GOOD\n";
    } else {
        std::cout << "Quality: ACCEPTABLE (consider recalibrating)\n";
    }

    std::cout << "\nCamera Matrix:\n" << cameraMatrix << "\n";
    std::cout << "\nDistortion Coefficients (k1, k2, p1, p2, k3):\n" << distCoeffs.t() << "\n";

    // Save intrinsic calibration with full error handling
    std::string filename = "intrinsic_camera_" + std::to_string(camera_id) + ".yml";
    
    try {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }
        
        fs << "camera_id" << camera_id;
        fs << "camera_matrix" << cameraMatrix;
        fs << "distortion_coefficients" << distCoeffs;
        fs << "rms_error" << rms;
        fs << "calibration_images" << captured;
        fs << "pattern_size_w" << patternSize.width;
        fs << "pattern_size_h" << patternSize.height;
        fs << "square_size_mm" << squareSize;
        fs.release();

        std::cout << "\n✓ Successfully saved to: " << filename << "\n";
        
        // Verify file was written
        if (fs::exists(filename)) {
            auto size = fs::file_size(filename);
            std::cout << "✓ File verified: " << size << " bytes\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ ERROR saving calibration: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\nCalibration complete! Ready for extrinsic calibration.\n";

    return 0;
}
