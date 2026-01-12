#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    std::cout << "Camera Calibration Tool\n";
    std::cout << "=====================\n\n";

    // Parse camera index from command line
    int camera_id = 0;
    if (argc > 1) {
        camera_id = std::stoi(argv[1]);
    }

    // Chessboard pattern
    cv::Size patternSize(8, 5);  // 8x5 inner corners
    float squareSize = 65.0f;    // mm

    // Open camera
    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open camera " << camera_id << "\n";
        std::cerr << "Usage: program.exe [camera_id]\n";
        std::cerr << "Example: program.exe 0\n";
        return 1;
    }

    std::cout << "Camera " << camera_id << " opened successfully\n";
    std::cout << "Controls:\n";
    std::cout << "  [SPACE] - Capture frame\n";
    std::cout << "  [C] - Start calibration\n";
    std::cout << "  [ESC] - Exit\n\n";

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    cv::Mat frame, gray;
    int frameCount = 0;
    int capturedCount = 0;

    // Create object points template
    std::vector<cv::Point3f> objPts;
    for (int y = 0; y < patternSize.height; y++) {
        for (int x = 0; x < patternSize.width; x++) {
            objPts.push_back(cv::Point3f(x * squareSize, y * squareSize, 0.0f));
        }
    }

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Failed to read frame\n";
            break;
        }

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Try to find chessboard
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, patternSize, corners,
                                             cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        // Refine corner positions
        if (found) {
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                           cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01));
            cv::drawChessboardCorners(frame, patternSize, corners, found);
        }

        // Display info
        cv::putText(frame, "Frames: " + std::to_string(frameCount), cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, "Captured: " + std::to_string(capturedCount), cv::Point(10, 70),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        if (found) {
            cv::putText(frame, "Chessboard FOUND - Press SPACE to capture", cv::Point(10, 110),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::putText(frame, "Searching for chessboard...", cv::Point(10, 110),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("Camera Calibration", frame);
        frameCount++;

        int key = cv::waitKey(30) & 0xFF;
        if (key == 27) {  // ESC
            break;
        } else if (key == 32 && found) {  // SPACE
            objectPoints.push_back(objPts);
            imagePoints.push_back(corners);
            capturedCount++;
            std::cout << "Captured frame #" << capturedCount << "\n";
        } else if (key == 'c' || key == 'C') {  // Calibrate
            if (capturedCount < 3) {
                std::cout << "Need at least 3 images for calibration\n";
                continue;
            }

            std::cout << "\nStarting calibration with " << capturedCount << " images...\n";

            cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
            cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
            std::vector<cv::Mat> rvecs, tvecs;

            double rms = cv::calibrateCamera(objectPoints, imagePoints, gray.size(),
                                            cameraMatrix, distCoeffs, rvecs, tvecs);

            std::cout << "Calibration complete!\n";
            std::cout << "RMS Reprojection Error: " << rms << "\n";
            std::cout << "\nCamera Matrix:\n" << cameraMatrix << "\n";
            std::cout << "\nDistortion Coefficients:\n" << distCoeffs << "\n";

            // Save results
            cv::FileStorage fs("calibration_camera_" + std::to_string(camera_id) + ".xml",
                             cv::FileStorage::WRITE);
            if (fs.isOpened()) {
                fs << "camera_matrix" << cameraMatrix;
                fs << "distortion_coefficients" << distCoeffs;
                fs << "rms_error" << rms;
                fs << "images_used" << capturedCount;
                fs.release();
                std::cout << "Results saved to calibration_camera_" << camera_id << ".xml\n";
            }

            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
