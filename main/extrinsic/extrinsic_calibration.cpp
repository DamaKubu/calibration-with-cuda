#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    std::cout << "=== EXTRINSIC CAMERA CALIBRATION (Multi-Camera) ===\n\n";

    if (argc < 3) {
        std::cerr << "Usage: extrinsic_calibration.exe <camera1_id> <camera2_id>\n";
        std::cerr << "Example: extrinsic_calibration.exe 0 1\n";
        return 1;
    }

    int cam1_id = std::stoi(argv[1]);
    int cam2_id = std::stoi(argv[2]);

    cv::VideoCapture cap1(cam1_id);
    cv::VideoCapture cap2(cam2_id);

    if (!cap1.isOpened() || !cap2.isOpened()) {
        std::cerr << "ERROR: Cannot open cameras\n";
        return 1;
    }

    std::cout << "Camera " << cam1_id << " and " << cam2_id << " opened\n";
    std::cout << "Controls:\n";
    std::cout << "  [SPACE] - Capture frame pair\n";
    std::cout << "  [C] - Calibrate extrinsic\n";
    std::cout << "  [ESC] - Exit\n\n";

    cv::Size patternSize(8, 5);
    float squareSize = 65.0f;
    std::vector<cv::Point3f> objPts;

    for (int y = 0; y < patternSize.height; y++) {
        for (int x = 0; x < patternSize.width; x++) {
            objPts.push_back(cv::Point3f(x * squareSize, y * squareSize, 0.0f));
        }
    }

    // Load intrinsic parameters (assuming they exist)
    cv::Mat K1 = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat D1 = cv::Mat::zeros(5, 1, CV_64F);
    cv::Mat K2 = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat D2 = cv::Mat::zeros(5, 1, CV_64F);

    std::string file1 = "intrinsic_camera_" + std::to_string(cam1_id) + ".yml";
    std::string file2 = "intrinsic_camera_" + std::to_string(cam2_id) + ".yml";

    cv::FileStorage fs1(file1, cv::FileStorage::READ);
    cv::FileStorage fs2(file2, cv::FileStorage::READ);

    if (fs1.isOpened()) {
        fs1["camera_matrix"] >> K1;
        fs1["distortion_coefficients"] >> D1;
        fs1.release();
        std::cout << "Loaded intrinsics for camera " << cam1_id << "\n";
    } else {
        std::cout << "WARNING: Could not load intrinsics for camera " << cam1_id 
                  << " - using identity\n";
    }

    if (fs2.isOpened()) {
        fs2["camera_matrix"] >> K2;
        fs2["distortion_coefficients"] >> D2;
        fs2.release();
        std::cout << "Loaded intrinsics for camera " << cam2_id << "\n";
    } else {
        std::cout << "WARNING: Could not load intrinsics for camera " << cam2_id 
                  << " - using identity\n";
    }

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints1, imagePoints2;
    int captured = 0;

    cv::Mat frame1, frame2, gray1, gray2;

    while (true) {
        cap1 >> frame1;
        cap2 >> frame2;

        if (frame1.empty() || frame2.empty()) break;

        cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners1, corners2;
        bool found1 = cv::findChessboardCorners(gray1, patternSize, corners1,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        bool found2 = cv::findChessboardCorners(gray2, patternSize, corners2,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found1) {
            cv::cornerSubPix(gray1, corners1, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01));
            cv::drawChessboardCorners(frame1, patternSize, corners1, true);
        }
        if (found2) {
            cv::cornerSubPix(gray2, corners2, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01));
            cv::drawChessboardCorners(frame2, patternSize, corners2, true);
        }

        cv::putText(frame1, "Cam " + std::to_string(cam1_id) + ": " + 
            (found1 ? "FOUND" : "Searching"), cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, found1 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        cv::putText(frame2, "Cam " + std::to_string(cam2_id) + ": " + 
            (found2 ? "FOUND" : "Searching"), cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, found2 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);

        cv::putText(frame1, "Pairs: " + std::to_string(captured), cv::Point(10, 80),
            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 0), 2);

        cv::imshow("Camera " + std::to_string(cam1_id), frame1);
        cv::imshow("Camera " + std::to_string(cam2_id), frame2);

        int key = cv::waitKey(30) & 0xFF;
        if (key == 27) break;
        else if (key == 32 && found1 && found2) {
            objectPoints.push_back(objPts);
            imagePoints1.push_back(corners1);
            imagePoints2.push_back(corners2);
            captured++;
            std::cout << "Captured pair: " << captured << "\n";
        }
        else if ((key == 'c' || key == 'C') && captured >= 3) {
            break;
        }
    }

    cap1.release();
    cap2.release();
    cv::destroyAllWindows();

    if (captured < 3) {
        std::cerr << "ERROR: Need at least 3 image pairs\n";
        return 1;
    }

    std::cout << "\n=== COMPUTING EXTRINSIC PARAMETERS ===\n";

    cv::Mat R, T;
    cv::Mat E, F;

    // Compute fundamental matrix
    std::vector<uchar> mask;
    F = cv::findFundamentalMat(imagePoints1[0], imagePoints2[0], 
        cv::FM_RANSAC, 1.0, 0.99, mask);

    std::cout << "Fundamental Matrix:\n" << F << "\n";

    // Compute essential matrix from F
    E = K2.t() * F * K1;

    std::cout << "Essential Matrix:\n" << E << "\n";

    // Recover pose
    cv::recoverPose(E, imagePoints1[0], imagePoints2[0], K1, K2, R, T, mask);

    std::cout << "\nRotation (Camera 2 relative to Camera 1):\n" << R << "\n";
    std::cout << "\nTranslation (Camera 2 relative to Camera 1):\n" << T << "\n";

    // Save extrinsic parameters
    std::string outfile = "extrinsic_cam" + std::to_string(cam1_id) + 
        "_cam" + std::to_string(cam2_id) + ".yml";

    cv::FileStorage out(outfile, cv::FileStorage::WRITE);
    out << "rotation" << R;
    out << "translation" << T;
    out << "calibration_pairs" << captured;
    out.release();

    std::cout << "\nSaved to: " << outfile << "\n";

    return 0;
}
