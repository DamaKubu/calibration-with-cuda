#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

static bool tryParseInt(const std::string& s, int& out) {
    try {
        size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (pos != s.size()) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

static bool loadIntrinsicsFromFile(const fs::path& path, cv::Mat& K, cv::Mat& D) {
    cv::FileStorage fs(path.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) return false;

    cv::Mat k, d;
    fs["camera_matrix"] >> k;
    fs["distortion_coefficients"] >> d;
    fs.release();

    if (k.empty() || d.empty()) return false;
    if (k.rows != 3 || k.cols != 3) return false;

    K = k;
    D = d;
    return true;
}

static fs::path findFirstExisting(const std::vector<fs::path>& candidates) {
    for (const auto& p : candidates) {
        std::error_code ec;
        if (fs::exists(p, ec) && !ec) return p;
    }
    return {};
}

static bool openCameraFromArg(const std::string& arg, cv::VideoCapture& cap, int& outIndex, std::string& outLabel) {
    // On Windows, CAP_DSHOW tends to give stable device ordering and supports opening by name via "video=<name>".
    int idx = -1;
    if (tryParseInt(arg, idx)) {
        outIndex = idx;
        outLabel = "index " + std::to_string(idx);
        return cap.open(idx, cv::CAP_DSHOW);
    }

    outIndex = -1;
    outLabel = "name '" + arg + "'";
    return cap.open("video=" + arg, cv::CAP_DSHOW);
}

int main(int argc, char** argv) {
    std::cout << "=== EXTRINSIC CAMERA CALIBRATION (Multi-Camera) ===\n\n";

    if (argc < 3) {
        std::cerr << "Usage: extrinsic_calibration.exe <cam1> <cam2> [options]\n";
        std::cerr << "  <camX> can be an integer index (e.g. 0) OR a DirectShow friendly name (e.g. \"Integrated Camera\")\n";
        std::cerr << "Options:\n";
        std::cerr << "  --intr1 <path>        Intrinsics file for cam1 (OpenCV .yml with camera_matrix, distortion_coefficients)\n";
        std::cerr << "  --intr2 <path>        Intrinsics file for cam2\n";
        std::cerr << "  --intrdir <dir>       Directory containing intrinsic_camera_<id>.yml for numeric cam ids\n";
        std::cerr << "  --fisheye             Force fisheye stereo calibration (expects D with 4 params)\n";
        std::cerr << "  --pinhole             Force pinhole stereo calibration\n";
        std::cerr << "Example (by index): extrinsic_calibration.exe 0 1 --intrdir ..\\..\\intrinsic\n";
        std::cerr << "Example (by name):  extrinsic_calibration.exe \"Integrated Camera\" \"FHD Camera\" --intr1 path1.yml --intr2 path2.yml\n";
        return 1;
    }

    std::string cam1_arg = argv[1];
    std::string cam2_arg = argv[2];

    fs::path intr1Path;
    fs::path intr2Path;
    fs::path intrDir;
    bool forceFisheye = false;
    bool forcePinhole = false;

    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--intr1" && i + 1 < argc) {
            intr1Path = fs::path(argv[++i]);
        } else if (a == "--intr2" && i + 1 < argc) {
            intr2Path = fs::path(argv[++i]);
        } else if (a == "--intrdir" && i + 1 < argc) {
            intrDir = fs::path(argv[++i]);
        } else if (a == "--fisheye") {
            forceFisheye = true;
        } else if (a == "--pinhole") {
            forcePinhole = true;
        } else {
            std::cerr << "Unknown/invalid option: " << a << "\n";
            return 1;
        }
    }

    int cam1_id = -1;
    int cam2_id = -1;
    std::string cam1_label, cam2_label;

    cv::VideoCapture cap1;
    cv::VideoCapture cap2;
    if (!openCameraFromArg(cam1_arg, cap1, cam1_id, cam1_label) ||
        !openCameraFromArg(cam2_arg, cap2, cam2_id, cam2_label)) {
        std::cerr << "ERROR: Cannot open cameras (CAP_DSHOW).\n";
        std::cerr << "  Cam1: " << cam1_label << "\n";
        std::cerr << "  Cam2: " << cam2_label << "\n";
        return 1;
    }

    std::cout << "Opened cameras (CAP_DSHOW):\n";
    std::cout << "  Cam1: " << cam1_label << "\n";
    std::cout << "  Cam2: " << cam2_label << "\n";
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

    // Load intrinsics (REQUIRED for accurate extrinsics)
    cv::Mat K1, D1, K2, D2;

    if (intr1Path.empty() && !intrDir.empty() && cam1_id >= 0) {
        intr1Path = intrDir / ("intrinsic_camera_" + std::to_string(cam1_id) + ".yml");
    }
    if (intr2Path.empty() && !intrDir.empty() && cam2_id >= 0) {
        intr2Path = intrDir / ("intrinsic_camera_" + std::to_string(cam2_id) + ".yml");
    }

    if (intr1Path.empty() && cam1_id >= 0) {
        intr1Path = findFirstExisting({
            fs::path("intrinsic_camera_" + std::to_string(cam1_id) + ".yml"),
            fs::path("..") / ("intrinsic_camera_" + std::to_string(cam1_id) + ".yml"),
            fs::path("..") / ".." / "intrinsic" / ("intrinsic_camera_" + std::to_string(cam1_id) + ".yml"),
            fs::path("..") / ".." / "intrinsic" / "cpp" / ("intrinsic_camera_" + std::to_string(cam1_id) + ".yml"),
        });
    }
    if (intr2Path.empty() && cam2_id >= 0) {
        intr2Path = findFirstExisting({
            fs::path("intrinsic_camera_" + std::to_string(cam2_id) + ".yml"),
            fs::path("..") / ("intrinsic_camera_" + std::to_string(cam2_id) + ".yml"),
            fs::path("..") / ".." / "intrinsic" / ("intrinsic_camera_" + std::to_string(cam2_id) + ".yml"),
            fs::path("..") / ".." / "intrinsic" / "cpp" / ("intrinsic_camera_" + std::to_string(cam2_id) + ".yml"),
        });
    }

    if (intr1Path.empty() || !loadIntrinsicsFromFile(intr1Path, K1, D1)) {
        std::cerr << "ERROR: Could not load intrinsics for cam1.\n";
        std::cerr << "  Provide --intr1 <path> OR use numeric cam id with --intrdir <dir>.\n";
        std::cerr << "  Expected keys: camera_matrix, distortion_coefficients\n";
        std::cerr << "  Tried: " << (intr1Path.empty() ? std::string("(none)") : intr1Path.string()) << "\n";
        return 1;
    }
    if (intr2Path.empty() || !loadIntrinsicsFromFile(intr2Path, K2, D2)) {
        std::cerr << "ERROR: Could not load intrinsics for cam2.\n";
        std::cerr << "  Provide --intr2 <path> OR use numeric cam id with --intrdir <dir>.\n";
        std::cerr << "  Expected keys: camera_matrix, distortion_coefficients\n";
        std::cerr << "  Tried: " << (intr2Path.empty() ? std::string("(none)") : intr2Path.string()) << "\n";
        return 1;
    }

    std::cout << "Loaded intrinsics:\n";
    std::cout << "  Cam1 intrinsics: " << intr1Path.string() << "\n";
    std::cout << "  Cam2 intrinsics: " << intr2Path.string() << "\n";

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints1, imagePoints2;
    int captured = 0;

    cv::Mat frame1, frame2, gray1, gray2;
    cv::Size imageSize;

    while (true) {
        cap1 >> frame1;
        cap2 >> frame2;

        if (frame1.empty() || frame2.empty()) break;
        if (imageSize.empty()) imageSize = frame1.size();

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

        cv::imshow("Camera 1", frame1);
        cv::imshow("Camera 2", frame2);

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
    double stereoRms = 0.0;

    const bool dLooksFisheye = (D1.total() == 4 && D2.total() == 4);
    const bool useFisheye = forceFisheye || (!forcePinhole && dLooksFisheye);

    std::cout << "Stereo model: " << (useFisheye ? "fisheye" : "pinhole") << "\n";
    if (useFisheye && !dLooksFisheye) {
        std::cerr << "ERROR: --fisheye requested but distortion does not look fisheye (expected 4 params).\n";
        return 1;
    }

    if (imageSize.empty()) {
        std::cerr << "ERROR: Unknown image size (no frames captured?).\n";
        return 1;
    }

    if (useFisheye) {
        // fisheye::stereoCalibrate uses different flags and does not directly output E/F.
        std::vector<std::vector<cv::Point3d>> objPts64;
        objPts64.reserve(objectPoints.size());
        for (const auto& op : objectPoints) {
            std::vector<cv::Point3d> tmp;
            tmp.reserve(op.size());
            for (const auto& p : op) tmp.emplace_back(p.x, p.y, p.z);
            objPts64.emplace_back(std::move(tmp));
        }

        const int flags = cv::fisheye::CALIB_FIX_INTRINSIC;
        const cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 200, 1e-7);
        stereoRms = cv::fisheye::stereoCalibrate(
            objPts64, imagePoints1, imagePoints2,
            K1, D1, K2, D2,
            imageSize, R, T,
            flags, criteria
        );
    } else {
        cv::Mat R1, R2, P1, P2, Q;
        const int flags = cv::CALIB_FIX_INTRINSIC;
        const cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 200, 1e-6);

        stereoRms = cv::stereoCalibrate(
            objectPoints,
            imagePoints1,
            imagePoints2,
            K1, D1,
            K2, D2,
            imageSize,
            R, T, E, F,
            flags,
            criteria
        );
    }

    // If we used fisheye, compute E/F from R/T and intrinsics for saving/inspection.
    if (E.empty()) {
        cv::Mat tx = (cv::Mat_<double>(3, 3) <<
            0, -T.at<double>(2), T.at<double>(1),
            T.at<double>(2), 0, -T.at<double>(0),
            -T.at<double>(1), T.at<double>(0), 0
        );
        E = tx * R;
    }
    if (F.empty()) {
        cv::Mat K1inv = K1.inv();
        cv::Mat K2invT = K2.inv().t();
        F = K2invT * E * K1inv;
    }

    std::cout << "Stereo RMS reprojection error: " << stereoRms << " pixels\n";
    std::cout << "Essential Matrix:\n" << E << "\n";
    std::cout << "Fundamental Matrix:\n" << F << "\n";

    std::cout << "\nRotation (Camera 2 relative to Camera 1):\n" << R << "\n";
    std::cout << "\nTranslation (Camera 2 relative to Camera 1):\n" << T << "\n";

    // Save extrinsic parameters
    std::string outfile;
    if (cam1_id >= 0 && cam2_id >= 0) {
        outfile = "extrinsic_cam" + std::to_string(cam1_id) + "_cam" + std::to_string(cam2_id) + ".yml";
    } else {
        outfile = "extrinsic_by_name.yml";
    }

    cv::FileStorage out(outfile, cv::FileStorage::WRITE);
    out << "rotation" << R;
    out << "translation" << T;
    out << "essential" << E;
    out << "fundamental" << F;
    out << "stereo_rms" << stereoRms;
    out << "intrinsics_cam1" << intr1Path.string();
    out << "intrinsics_cam2" << intr2Path.string();
    out << "calibration_pairs" << captured;
    out.release();

    std::cout << "\nSaved to: " << outfile << "\n";

    return 0;
}
