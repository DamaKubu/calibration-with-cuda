#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void printUsage() {
    std::cout
        << "Capture calibration images (lossless PNG, full-res)\n"
        << "Usage:\n"
        << "  capture_calibration_images_fisheye.exe --camera 0 --out calib_cam0 --count 60 --pattern 8x5\n"
        << "\nOptions:\n"
        << "  --camera <int>            Camera index (default 0)\n"
        << "  --out <dir>               Output directory (default calibration_images)\n"
        << "  --count <int>             Number of images to save (default 60)\n"
        << "  --pattern <WxH>           Inner corners, e.g. 8x5 (default 8x5)\n"
        << "  --try-max-res <0|1>       Try common max resolutions (default 1)\n"
        << "  --min-blur <float>        Variance-of-Laplacian threshold (default 140)\n"
        << "  --min-shift-px <float>    Minimum mean corner shift between saves (default 14)\n"
        << "  --auto <0|1>              Auto-capture enabled (default 1)\n"
        << "  --interval-ms <int>       Auto-capture interval in ms (default 700)\n"
        << "  --png-compression <0-9>   PNG compression (lossless). 0 fastest (default 0)\n"
        << "  --rotate <0|90|180|270>   Rotate frames before detect/save (default 0)\n"
        << "  --flip <none|x|y|xy>      Flip frames before detect/save (default none)\n"
        << "  --backend <dshow|msmf|any> Capture backend hint (default any)\n"
        << "  --mjpg <0|1>              Try MJPG stream (some cams crop/zoom). (default 1)\n"
        << "  --size <WxH>              Force capture size, e.g. 1280x960 (default unset)\n"
        << "  --zoom <double>           Force camera zoom if supported (default 0)\n"
        << "\nRuntime controls:\n"
        << "  [Space]=save  [A]=toggle auto  [R]=rotate  [F]=flip  [Esc]=quit\n";
}

static bool parsePattern(const std::string& s, cv::Size& pattern) {
    auto x = s.find('x');
    if (x == std::string::npos) return false;
    try {
        int w = std::stoi(s.substr(0, x));
        int h = std::stoi(s.substr(x + 1));
        if (w <= 0 || h <= 0) return false;
        pattern = cv::Size(w, h);
        return true;
    } catch (...) {
        return false;
    }
}

static double blurScoreVarianceOfLaplacian(const cv::Mat& gray) {
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0];
}

static bool cornersChangedEnough(
    const std::optional<std::vector<cv::Point2f>>& prev,
    const std::vector<cv::Point2f>& now,
    double minShiftPx) {
    if (!prev.has_value()) return true;
    if (prev->size() != now.size() || now.empty()) return true;

    double sum = 0.0;
    for (size_t i = 0; i < now.size(); ++i) {
        const cv::Point2f d = (*prev)[i] - now[i];
        sum += std::sqrt(d.x * d.x + d.y * d.y);
    }
    const double mean = sum / static_cast<double>(now.size());
    return mean >= minShiftPx;
}

static cv::Size trySetMaxResolution(cv::VideoCapture& cap) {
    const std::vector<cv::Size> candidates = {
        {3840, 2160},
        {2560, 1440},
        {1920, 1080},
        {1600, 1200},
        {1280, 720},
        {1024, 768},
        {800, 600},
        {640, 480},
    };

    int bestW = 0;
    int bestH = 0;

    for (const auto& sz : candidates) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(sz.width));
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(sz.height));

        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) continue;
        const int gotW = frame.cols;
        const int gotH = frame.rows;

        if (gotW >= bestW && gotH >= bestH) {
            bestW = gotW;
            bestH = gotH;
        }

        if (gotW == sz.width && gotH == sz.height) {
            return sz;
        }
    }

    if (bestW > 0 && bestH > 0) return {bestW, bestH};

    int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    return {w, h};
}

static int backendFromString(const std::string& s) {
    if (s == "dshow") return cv::CAP_DSHOW;
    if (s == "msmf") return cv::CAP_MSMF;
    return cv::CAP_ANY;
}

static int flipCodeFromString(const std::string& s) {
    if (s == "none") return 999; // sentinel
    if (s == "x") return 0;      // vertical flip
    if (s == "y") return 1;      // horizontal flip
    if (s == "xy") return -1;    // both
    return 999;
}

static std::string nextFlipMode(const std::string& current) {
    if (current == "none") return "x";
    if (current == "x") return "y";
    if (current == "y") return "xy";
    return "none";
}

static void applyRotateFlip(cv::Mat& img, int rotateDeg, int flipCode) {
    if (rotateDeg == 90) {
        cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);
    } else if (rotateDeg == 180) {
        cv::rotate(img, img, cv::ROTATE_180);
    } else if (rotateDeg == 270) {
        cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
    }

    if (flipCode != 999) {
        cv::flip(img, img, flipCode);
    }
}

int main(int argc, char** argv) {
    int cameraId = 0;
    std::string outDir = "calibration_images";
    int targetCount = 60;
    cv::Size pattern(8, 5);
    bool tryMaxRes = true;
    double minBlur = 140.0;
    double minShiftPx = 14.0;
    bool autoEnabled = true;
    int intervalMs = 700;
    int pngCompression = 0;
    std::string backend = "any";
    int rotateDeg = 0;
    std::string flipMode = "none";
    bool enableMJPG = true;
    cv::Size forceSize(0, 0);
    double zoomValue = 0.0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };

        if (a == "--help" || a == "-h") {
            printUsage();
            return 0;
        } else if (a == "--camera") {
            cameraId = std::stoi(need("--camera"));
        } else if (a == "--out") {
            outDir = need("--out");
        } else if (a == "--count") {
            targetCount = std::stoi(need("--count"));
        } else if (a == "--pattern") {
            cv::Size p;
            if (!parsePattern(need("--pattern"), p)) {
                std::cerr << "Bad pattern; expected WxH e.g. 8x5\n";
                return 2;
            }
            pattern = p;
        } else if (a == "--try-max-res") {
            tryMaxRes = (std::stoi(need("--try-max-res")) != 0);
        } else if (a == "--min-blur") {
            minBlur = std::stod(need("--min-blur"));
        } else if (a == "--min-shift-px") {
            minShiftPx = std::stod(need("--min-shift-px"));
        } else if (a == "--auto") {
            autoEnabled = (std::stoi(need("--auto")) != 0);
        } else if (a == "--interval-ms") {
            intervalMs = std::stoi(need("--interval-ms"));
        } else if (a == "--png-compression") {
            pngCompression = std::stoi(need("--png-compression"));
            pngCompression = std::clamp(pngCompression, 0, 9);
        } else if (a == "--rotate") {
            rotateDeg = std::stoi(need("--rotate"));
            if (!(rotateDeg == 0 || rotateDeg == 90 || rotateDeg == 180 || rotateDeg == 270)) {
                std::cerr << "--rotate must be one of 0,90,180,270\n";
                return 2;
            }
        } else if (a == "--flip") {
            flipMode = need("--flip");
            std::transform(flipMode.begin(), flipMode.end(), flipMode.begin(), ::tolower);
            if (flipCodeFromString(flipMode) == 999 && flipMode != "none") {
                std::cerr << "--flip must be one of none,x,y,xy\n";
                return 2;
            }
        } else if (a == "--backend") {
            backend = need("--backend");
            std::transform(backend.begin(), backend.end(), backend.begin(), ::tolower);
        } else if (a == "--mjpg") {
            enableMJPG = (std::stoi(need("--mjpg")) != 0);
        } else if (a == "--size") {
            cv::Size s;
            if (!parsePattern(need("--size"), s)) {
                std::cerr << "Bad --size; expected WxH e.g. 1280x960\n";
                return 2;
            }
            forceSize = s;
        } else if (a == "--zoom") {
            zoomValue = std::stod(need("--zoom"));
        } else {
            std::cerr << "Unknown arg: " << a << "\n";
            printUsage();
            return 2;
        }
    }

    fs::create_directories(outDir);
    const fs::path outAbs = fs::absolute(fs::path(outDir));

    const int capBackend = backendFromString(backend);
    cv::VideoCapture cap;
    if (capBackend == cv::CAP_ANY) {
        cap.open(cameraId);
    } else {
        cap.open(cameraId, capBackend);
    }

    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open camera " << cameraId << "\n";
        return 1;
    }

    // Some webcams appear "zoomed" when the driver crops due to aspect mismatch or MJPG mode.
    // Let the user disable MJPG and/or force a size.
    if (enableMJPG) {
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    }

    // Reset/force zoom if supported by the backend/driver.
    cap.set(cv::CAP_PROP_ZOOM, zoomValue);

    cv::Size actualSize;
    if (forceSize.width > 0 && forceSize.height > 0) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(forceSize.width));
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(forceSize.height));

        cv::Mat probe;
        if (cap.read(probe) && !probe.empty()) {
            actualSize = {probe.cols, probe.rows};
        } else {
            actualSize = {
                static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)),
            };
        }
    } else if (tryMaxRes) {
        actualSize = trySetMaxResolution(cap);
    } else {
        actualSize = {
            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)),
        };
    }

    std::cout << "Capture calibration images (C++)\n";
    std::cout << "Camera: " << cameraId << " | Pattern: " << pattern.width << "x" << pattern.height << "\n";
    std::cout << "Resolution: " << actualSize.width << "x" << actualSize.height << "\n";
    if (forceSize.width > 0 && forceSize.height > 0) {
        std::cout << "Requested size: " << forceSize.width << "x" << forceSize.height << "\n";
    }
    std::cout << "Output: " << outAbs.string() << " | Target: " << targetCount << " PNGs\n";
    std::cout << "Auto: " << (autoEnabled ? "ON" : "OFF") << " | Interval: " << intervalMs << "ms\n";
    std::cout << "min_blur: " << minBlur << " | min_shift_px: " << minShiftPx << "\n";
    std::cout << "rotate: " << rotateDeg << " | flip: " << flipMode << "\n";
    std::cout << "mjpg: " << (enableMJPG ? "ON" : "OFF") << " | zoom: " << cap.get(cv::CAP_PROP_ZOOM) << "\n";
    std::cout << "Controls: [Space]=save  [A]=toggle auto  [R]=rotate  [F]=flip  [Esc]=quit\n\n";

    const std::string winName = "Capture Calibration (C++)";
    cv::namedWindow(winName, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    // Start with a manageable window size; user can resize freely.
    {
        const int initW = std::min(actualSize.width, 1280);
        const int initH = std::min(actualSize.height, 720);
        if (initW > 0 && initH > 0) {
            cv::resizeWindow(winName, initW, initH);
        }
    }

    int saved = 0;
    std::optional<std::vector<cv::Point2f>> prevCorners;
    auto lastSave = std::chrono::steady_clock::now() - std::chrono::milliseconds(intervalMs);

    while (saved < targetCount) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            continue;
        }

        // Apply orientation fix BEFORE detection + saving.
        if (rotateDeg != 0 || flipMode != "none") {
            applyRotateFlip(frame, rotateDeg, flipCodeFromString(flipMode));
        }

        cv::Mat frameRaw = frame.clone();
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        const double blur = blurScoreVarianceOfLaplacian(gray);

        std::vector<cv::Point2f> corners;
        bool found = false;
        if (cv::checkRange(gray)) {
            // Prefer SB for best-quality if available
#if (CV_VERSION_MAJOR >= 4)
            found = cv::findChessboardCornersSB(
                gray,
                pattern,
                corners,
                cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY);
#else
            found = cv::findChessboardCorners(
                gray,
                pattern,
                corners,
                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
#endif
        }

        if (found) {
            cv::cornerSubPix(
                gray,
                corners,
                cv::Size(15, 15),
                cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 80, 1e-12));
            cv::drawChessboardCorners(frame, pattern, corners, true);
        }

        const std::string status =
            std::to_string(saved) + "/" + std::to_string(targetCount) +
            "  auto=" + (autoEnabled ? std::string("1") : std::string("0")) +
            "  blur=" + std::to_string(static_cast<int>(blur)) +
            "  " + std::to_string(frame.cols) + "x" + std::to_string(frame.rows) +
            "  rot=" + std::to_string(rotateDeg) +
            "  flip=" + flipMode;

        cv::putText(frame, status, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 0}, 2);
        if (!found) {
            cv::putText(frame, "Searching chessboard...", {10, 70}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 0, 255}, 2);
        } else {
            cv::putText(frame, "FOUND", {10, 70}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);
        }

        cv::imshow(winName, frame);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 27) break;
        if (key == 'a' || key == 'A') {
            autoEnabled = !autoEnabled;
            std::cout << "Auto-capture: " << (autoEnabled ? "ON" : "OFF") << "\n";
        }
        if (key == 'r' || key == 'R') {
            rotateDeg = (rotateDeg + 90) % 360;
            std::cout << "Rotate: " << rotateDeg << "\n";
        }
        if (key == 'f' || key == 'F') {
            flipMode = nextFlipMode(flipMode);
            std::cout << "Flip: " << flipMode << "\n";
        }

        auto now = std::chrono::steady_clock::now();
        const bool manualSave = (key == 32);
        const bool autoSave = autoEnabled && (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSave).count() >= intervalMs);

        if ((manualSave || autoSave) && found) {
            if (blur < minBlur) {
                if (manualSave) {
                    std::cout << "Skip: blur " << blur << " < " << minBlur << "\n";
                }
                continue;
            }
            if (!cornersChangedEnough(prevCorners, corners, minShiftPx)) {
                if (manualSave) {
                    std::cout << "Skip: pose too similar (min_shift_px=" << minShiftPx << ")\n";
                }
                continue;
            }

            const fs::path outPath = fs::path(outDir) / ("calib_" + cv::format("%04d", saved) + ".png");
            const std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, pngCompression};
            if (!cv::imwrite(outPath.string(), frameRaw, params)) {
                std::cerr << "ERROR: Failed to write " << outPath.string() << "\n";
                continue;
            }

            prevCorners = corners;
            lastSave = now;
            ++saved;
            std::cout << (autoSave ? "Auto-saved: " : "Saved: ") << outPath.string() << "\n";
        }
    }

    cap.release();
    cv::destroyAllWindows();

    std::cout << "Done. Saved " << saved << " images to " << outDir << "\n";
    return 0;
}
