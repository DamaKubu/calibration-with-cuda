#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct BoardSpec {
    int cols = 8;
    int rows = 5;
    double squareMm = 65.0;
};

enum class Quality { Fast, Best };
enum class Model { Standard, Fisheye };

auto nowMs() -> int64_t {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

auto splitCsvInts(const std::string& s) -> std::vector<int> {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        out.push_back(std::stoi(item));
    }
    return out;
}

auto parsePattern(const std::string& pattern) -> BoardSpec {
    BoardSpec spec;
    std::string p = pattern;
    p.erase(std::remove_if(p.begin(), p.end(), ::isspace), p.end());
    auto xPos = p.find('x');
    if (xPos == std::string::npos) {
        xPos = p.find('X');
    }
    if (xPos == std::string::npos) {
        throw std::runtime_error("Invalid --pattern; expected like 8x5");
    }
    spec.cols = std::stoi(p.substr(0, xPos));
    spec.rows = std::stoi(p.substr(xPos + 1));
    return spec;
}

auto parseQuality(const std::string& s) -> Quality {
    if (s == "fast") {
        return Quality::Fast;
    }
    if (s == "best") {
        return Quality::Best;
    }
    throw std::runtime_error("Invalid --quality; use fast|best");
}

auto parseModel(const std::string& s) -> Model {
    if (s == "standard") {
        return Model::Standard;
    }
    if (s == "fisheye") {
        return Model::Fisheye;
    }
    throw std::runtime_error("Invalid --model; use standard|fisheye");
}

auto makeObjectPoints(const BoardSpec& board) -> std::vector<cv::Point3f> {
    std::vector<cv::Point3f> obj;
    obj.reserve(static_cast<size_t>(board.cols) * static_cast<size_t>(board.rows));
    for (int y = 0; y < board.rows; ++y) {
        for (int x = 0; x < board.cols; ++x) {
            obj.emplace_back(static_cast<float>(x * board.squareMm), static_cast<float>(y * board.squareMm), 0.0f);
        }
    }
    return obj;
}

auto findChessboardCornersAny(const cv::Mat& gray, const BoardSpec& board, Quality quality)
    -> std::optional<std::vector<cv::Point2f>> {
    const cv::Size patternSize(board.cols, board.rows);

    std::vector<cv::Point2f> corners;
    bool found = false;

#if CV_VERSION_MAJOR >= 4
    if (quality == Quality::Best) {
        int flags = cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY;
        found = cv::findChessboardCornersSB(gray, patternSize, corners, flags);
    } else {
        int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
        found = cv::findChessboardCorners(gray, patternSize, corners, flags);
    }
#else
    (void)quality;
    int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
    found = cv::findChessboardCorners(gray, patternSize, corners, flags);
#endif

    if (!found) {
        return std::nullopt;
    }

    const cv::Size win = (quality == Quality::Best) ? cv::Size(15, 15) : cv::Size(11, 11);
    const int iters = (quality == Quality::Best) ? 60 : 30;
    const cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, iters, 1e-8);
    cv::cornerSubPix(gray, corners, win, cv::Size(-1, -1), criteria);

    return corners;
}

auto meanCornerShiftPx(const std::vector<cv::Point2f>& prev, const std::vector<cv::Point2f>& cur) -> double {
    if (prev.size() != cur.size() || prev.empty()) {
        return 1e9;
    }

    double sum = 0.0;
    for (size_t i = 0; i < prev.size(); ++i) {
        const auto d = prev[i] - cur[i];
        sum += std::sqrt(d.dot(d));
    }
    return sum / static_cast<double>(prev.size());
}

struct CaptureState {
    int camIndex = 0;
    cv::VideoCapture cap;
    int captured = 0;
    int64_t lastSaveMs = 0;
    std::optional<std::vector<cv::Point2f>> prevCorners;
    cv::Mat lastFrame;
    std::optional<std::vector<cv::Point2f>> lastCorners;
};

auto ensureDir(const fs::path& p) -> void {
    std::error_code ec;
    fs::create_directories(p, ec);
    if (ec) {
        throw std::runtime_error("Failed to create directory: " + p.string());
    }
}

auto saveFrame(const fs::path& outPath, const cv::Mat& frame) -> bool {
    ensureDir(outPath.parent_path());
    return cv::imwrite(outPath.string(), frame);
}

struct CalibrationResult {
    bool ok = false;
    int camIndex = 0;
    cv::Size imageSize;
    double rms = 0.0;
    cv::Mat K;
    cv::Mat D;
    int imagesUsed = 0;
    std::string error;
};

auto listImagesSorted(const fs::path& dir) -> std::vector<fs::path> {
    std::vector<fs::path> out;
    if (!fs::exists(dir)) {
        return out;
    }

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto ext = entry.path().extension().string();
        std::string lower = ext;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower == ".jpg" || lower == ".jpeg" || lower == ".png" || lower == ".bmp") {
            out.push_back(entry.path());
        }
    }

    std::sort(out.begin(), out.end());
    return out;
}

auto calibrateFromImages(int camIndex,
                         const fs::path& imagesDir,
                         const BoardSpec& board,
                         Quality quality,
                         Model model)
    -> CalibrationResult {
    CalibrationResult res;
    res.camIndex = camIndex;

    const auto images = listImagesSorted(imagesDir);
    if (images.empty()) {
        res.error = "No images found in " + imagesDir.string();
        return res;
    }

    const auto objTemplate = makeObjectPoints(board);

    std::vector<std::vector<cv::Point3f>> objpoints;
    std::vector<std::vector<cv::Point2f>> imgpoints;

    cv::Mat lastGray;

    for (const auto& path : images) {
        const cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
        if (img.empty()) {
            continue;
        }

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        lastGray = gray;

        const auto cornersOpt = findChessboardCornersAny(gray, board, quality);
        if (!cornersOpt.has_value()) {
            continue;
        }

        objpoints.push_back(objTemplate);
        imgpoints.push_back(*cornersOpt);
    }

    res.imagesUsed = static_cast<int>(objpoints.size());
    if (res.imagesUsed < 5) {
        res.error = "Not enough images with detected corners (need >=5), got " + std::to_string(res.imagesUsed);
        return res;
    }

    res.imageSize = lastGray.size();

    if (model == Model::Fisheye) {
        res.K = cv::Mat::zeros(3, 3, CV_64F);
        res.D = cv::Mat::zeros(4, 1, CV_64F);

        std::vector<cv::Mat> rvecs, tvecs;
        int flags = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC | cv::fisheye::CALIB_CHECK_COND |
            cv::fisheye::CALIB_FIX_SKEW;
        const int iters = (quality == Quality::Best) ? 200 : 100;
        const cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, iters, 1e-10);

        try {
            res.rms = cv::fisheye::calibrate(objpoints, imgpoints, res.imageSize, res.K, res.D, rvecs, tvecs,
                                             flags, criteria);
            res.ok = true;
        } catch (const cv::Exception& e) {
            res.error = e.what();
        }

        return res;
    }

    res.K = cv::Mat::eye(3, 3, CV_64F);
    res.D = cv::Mat::zeros((quality == Quality::Best) ? 8 : 5, 1, CV_64F);

    std::vector<cv::Mat> rvecs, tvecs;
    int flags = (quality == Quality::Best) ? cv::CALIB_RATIONAL_MODEL : 0;
    const int iters = (quality == Quality::Best) ? 200 : 100;
    const cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, iters, 1e-10);

    try {
        res.rms = cv::calibrateCamera(objpoints, imgpoints, res.imageSize, res.K, res.D, rvecs, tvecs, flags,
                                      criteria);
        res.ok = true;
    } catch (const cv::Exception& e) {
        res.error = e.what();
    }

    return res;
}

auto writePerCameraJson(const fs::path& outPath,
                        const BoardSpec& board,
                        Quality quality,
                        Model model,
                        const CalibrationResult& r)
    -> void {
    ensureDir(outPath.parent_path());

    cv::FileStorage fs(outPath.string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

    fs << "camera_index" << r.camIndex;
    fs << "calibration_success" << r.ok;
    fs << "pattern_cols" << board.cols;
    fs << "pattern_rows" << board.rows;
    fs << "square_mm" << board.squareMm;
    fs << "quality" << ((quality == Quality::Best) ? "best" : "fast");
    fs << "model" << ((model == Model::Fisheye) ? "fisheye" : "standard");

    fs << "images_used" << r.imagesUsed;
    fs << "image_size" << "[" << r.imageSize.width << r.imageSize.height << "]";
    fs << "rms_reprojection_error" << r.rms;

    if (r.ok) {
        fs << "K" << r.K;
        fs << "D" << r.D;
    } else {
        fs << "error" << r.error;
    }
}

auto writeAggregateJson(const fs::path& outPath,
                        const BoardSpec& board,
                        Quality quality,
                        Model model,
                        const std::vector<CalibrationResult>& results)
    -> void {
    ensureDir(outPath.parent_path());

    cv::FileStorage fs(outPath.string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

    fs << "pattern" << "{";
    fs << "cols" << board.cols;
    fs << "rows" << board.rows;
    fs << "square_mm" << board.squareMm;
    fs << "}";

    fs << "quality" << ((quality == Quality::Best) ? "best" : "fast");
    fs << "model" << ((model == Model::Fisheye) ? "fisheye" : "standard");

    fs << "cameras" << "[";
    for (const auto& r : results) {
        fs << "{";
        fs << "camera_index" << r.camIndex;
        fs << "calibration_success" << r.ok;
        fs << "images_used" << r.imagesUsed;
        fs << "image_size" << "[" << r.imageSize.width << r.imageSize.height << "]";
        fs << "rms_reprojection_error" << r.rms;
        if (r.ok) {
            fs << "K" << r.K;
            fs << "D" << r.D;
        } else {
            fs << "error" << r.error;
        }
        fs << "}";
    }
    fs << "]";
}

struct Args {
    std::string mode = "capture"; // capture|calibrate|all
    std::vector<int> cams = {0};
    fs::path outDir = "calibration_images";
    BoardSpec board;
    int count = 30;
    bool autoCapture = true;
    int intervalMs = 800;
    Quality quality = Quality::Best;
    Model model = Model::Standard;
    int width = 0;
    int height = 0;
};

auto printUsage() -> void {
    std::cout
        << "multi_cam_intrinsic_calibrator\n\n"
        << "Modes:\n"
        << "  --mode capture     Capture images (default)\n"
        << "  --mode calibrate   Calibrate from saved images\n"
        << "  --mode all         Capture then calibrate\n\n"
        << "Common options:\n"
        << "  --cams 0,1,2       Camera indices (default: 0)\n"
        << "  --out DIR          Output directory (default: calibration_images)\n"
        << "  --pattern 8x5      Chessboard inner corners (default: 8x5)\n"
        << "  --square-mm 65     Square size in mm (default: 65)\n"
        << "  --quality fast|best (default: best)\n"
        << "  --model standard|fisheye (default: standard)\n\n"
        << "Capture options:\n"
        << "  --count N          Images per camera (default: 30)\n"
        << "  --auto 0|1         Auto capture (default: 1)\n"
        << "  --interval-ms 800  Auto capture interval (default: 800)\n"
        << "  --width W --height H  Force capture resolution (optional)\n\n"
        << "Controls (capture window):\n"
        << "  [Space] save frame(s) if chessboard found\n"
        << "  [A] toggle auto\n"
        << "  [Esc] quit\n";
}

auto parseArgs(int argc, char** argv) -> Args {
    Args a;

    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto requireValue = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        if (k == "--help" || k == "-h") {
            printUsage();
            std::exit(0);
        } else if (k == "--mode") {
            a.mode = requireValue(k);
        } else if (k == "--cams") {
            a.cams = splitCsvInts(requireValue(k));
            if (a.cams.empty()) {
                throw std::runtime_error("--cams cannot be empty");
            }
        } else if (k == "--out") {
            a.outDir = requireValue(k);
        } else if (k == "--pattern") {
            a.board = parsePattern(requireValue(k));
        } else if (k == "--square-mm") {
            a.board.squareMm = std::stod(requireValue(k));
        } else if (k == "--quality") {
            a.quality = parseQuality(requireValue(k));
        } else if (k == "--model") {
            a.model = parseModel(requireValue(k));
        } else if (k == "--count") {
            a.count = std::stoi(requireValue(k));
        } else if (k == "--auto") {
            a.autoCapture = (std::stoi(requireValue(k)) != 0);
        } else if (k == "--interval-ms") {
            a.intervalMs = std::stoi(requireValue(k));
        } else if (k == "--width") {
            a.width = std::stoi(requireValue(k));
        } else if (k == "--height") {
            a.height = std::stoi(requireValue(k));
        } else {
            throw std::runtime_error("Unknown argument: " + k);
        }
    }

    return a;
}

auto runCapture(const Args& args) -> int {
    std::vector<CaptureState> states;
    states.reserve(args.cams.size());

    for (int cam : args.cams) {
        CaptureState s;
        s.camIndex = cam;
        s.cap.open(cam);
        if (!s.cap.isOpened()) {
            std::cerr << "Cannot open camera " << cam << "\n";
            return 2;
        }
        if (args.width > 0) {
            s.cap.set(cv::CAP_PROP_FRAME_WIDTH, args.width);
        }
        if (args.height > 0) {
            s.cap.set(cv::CAP_PROP_FRAME_HEIGHT, args.height);
        }
        states.push_back(std::move(s));
    }

    std::cout << "Controls: [Space]=save  [A]=toggle auto  [Esc]=quit\n";
    std::cout << "Auto-capture: " << (args.autoCapture ? "true" : "false")
              << " | Quality: " << ((args.quality == Quality::Best) ? "best" : "fast")
              << " | Pattern: " << args.board.cols << "x" << args.board.rows << "\n";

    bool autoEnabled = args.autoCapture;

    auto trySave = [&](CaptureState& s, const char* reason, double minShiftPx, int64_t minIntervalMs) {
        if (!s.lastCorners.has_value() || s.lastFrame.empty() || s.captured >= args.count) {
            return;
        }

        const int64_t t = nowMs();
        if (minIntervalMs > 0 && (t - s.lastSaveMs) < minIntervalMs) {
            return;
        }

        if (s.prevCorners.has_value()) {
            const double shift = meanCornerShiftPx(*s.prevCorners, *s.lastCorners);
            if (shift < minShiftPx) {
                return;
            }
        }

        const fs::path camDir = args.outDir / ("cam" + std::to_string(s.camIndex));
        const fs::path outPath = camDir / ("calib_" + std::to_string(s.captured) + ".jpg");
        if (saveFrame(outPath, s.lastFrame)) {
            s.prevCorners = *s.lastCorners;
            s.lastSaveMs = t;
            s.captured += 1;
            std::cout << reason << " saved " << outPath.string() << "\n";
        }
    };

    while (true) {
        bool allDone = true;

        for (auto& s : states) {
            if (s.captured < args.count) {
                allDone = false;
            }

            cv::Mat frame;
            if (!s.cap.read(frame) || frame.empty()) {
                continue;
            }

            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            const auto cornersOpt = findChessboardCornersAny(gray, args.board, args.quality);
            const bool found = cornersOpt.has_value();
            if (found) {
                cv::drawChessboardCorners(frame, cv::Size(args.board.cols, args.board.rows), *cornersOpt, true);
            }

            // Keep latest per-camera frame/corners for manual capture.
            s.lastFrame = frame.clone();
            s.lastCorners = cornersOpt;

            // Overlay
            std::ostringstream oss;
            oss << "cam" << s.camIndex << "  " << s.captured << "/" << args.count << "  auto="
                << (autoEnabled ? "1" : "0");
            cv::putText(frame, oss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(0, 255, 0), 2);

            const std::string win = "Capture - cam" + std::to_string(s.camIndex);
            cv::imshow(win, frame);

            // Auto-save
            if (autoEnabled && found) {
                trySave(s, "Auto", 14.0, args.intervalMs);
            }
        }

        if (allDone) {
            break;
        }

        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27) { // ESC
            break;
        }
        if (key == 'a' || key == 'A') {
            autoEnabled = !autoEnabled;
            std::cout << "Auto-capture: " << (autoEnabled ? "true" : "false") << "\n";
        }
        if (key == 32) { // Space
            // Manual capture from all cameras using last frame/corners
            for (auto& s : states) {
                trySave(s, "Manual", 12.0, 0);
            }
        }
    }

    for (auto& s : states) {
        s.cap.release();
    }
    cv::destroyAllWindows();

    std::cout << "Done. Images saved under " << args.outDir.string() << "\\camX\\" << "\n";
    return 0;
}

auto runCalibrate(const Args& args) -> int {
    std::vector<CalibrationResult> results;
    results.reserve(args.cams.size());

    for (int cam : args.cams) {
        const fs::path imagesDir = args.outDir / ("cam" + std::to_string(cam));
        auto r = calibrateFromImages(cam, imagesDir, args.board, args.quality, args.model);
        results.push_back(r);

        const fs::path perCamJson = args.outDir / ("intrinsics_cam" + std::to_string(cam) + ".json");
        writePerCameraJson(perCamJson, args.board, args.quality, args.model, r);

        if (r.ok) {
            std::cout << "cam" << cam << ": OK  rms=" << r.rms << "  images=" << r.imagesUsed << "\n";
        } else {
            std::cout << "cam" << cam << ": FAIL  " << r.error << "\n";
        }
    }

    const fs::path aggJson = args.outDir / "intrinsics_all_cameras.json";
    writeAggregateJson(aggJson, args.board, args.quality, args.model, results);
    std::cout << "Wrote " << aggJson.string() << " and per-camera intrinsics_camX.json\n";

    return 0;
}

int main(int argc, char** argv) {
    try {
        const auto args = parseArgs(argc, argv);

        if (args.mode == "capture") {
            return runCapture(args);
        }
        if (args.mode == "calibrate") {
            return runCalibrate(args);
        }
        if (args.mode == "all") {
            const int rc = runCapture(args);
            if (rc != 0) {
                return rc;
            }
            return runCalibrate(args);
        }

        std::cerr << "Unknown --mode: " << args.mode << "\n";
        printUsage();
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
