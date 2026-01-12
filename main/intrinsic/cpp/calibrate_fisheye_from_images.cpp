#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void printUsage() {
    std::cout
    << "Offline intrinsic calibration with outlier rejection (C++)\n"
        << "Usage:\n"
    << "  calibrate_fisheye_from_images.exe --images-dir calib_cam0 --pattern 8x5 --square-mm 65 \\\n"
    << "      --model standard --output intrinsics_cam0.txt\n"
        << "\nOptions:\n"
        << "  --images-dir <dir>            Directory of PNG images\n"
        << "  --ext <png|jpg>               Extension filter (default png)\n"
        << "  --pattern <WxH>               Inner corners, e.g. 8x5 (default 8x5)\n"
        << "  --square-mm <float>           Square size in mm (default 65)\n"
    << "  --model <standard|fisheye>    Lens model (default standard)\n"
        << "  --max-per-image-error <float> Reject images above this reproj error (px) (default 1.2)\n"
        << "  --max-remove <int>            Max outliers to remove (default 15)\n"
        << "  --output <file.txt>           Output text file (default intrinsics.txt)\n";
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

static std::vector<fs::path> listImages(const fs::path& dir, const std::string& extLower) {
    std::vector<fs::path> files;
    if (!fs::exists(dir) || !fs::is_directory(dir)) return files;

    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        auto p = e.path();
        std::string ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ("." + extLower)) files.push_back(p);
    }

    std::sort(files.begin(), files.end());
    return files;
}

static double perImageReprojError(
    const std::vector<cv::Point3f>& obj,
    const std::vector<cv::Point2f>& img,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    const cv::Mat& K,
    const cv::Mat& D,
    bool fisheye) {
    std::vector<cv::Point2f> proj;
    if (fisheye) {
        cv::fisheye::projectPoints(obj, proj, rvec, tvec, K, D);
    } else {
        cv::projectPoints(obj, rvec, tvec, K, D, proj);
    }
    if (proj.size() != img.size() || img.empty()) return 1e9;

    double sum = 0.0;
    for (size_t i = 0; i < img.size(); ++i) {
        cv::Point2f d = proj[i] - img[i];
        sum += d.x * d.x + d.y * d.y;
    }
    return std::sqrt(sum / static_cast<double>(img.size()));
}

static double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

int main(int argc, char** argv) {
    fs::path imagesDir;
    std::string extLower = "png";
    cv::Size pattern(8, 5);
    double squareMm = 65.0;
    std::string model = "standard";
    double maxPerImageError = 1.2;
    int maxRemove = 15;
    std::string output = "intrinsics.txt";

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
        } else if (a == "--images-dir") {
            imagesDir = need("--images-dir");
        } else if (a == "--ext") {
            extLower = need("--ext");
            std::transform(extLower.begin(), extLower.end(), extLower.begin(), ::tolower);
        } else if (a == "--pattern") {
            cv::Size p;
            if (!parsePattern(need("--pattern"), p)) {
                std::cerr << "Bad pattern; expected WxH e.g. 8x5\n";
                return 2;
            }
            pattern = p;
        } else if (a == "--square-mm") {
            squareMm = std::stod(need("--square-mm"));
        } else if (a == "--model") {
            model = need("--model");
            std::transform(model.begin(), model.end(), model.begin(), ::tolower);
            if (model != "standard" && model != "fisheye") {
                std::cerr << "--model must be 'standard' or 'fisheye'\n";
                return 2;
            }
        } else if (a == "--max-per-image-error") {
            maxPerImageError = std::stod(need("--max-per-image-error"));
        } else if (a == "--max-remove") {
            maxRemove = std::stoi(need("--max-remove"));
        } else if (a == "--output") {
            output = need("--output");
        } else {
            std::cerr << "Unknown arg: " << a << "\n";
            printUsage();
            return 2;
        }
    }

    std::cout << "Build: " << __DATE__ << " " << __TIME__ << "\n";

    if (imagesDir.empty()) {
        std::cerr << "--images-dir is required\n";
        printUsage();
        return 2;
    }

    const auto files = listImages(imagesDir, extLower);
    if (files.empty()) {
        std::cerr << "No images found in " << imagesDir.string() << " with extension " << extLower << "\n";
        return 1;
    }

    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<std::string> paths;
    cv::Size imageSize;

    std::cout << "Detecting chessboards in " << files.size() << " images...\n";

    for (const auto& p : files) {
        cv::Mat img = cv::imread(p.string(), cv::IMREAD_COLOR);
        if (img.empty()) continue;

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        if (imageSize.empty()) {
            imageSize = gray.size();
        } else if (gray.size() != imageSize) {
            std::cerr << "ERROR: Mixed image sizes. Found " << gray.cols << "x" << gray.rows
                      << " but expected " << imageSize.width << "x" << imageSize.height << "\n";
            return 1;
        }

        std::vector<cv::Point2f> corners;
        bool found = false;
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

        if (!found) continue;

        cv::cornerSubPix(
            gray,
            corners,
            cv::Size(15, 15),
            cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 80, 1e-12));

        imagePoints.push_back(corners);
        paths.push_back(p.string());
    }

    if (imagePoints.size() < 8) {
        std::cerr << "ERROR: Not enough valid detections: " << imagePoints.size() << " (need >= 8)\n";
        return 1;
    }

    // Build object points template
    std::vector<cv::Point3f> obj;
    obj.reserve(static_cast<size_t>(pattern.width * pattern.height));
    for (int y = 0; y < pattern.height; ++y) {
        for (int x = 0; x < pattern.width; ++x) {
            obj.emplace_back(static_cast<float>(x * squareMm), static_cast<float>(y * squareMm), 0.0f);
        }
    }

    std::vector<int> kept;
    kept.resize(static_cast<int>(imagePoints.size()));
    std::iota(kept.begin(), kept.end(), 0);

    struct Rejected {
        std::string path;
        double errorPx;
    };

    std::vector<Rejected> rejected;

    const bool useFisheye = (model == "fisheye");

    auto runCalib = [&](const std::vector<int>& idxs,
                        cv::Mat& K,
                        cv::Mat& D,
                        std::vector<cv::Mat>& rvecs,
                        std::vector<cv::Mat>& tvecs,
                        std::vector<double>& perErr) -> double {
        std::vector<std::vector<cv::Point3f>> objectPoints;
        std::vector<std::vector<cv::Point2f>> imgPts;
        objectPoints.reserve(idxs.size());
        imgPts.reserve(idxs.size());

        for (int i : idxs) {
            objectPoints.push_back(obj);
            imgPts.push_back(imagePoints[static_cast<size_t>(i)]);
        }

        if (useFisheye) {
            K = cv::Mat::zeros(3, 3, CV_64F);
            D = cv::Mat::zeros(4, 1, CV_64F);
        } else {
            K = cv::Mat::eye(3, 3, CV_64F);
            D = cv::Mat::zeros(8, 1, CV_64F); // rational model (k1..k6 + p1,p2)
        }
        rvecs.clear();
        tvecs.clear();

        const cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 500, 1e-12);

        double rms = 0.0;
        if (useFisheye) {
            const int flags = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC |
                              cv::fisheye::CALIB_CHECK_COND |
                              cv::fisheye::CALIB_FIX_SKEW;
            rms = cv::fisheye::calibrate(
                objectPoints,
                imgPts,
                imageSize,
                K,
                D,
                rvecs,
                tvecs,
                flags,
                criteria);
        } else {
            const int flags = cv::CALIB_RATIONAL_MODEL;
            rms = cv::calibrateCamera(
                objectPoints,
                imgPts,
                imageSize,
                K,
                D,
                rvecs,
                tvecs,
                flags,
                criteria);
        }

        perErr.clear();
        perErr.reserve(idxs.size());
        for (size_t j = 0; j < idxs.size(); ++j) {
            const int idx = idxs[j];
            const double e = perImageReprojError(obj, imagePoints[static_cast<size_t>(idx)], rvecs[j], tvecs[j], K, D, useFisheye);
            perErr.push_back(e);
        }

        return rms;
    };

    cv::Mat K, D;
    std::vector<cv::Mat> rvecs, tvecs;
    std::vector<double> perErr;

    double rms = 0.0;

    for (int iter = 0; iter <= maxRemove; ++iter) {
        rms = runCalib(kept, K, D, rvecs, tvecs, perErr);
        auto itWorst = std::max_element(perErr.begin(), perErr.end());
        const double worst = *itWorst;
        const int worstPos = static_cast<int>(std::distance(perErr.begin(), itWorst));

        std::cout << "Iter " << iter << ": kept=" << kept.size()
                  << " rms=" << rms
                  << " mean=" << mean(perErr)
                  << " max=" << worst << "\n";

        if (worst > maxPerImageError && kept.size() > 8) {
            const int worstIdx = kept[worstPos];
            rejected.push_back({paths[static_cast<size_t>(worstIdx)], worst});
            kept.erase(kept.begin() + worstPos);
            continue;
        }
        break;
    }

    // Final per-image errors map
    std::map<std::string, double> perImage;
    for (size_t j = 0; j < kept.size(); ++j) {
        const int idx = kept[j];
        perImage[paths[static_cast<size_t>(idx)]] = perErr[j];
    }

    std::cout << "\n=== FINAL INTRINSICS ===\n";
    std::cout << "Model:        " << model << "\n";
    std::cout << "Used images: " << kept.size() << " (rejected " << rejected.size() << ")\n";
    std::cout << "RMS (solver): " << rms << " px\n";
    std::cout << "Mean reproj:  " << mean(perErr) << " px\n";
    std::cout << "Max reproj:   " << (*std::max_element(perErr.begin(), perErr.end())) << " px\n";
    std::cout << "Image size:   " << imageSize.width << "x" << imageSize.height << "\n\n";
    std::cout << "K:\n" << K << "\n\n";
    std::cout << "D:\n" << D.t() << "\n";

    // Save plain text (simple + robust)
    const fs::path outPath = fs::path(output);
    const fs::path outAbs = fs::absolute(outPath);
    try {
        if (outPath.has_parent_path()) {
            fs::create_directories(outPath.parent_path());
        }
    } catch (...) {
        // ignore
    }

    std::ofstream os(outPath, std::ios::out | std::ios::trunc);
    if (!os.is_open()) {
        std::cerr << "ERROR: Cannot write " << outAbs.string() << "\n";
        return 1;
    }

    const double maxErr = *std::max_element(perErr.begin(), perErr.end());

    os << "model=" << model << "\n";
    os << "pattern=" << pattern.width << "x" << pattern.height << "\n";
    os << "square_mm=" << squareMm << "\n";
    os << "image_size=" << imageSize.width << "x" << imageSize.height << "\n";
    os << "used_images=" << kept.size() << "\n";
    os << "rejected_images=" << rejected.size() << "\n";
    os << "rms_solver_px=" << rms << "\n";
    os << "mean_reproj_px=" << mean(perErr) << "\n";
    os << "max_reproj_px=" << maxErr << "\n";

    os << "\nK:\n";
    for (int r = 0; r < 3; ++r) {
        os << K.at<double>(r, 0) << " " << K.at<double>(r, 1) << " " << K.at<double>(r, 2) << "\n";
    }

    os << "\nD:";
    {
        cv::Mat d = D.reshape(1, 1);
        for (int i = 0; i < d.cols; ++i) {
            os << (i == 0 ? " " : " ") << d.at<double>(0, i);
        }
        os << "\n";
    }

    os << "\nper_image_error_px:\n";
    for (const auto& kv : perImage) {
        os << fs::path(kv.first).filename().string() << " " << kv.second << "\n";
    }

    if (!rejected.empty()) {
        os << "\nrejected:\n";
        for (const auto& r : rejected) {
            os << fs::path(r.path).filename().string() << " " << r.errorPx << "\n";
        }
    }

    os.flush();
    if (!os.good()) {
        std::cerr << "ERROR: Write failed for " << outAbs.string() << "\n";
        return 1;
    }

    std::cout << "\nSaved: " << outAbs.string() << "\n";

    return 0;
}
