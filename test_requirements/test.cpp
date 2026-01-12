#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC3);
    std::cout << "Image created: " << img.rows << "x" << img.cols << std::endl;
    return 0;
}
