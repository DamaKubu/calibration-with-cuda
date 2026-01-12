#include <opencv2/opencv.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>

int main() {
    try {
        // Open default camera
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open camera!" << std::endl;
            return -1;
        }

        // Create CUDA MOG2 background subtractor
        cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> mog2 = 
            cv::cuda::createBackgroundSubtractorMOG2();
        
        cv::cuda::GpuMat frame_gpu, fgmask_gpu;
        cv::Mat frame, fgmask;

        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            // Upload frame to GPU
            frame_gpu.upload(frame);

            // Apply MOG2
            mog2->apply(frame_gpu, fgmask_gpu);

            // Download mask to CPU
            fgmask_gpu.download(fgmask);

            // Count changed pixels
            int changed_pixels = cv::countNonZero(fgmask);
            std::cout << "\rChanged pixels: " << changed_pixels << "   " << std::flush;

            // Show results
            cv::imshow("Camera", frame);
            cv::imshow("Foreground mask", fgmask);

            if (cv::waitKey(1) == 27) break; // ESC to exit
        }

        cap.release();
        cv::destroyAllWindows();
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Exception: " << e.what() << std::endl;
    }
    return 0;
}
