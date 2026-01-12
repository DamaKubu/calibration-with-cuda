#include <opencv2/opencv.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>
#include <vector>

int main() {
    try {
        // Try to open all available cameras
        std::vector<cv::VideoCapture> cameras;
        std::vector<int> camera_indices;
        std::vector<cv::Ptr<cv::cuda::BackgroundSubtractorMOG2>> mog2_list;
        
        std::cout << "Scanning for available cameras..." << std::endl;
        
        // Try cameras 0-5
        for (int i = 0; i < 6; i++) {
            cv::VideoCapture cap(i);
            if (cap.isOpened()) {
                cameras.push_back(cap);
                camera_indices.push_back(i);
                mog2_list.push_back(cv::cuda::createBackgroundSubtractorMOG2());
                std::cout << "Camera " << i << " opened successfully!" << std::endl;
            }
        }
        
        if (cameras.empty()) {
            std::cerr << "Error: No cameras found!" << std::endl;
            return -1;
        }
        
        std::cout << "Total cameras found: " << cameras.size() << std::endl;
        std::cout << "Press ESC to exit" << std::endl << std::endl;
        
        std::vector<cv::cuda::GpuMat> frames_gpu(cameras.size());
        std::vector<cv::cuda::GpuMat> fgmasks_gpu(cameras.size());
        std::vector<cv::Mat> frames(cameras.size());
        std::vector<cv::Mat> fgmasks(cameras.size());

        while (true) {
            // Capture and process all cameras
            bool all_empty = true;
            for (size_t i = 0; i < cameras.size(); i++) {
                cameras[i] >> frames[i];
                if (!frames[i].empty()) {
                    all_empty = false;
                    
                    // Upload frame to GPU
                    frames_gpu[i].upload(frames[i]);

                    // Apply MOG2 - detects moving objects
                    mog2_list[i]->apply(frames_gpu[i], fgmasks_gpu[i]);

                    // Download mask to CPU
                    fgmasks_gpu[i].download(fgmasks[i]);

                    // Count changed pixels (foreground objects detected)
                    int changed_pixels = cv::countNonZero(fgmasks[i]);
                    
                    std::string window_name = "Camera " + std::to_string(camera_indices[i]);
                    std::string mask_name = "Mask - Camera " + std::to_string(camera_indices[i]);

                    // Make windows resizable (default imshow creates AUTOSIZE windows).
                    static std::vector<int> created(16, 0);
                    if (camera_indices[i] >= 0 && camera_indices[i] < static_cast<int>(created.size()) && created[camera_indices[i]] == 0) {
                        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
                        cv::namedWindow(mask_name, cv::WINDOW_NORMAL);
                        cv::resizeWindow(window_name, 960, 540);
                        cv::resizeWindow(mask_name, 960, 540);
                        created[camera_indices[i]] = 1;
                    }
                    
                    // Display info on the frame
                    cv::putText(frames[i], "Camera: " + std::to_string(camera_indices[i]), 
                               cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                    cv::putText(frames[i], "Changed pixels: " + std::to_string(changed_pixels), 
                               cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
                    
                    // Show results
                    cv::imshow(window_name, frames[i]);
                    cv::imshow(mask_name, fgmasks[i]);
                    
                    std::cout << "\rCamera " << camera_indices[i] << " | Changed pixels: " 
                              << changed_pixels << "   " << std::flush;
                }
            }
            
            if (all_empty) break;

            if (cv::waitKey(1) == 27) break; // ESC to exit
        }

        // Release all cameras
        for (auto& cap : cameras) {
            cap.release();
        }
        cv::destroyAllWindows();
        
        std::cout << "\n\nAll cameras closed." << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Exception: " << e.what() << std::endl;
    }
    return 0;
}
