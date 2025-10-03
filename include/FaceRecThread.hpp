#pragma once
#include <atomic>
#include <chrono>
#include <map>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/objdetect/face.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#define IN_TRAIN

class FaceRec {
   public:
    FaceRec(const std::string detector_model_path, int size_w, int size_h) : cap_(0) {
        recognizer = cv::FaceRecognizerSF::create(detector_model_path, "");

        if (!cap_.isOpened()) {
            throw std::runtime_error("Error: Could not open camera.");
        }
    }

    ~FaceRec() { cap_.release(); }

    int start() {
        if (running_.load(std::memory_order_acquire)) {
            return 1;  // Already running
        }

        running_.store(true, std::memory_order_release);
        return 0;
    }

    void run() {
        while (running_.load(std::memory_order_acquire)) {
            cv::Mat frame;
            cap_ >> frame;  // Capture a new frame
            if (frame.empty()) {
                std::cerr << "WARNING: Empty frame captured." << std::endl;
                continue;
            }

            cv::Mat feature;
            recognizer->feature(frame, feature);

            // Here you would add face recognition processing
            // For demonstration, we just display the frame
            showFrame(frame);

            std::this_thread::sleep_for(
                std::chrono::milliseconds(30));  // Simulate processing delay
        }
    }

    int stop() {
        if (!running_.load(std::memory_order_acquire)) return 1;
        running_.store(false, std::memory_order_release);
        return 0;
    }

#ifdef IN_TRAIN
    void train(const std::map<int, std::string>& label_name_map) {
        // TODO:
        // Implement training logic here
        // For example, load images, extract features, and train the recognizer
        std::cout << "Training with " << label_name_map.size() << " labels." << std::endl;
    }
#endif

   private:
    std::atomic<bool> running_{false};

    void showFrame(const cv::Mat& frame) {
        // TODO: draw the human face rectangle and name
        cv::imshow("Face Recognition", frame);
        cv::waitKey(1);  // Needed to display the image
    }

    cv::VideoCapture cap_;  // OpenCV camera object
    cv::Ptr<cv::FaceRecognizerSF> recognizer;
};
