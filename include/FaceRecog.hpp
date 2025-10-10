#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/objdetect/face.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct FaceRecogConf {
    // the model path
    //  the Yunet onnx path
    std::string yunet_onnx_path = "models/face_detection_yunet_2022mar.onnx";
    std::string sface_onnx_path = "models/face_recognition_sface_2021dec.onnx";

    // the input cam index
    int cam_index = 0;

    // the input camera resolution
    int cam_w = 1280;
    int cam_h = 720;
    int FPS = 30;

    // the detector model input resolution
    int detect_w = 320;
    int detect_h = 320;

    //
    float det_score_thr = 0.5;
    float det_nms_thr = 0.3;
    int det_toqk = 5000;
    double recog_cosine_thr = 0.45;

    int backend = cv::dnn::DNN_BACKEND_DEFAULT;
    int target = cv::dnn::DNN_TARGET_CPU;

    // the ui config
    std::string win_name = "FaceRecog";
};

class FaceRecog {
   public:
    explicit FaceRecog(FaceRecogConf conf) : m_conf(conf) {
        // open and set the camera
        if (!this->m_cap.open(m_conf.cam_index))
            throw std::runtime_error("Error: Could not open the camera index" +
                                     std::to_string(m_conf.cam_index));
        m_cap.set(cv::CAP_PROP_FRAME_WIDTH, m_conf.cam_w);
        m_cap.set(cv::CAP_PROP_FRAME_HEIGHT, m_conf.cam_h);
        m_cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        m_cap.set(cv::CAP_PROP_FPS, m_conf.FPS);

        // setup the detector (YuNet)
        m_detector = cv::FaceDetectorYN::create(m_conf.yunet_onnx_path, "",
                                                cv::Size(m_conf.detect_w, m_conf.detect_h),
                                                m_conf.det_score_thr, m_conf.det_nms_thr,
                                                m_conf.det_toqk, m_conf.backend, m_conf.target);
        if (m_detector.empty()) {
            throw std::runtime_error(
                "Error: create FaceDetectorYN failed. Check YuNet model path.");
        }

        // 3 setup the recognizer
        m_recognizer =
            cv::FaceRecognizerSF::create(m_conf.sface_onnx_path, "", m_conf.backend, m_conf.target);
        if (m_recognizer.empty()) {
            throw std::runtime_error(
                "Error: create FaceRecognizerSF failed. Check SFace model path.");
        }

        // setup the window name
        cv::namedWindow(m_conf.win_name);
    }

    ~FaceRecog() {
        stop();
        // TODO: release the resource like the camera
        m_cap.release();
        cv::destroyAllWindows();
    }

    bool start() {
        bool expect = false;
        if (!m_is_running.compare_exchange_strong(expect, true)) return false;
        m_th = std::thread(&FaceRecog::worker, this);
        return true;
    }

    void worker() {
        cv::Mat frame;

        // 简单的 FPS 统计
        auto t_prev = std::chrono::steady_clock::now();
        double fps = 0.0;

        while (m_is_running.load(std::memory_order_acquire)) {
            if (!m_cap.read(frame) || frame.empty()) {
                // if cannot read the frame, just skip this time
                std::cerr << "Error: Could not read a frame from camera." << std::endl;
                continue;
            }

            // 对当前帧设置输入尺寸（YuNet 要与当前图像尺寸对应）
            m_detector->setInputSize(frame.size());

            cv::Mat detections;
            m_detector->detect(frame, detections);
            // \- 0-1: x, y of bbox top left corner
            // \- 2-3: width, height of bbox
            // \- 4-5: x, y of right eye (blue point in the example image)
            // \- 6-7: x, y of left eye (red point in the example image)
            // \- 8-9: x, y of nose tip (green point in the example image)
            // \- 10-11: x, y of right corner of mouth (pink point in the example image)
            // \- 12-13: x, y of left corner of mouth (yellow point in the example image)
            // \- 14: face score

            std::vector<cv::Rect> rects;
            std::vector<std::string> tags;

            for (int i = 0; i < detections.rows; i++) {
                float score = detections.at<float>(i, 4);
                if (score < m_conf.det_score_thr) continue;

                int x = std::max(0, (int)std::floor(detections.at<float>(i, 0)));
                int y = std::max(0, (int)std::floor(detections.at<float>(i, 1)));
                int w = (int)std::floor(detections.at<float>(i, 2));
                int h = (int)std::floor(detections.at<float>(i, 3));
                cv::Rect box(x, y, std::min(w, frame.cols - x), std::min(h, frame.rows - y));
                rects.push_back(box);

                cv::Mat face_align;
                m_recognizer->alignCrop(frame, detections.row(i),
                                        face_align);  // use the api to align and crop automatically
                                                      // in to the variable face_align

                cv::Mat feat;
                if (!face_align.empty()) {  // get the feature of face
                    m_recognizer->feature(face_align, feat);
                }

                std::string tag = "Unknown";
                double best_sim = -2.0;

                if (!feat.empty()) {
                    normalize_l2_(feat);
                    std::lock_guard<std::mutex> lk(m_db_mtx);
                    for (auto& kv : m_db) {
                        double sim = cosine_(feat, kv.second);  // [-1,1]
                        if (sim > best_sim) {
                            best_sim = sim;
                            tag = m_label2name[kv.first];
                        }
                    }
                    if (best_sim < m_conf.recog_cosine_thr) {
                        tag = "Unknown";
                    } else {
                        char buf[128];
                        std::snprintf(buf, sizeof(buf), "%s (%.2f)", tag.c_str(), best_sim);
                        tag = buf;
                    }
                }
                tags.push_back(tag);
            }

            // show image
            // 叠加人脸框、姓名与 FPS
            draw_(frame, rects, tags, fps);
            cv::imshow(m_conf.win_name, frame);  // Show the frame
            if (cv::waitKey(1) == 27) break;     // Exit on 'ESC' key

            // prevent high CPU usage
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            // 更新 FPS（EMA）
            auto t_now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(t_now - t_prev).count();
            t_prev = t_now;
            if (dt > 1e-6) {
                double inst = 1.0 / dt;
                fps = fps * 0.9 + inst * 0.1;
            }
        }
    }

    bool stop() {
        if (m_is_running.load(std::memory_order_acquire)) {
            m_is_running.store(false, std::memory_order_release);
            if (m_th.joinable()) {
                m_th.join();
            }
            return true;
        }
        return false;
    }

    // ======== 训练 / 建库（仅从图像列表/文件夹） ========

    // 从图片路径列表注册一个人
    bool enroll_images(int label, const std::string& name,
                       const std::vector<std::string>& img_paths) {
        if (img_paths.empty()) return false;
        std::vector<cv::Mat> feats;
        feats.reserve(img_paths.size());

        for (auto& p : img_paths) {
            cv::Mat img = cv::imread(p);
            if (img.empty()) continue;

            // 对齐第一张置信度最高的人脸
            cv::Mat aligned;
            {
                std::lock_guard<std::mutex> lk(m_det_mtx);
                m_detector->setInputSize(cv::Size(m_conf.detect_w, m_conf.detect_h));
                cv::Mat det;
                m_detector->detect(img, det);
                if (det.empty()) continue;
                int best = -1;
                float best_score = 0.f;
                for (int i = 0; i < det.rows; ++i) {
                    float sc = det.at<float>(i, 4);
                    if (sc > best_score) {
                        best_score = sc;
                        best = i;
                    }
                }
                if (best < 0) continue;
                m_recognizer->alignCrop(img, det.row(best), aligned);
            }
            if (aligned.empty()) continue;

            cv::Mat feat;
            m_recognizer->feature(aligned, feat);
            if (feat.empty()) continue;
            normalize_l2_(feat);
            feats.push_back(feat);
        }

        if (feats.empty()) return false;

        cv::Mat mean = mean_feature_(feats);  // 多图均值特征，更稳
        {
            std::lock_guard<std::mutex> lk(m_db_mtx);
            m_db[label] = mean;
            m_label2name[label] = name;
        }
        return true;
    }

    // 从目录注册一个人（递归遍历，按扩展名过滤）
    bool enroll_dir(int label, const std::string& name, const std::string& dir,
                    const std::vector<std::string>& exts = {".jpg", ".jpeg", ".png", ".bmp"}) {
        auto imgs = _list_images(dir, exts);
        return enroll_images(label, name, imgs);
    }

    // 从数据集根目录自动注册（每个一级子目录 = 1个人），返回成功注册的人数
    int enroll_from_folder_tree(const std::string& root, const std::vector<std::string>& exts = {
                                                             ".jpg", ".jpeg", ".png", ".bmp"}) {
        namespace fs = std::filesystem;
        int ok = 0, next_label = 0;

        std::error_code ec;
        for (auto& entry : fs::directory_iterator(root, ec)) {
            if (ec) break;
            if (!entry.is_directory()) continue;

            const std::string person_dir = entry.path().string();
            const std::string person_name = entry.path().filename().string();
            const int label = next_label++;

            auto imgs = _list_images(person_dir, exts);
            if (imgs.empty()) {
                std::cerr << "[enroll] skip '" << person_name << "' (no images)\n";
                continue;
            }

            if (enroll_images(label, person_name, imgs)) {
                std::cout << "[enroll] " << person_name << " (" << imgs.size() << " imgs) -> label "
                          << label << "\n";
                ++ok;
            } else {
                std::cerr << "[enroll] failed: " << person_name << "\n";
            }
        }
        return ok;
    }

    // 保存/载入特征库（YAML）
    bool save_gallery(const std::string& yaml_path) {
        std::lock_guard<std::mutex> lk(m_db_mtx);
        try {
            cv::FileStorage fs(yaml_path, cv::FileStorage::WRITE);
            fs << "count" << (int)m_db.size();
            fs << "recog_cosine_thr" << m_conf.recog_cosine_thr;
            fs << "items" << "[";
            for (auto& kv : m_db) {
                int label = kv.first;
                fs << "{";
                fs << "label" << label;
                fs << "name" << m_label2name[label];
                fs << "feat" << kv.second;
                fs << "}";
            }
            fs << "]";
            return true;
        } catch (...) {
            return false;
        }
    }

    bool load_gallery(const std::string& yaml_path) {
        std::lock_guard<std::mutex> lk(m_db_mtx);
        m_db.clear();
        m_label2name.clear();
        try {
            cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
            if (!fs.isOpened()) return false;
            double thr = m_conf.recog_cosine_thr;
            fs["recog_cosine_thr"] >> thr;
            if (thr > 0.0) m_conf.recog_cosine_thr = thr;

            cv::FileNode items = fs["items"];
            for (const auto& n : items) {
                int label = -1;
                std::string name;
                cv::Mat feat;
                n["label"] >> label;
                n["name"] >> name;
                n["feat"] >> feat;
                if (label < 0 || feat.empty()) continue;
                normalize_l2_(feat);
                m_db[label] = feat;
                m_label2name[label] = name;
            }
            return !m_db.empty();
        } catch (...) {
            return false;
        }
    }

   private:
    // the config
    FaceRecogConf m_conf;

    // the multi-thread control
    std::atomic<bool> m_is_running{false};
    std::thread m_th;

    // the resources
    // 1. camera
    cv::VideoCapture m_cap;

    // 2. detector
    cv::Ptr<cv::FaceDetectorYN> m_detector;
    cv::Ptr<cv::FaceRecognizerSF> m_recognizer;

    std::map<int, cv::Mat> m_db;
    std::map<int, std::string> m_label2name;
    std::mutex m_db_mtx;

    // 与注册过程共享 detector 时的轻量互斥，避免 setInputSize/detect 与 worker 竞争
    std::mutex m_det_mtx;

   private:
    static void normalize_l2_(cv::Mat& feat_rowvec) {
        // SFace 输出常用 L2 归一化保证余弦稳定
        CV_Assert(feat_rowvec.total() > 0);
        cv::Mat f;
        feat_rowvec.convertTo(f, CV_32F);
        float* p = f.ptr<float>(0);
        double n2 = 0.0;
        for (int i = 0; i < f.total(); ++i) n2 += p[i] * p[i];
        n2 = std::sqrt(std::max(n2, 1e-12));
        for (int i = 0; i < f.total(); ++i) p[i] = (float)(p[i] / n2);
        feat_rowvec = f;
    }

    static double cosine_(const cv::Mat& a_row, const cv::Mat& b_row) {
        CV_Assert(a_row.total() == b_row.total());
        const float* a = a_row.ptr<float>(0);
        const float* b = b_row.ptr<float>(0);
        double s = 0.0;
        for (int i = 0; i < a_row.total(); ++i) s += (double)a[i] * (double)b[i];
        return s;  // a,b 已L2归一化 → [-1,1]
    }

    static cv::Mat mean_feature_(const std::vector<cv::Mat>& feats) {
        CV_Assert(!feats.empty());
        cv::Mat acc = cv::Mat::zeros(feats[0].size(), CV_32F);
        for (auto& f : feats) {
            cv::Mat ff;
            f.convertTo(ff, CV_32F);
            acc += ff;
        }
        acc /= (float)feats.size();
        normalize_l2_(acc);
        return acc;
    }

    // 从一张图里自动找第一张脸并对齐（用于注册）
    bool auto_align_first_face_(const cv::Mat& img, cv::Mat& aligned) {
        cv::Mat det;
        m_detector->setInputSize(cv::Size(m_conf.detect_w, m_conf.detect_h));
        m_detector->detect(img, det);
        if (det.empty()) return false;
        int best = -1;
        float best_score = 0.f;
        for (int i = 0; i < det.rows; ++i) {
            float sc = det.at<float>(i, 4);
            if (sc > best_score) {
                best_score = sc;
                best = i;
            }
        }
        if (best < 0) return false;
        m_recognizer->alignCrop(img, det.row(best), aligned);
        return !aligned.empty();
    }

    // 绘制框、姓名与 fps
    void draw_(cv::Mat& frame, const std::vector<cv::Rect>& rects,
               const std::vector<std::string>& tags, double fps) {
        for (size_t i = 0; i < rects.size(); ++i) {
            const auto& r = rects[i];
            cv::rectangle(frame, r, {0, 255, 0}, 2);
            auto label = (i < tags.size() ? tags[i] : "Unknown");
            int base = 0;
            cv::Size sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &base);
            cv::rectangle(frame, {r.x, std::max(0, r.y - sz.height - 6)}, {r.x + sz.width + 4, r.y},
                          {0, 255, 0}, cv::FILLED);
            cv::putText(frame, label, {r.x + 2, r.y - 4}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 0, 0},
                        2);
        }
        // 左上角显示 FPS
        char buf[64];
        std::snprintf(buf, sizeof(buf), "FPS: %.1f", fps);
        cv::putText(frame, buf, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 255}, 2);
    }

    // 递归收集图片
    static bool _has_ext(const std::string& p, const std::vector<std::string>& exts) {
        std::string s = p;
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });
        for (auto& e : exts) {
            std::string ee = e;
            std::transform(ee.begin(), ee.end(), ee.begin(),
                           [](unsigned char c) { return (char)std::tolower(c); });
            if (s.size() >= ee.size() && s.rfind(ee) == s.size() - ee.size()) return true;
        }
        return false;
    }

    static std::vector<std::string> _list_images(const std::string& dir,
                                                 const std::vector<std::string>& exts) {
        std::vector<std::string> out;
        std::error_code ec;
        for (auto& entry : std::filesystem::recursive_directory_iterator(dir, ec)) {
            if (ec) break;
            if (!entry.is_regular_file()) continue;
            auto p = entry.path().string();
            if (_has_ext(p, exts)) out.push_back(p);
        }
        std::sort(out.begin(), out.end());
        return out;
    }
};
