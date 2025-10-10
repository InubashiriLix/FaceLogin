#include <chrono>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "FaceRecog.hpp"

static std::atomic<bool> g_stop{false};
void on_sigint(int) { g_stop.store(true); }

static void print_usage(const char* prog) {
    std::cout
        <<
        R"(Usage:
  )" << prog
        << R"( --dataset <root> [--save <gallery.yaml>] [--no-run]
  )" << prog
        << R"( --load <gallery.yaml> [--dataset <root>] [--no-run]
  )" << prog
        << R"( [--yunet <onnx>] [--sface <onnx>] [--cam <idx>] [--thr <cos_thr>] [--win <name>]

Options:
  --dataset <root>        数据集根目录（每个一级子目录=一个身份；里面放若干 *.jpg/*.png）
  --save <gallery.yaml>   训练完成后将特征库保存到 YAML
  --load <gallery.yaml>   启动时从 YAML 载入特征库（可与 --dataset 合用：先载入再增量注册）
  --yunet <path>          YuNet onnx 模型路径（缺省使用 FaceRecogConf 的默认值）
  --sface <path>          SFace onnx 模型路径（缺省使用 FaceRecogConf 的默认值）
  --cam <idx>             摄像头索引（默认 0）
  --thr <cos_thr>         识别阈值（默认 0.45，越大越严格）
  --win <name>            窗口名（默认 "FaceRecog"）
  --no-run                只训练/保存，不启动摄像头推理
  --help                  打印本帮助

示例:
  # 1) 从文件夹训练并保存，不跑相机
  )" << prog
        << R"( --dataset dataset_root --save gallery.yaml --no-run

  # 2) 直接加载已有库并跑相机识别
  )" << prog
        << R"( --load gallery.yaml

  # 3) 先加载旧库，再从新数据集增量注册，随后保存并跑相机
  )" << prog
        << R"( --load old.yaml --dataset new_dataset --save merged.yaml
)" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "shit" << std::endl;
    std::signal(SIGINT, on_sigint);

    std::string dataset_root;
    std::string save_yaml;
    std::string load_yaml;
    std::string yunet_path;
    std::string sface_path;
    std::string win_name;
    int cam_idx = 0;
    double thr = -1.0;
    bool no_run = false;

    // ---- 简易参数解析（跨平台、无依赖） ----
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_arg = [&](const char* flag) -> bool {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after " << flag << "\n";
                std::exit(1);
            }
            return true;
        };
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (a == "--dataset") {
            if (need_arg("--dataset")) dataset_root = argv[++i];
        } else if (a == "--save") {
            if (need_arg("--save")) save_yaml = argv[++i];
        } else if (a == "--load") {
            if (need_arg("--load")) load_yaml = argv[++i];
        } else if (a == "--yunet") {
            if (need_arg("--yunet")) yunet_path = argv[++i];
        } else if (a == "--sface") {
            if (need_arg("--sface")) sface_path = argv[++i];
        } else if (a == "--cam") {
            if (need_arg("--cam")) cam_idx = std::stoi(argv[++i]);
        } else if (a == "--thr") {
            if (need_arg("--thr")) thr = std::stod(argv[++i]);
        } else if (a == "--win") {
            if (need_arg("--win")) win_name = argv[++i];
        } else if (a == "--no-run") {
            no_run = true;
        } else {
            std::cerr << "Unknown arg: " << a << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // ---- 配置 ----
    FaceRecogConf conf;  // 使用你的默认值
    if (!yunet_path.empty()) conf.yunet_onnx_path = yunet_path;
    if (!sface_path.empty()) conf.sface_onnx_path = sface_path;
    if (!win_name.empty()) conf.win_name = win_name;
    conf.cam_index = cam_idx;
    if (thr > 0.0) conf.recog_cosine_thr = thr;

    // ---- 创建应用（会打开相机与加载模型）----
    try {
        FaceRecog app(conf);

        // ---- 载入旧库（可选）----
        if (!load_yaml.empty()) {
            if (app.load_gallery(load_yaml)) {
                std::cout << "[load] loaded gallery: " << load_yaml << "\n";
            } else {
                std::cerr << "[load] failed to load gallery: " << load_yaml << "\n";
            }
        }

        // ---- 从数据集根目录训练/建库（可选）----
        if (!dataset_root.empty()) {
            int ok = app.enroll_from_folder_tree(dataset_root);
            std::cout << "[enroll] people added from '" << dataset_root << "': " << ok << "\n";
        }

        // ---- 保存库（可选）----
        if (!save_yaml.empty()) {
            if (app.save_gallery(save_yaml)) {
                std::cout << "[save] gallery saved to: " << save_yaml << "\n";
            } else {
                std::cerr << "[save] failed to save gallery: " << save_yaml << "\n";
            }
        }

        // ---- 只训练不跑相机 ----
        if (no_run) {
            std::cout << "[done] training/building finished. '--no-run' set, exit.\n";
            return 0;
        }

        // ---- 跑相机识别 ----
        if (!app.start()) {
            std::cerr << "Failed to start worker.\n";
            return 2;
        }
        std::cout << "[run] press Ctrl+C 退出；或在窗口里按 ESC 停止画面\n";

        // Ctrl+C 时退出；即便你在窗口里按了 ESC 停止了处理线程，这里依然建议 Ctrl+C 触发
        // stop()，以便 join。
        while (!g_stop.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        app.stop();
        std::cout << "[run] stopped.\n";
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return 3;
    }
    return 0;
}
