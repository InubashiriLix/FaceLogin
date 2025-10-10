cmake_minimum_required(VERSION 3.20)
project(FaceLogin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui dnn objdetect)

add_executable(train
    src/main.cc
)

target_include_directories(train PRIVATE
    include
)

target_link_libraries(train PRIVATE
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_dnn
    opencv_objdetect
)
