cmake_minimum_required(VERSION 3.20)
project(FaceLogin LANGUAGES CXX)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 生成 compile_commands.json 方便 LSP
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

add_library(FaceLogin STATIC
    src/add.cc
)

target_include_directories(FaceLogin PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(FaceLogin PRIVATE ${OpenCV_LIBS})

add_executable(main
    src/main.cc
)

target_link_libraries(main PRIVATE FaceLogin ${OpenCV_LIBS})
