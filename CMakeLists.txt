cmake_minimum_required(VERSION 3.0)

add_subdirectory(quirc)
project(qrcode)

file(GLOB QR_SRC "src/*.cpp")
file(GLOB QR_INC "src/*.h")

file(GLOB AT_SRC "apriltag/*.cpp")
file(GLOB AT_INC "apriltag/*.h")

source_group("QR Code" FILES ${QR_SRC} ${QR_INC})
source_group("AprilTag" FILES ${AT_SRC} ${AT_INC})

find_package(OpenCV REQUIRED)
include_directories(. quirc/include ${OpenCV_INCLUDE_DIRS})
add_definitions(-DHAVE_OPENCV -DHAVE_QUIRC -DNOMINMAX)

add_executable(qrcode ${QR_SRC} ${QR_INC} ${AT_SRC} ${AT_INC})
target_link_libraries(qrcode ${OpenCV_LIBS} quirc)
