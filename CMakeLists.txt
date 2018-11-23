cmake_minimum_required(VERSION 3.0)

add_subdirectory(quirc)
project(qrcode)

file(GLOB SRC_LIST "src/*.cpp")
file(GLOB INC_LIST "src/*.h")

find_package(OpenCV REQUIRED)
include_directories(. quirc/include ${OpenCV_INCLUDE_DIRS})
add_definitions(-DHAVE_OPENCV -DHAVE_QUIRC)

add_executable(qrcode ${SRC_LIST} ${INC_LIST})
target_link_libraries(qrcode ${OpenCV_LIBS} quirc)
