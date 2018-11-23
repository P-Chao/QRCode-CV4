cmake_minimum_required(VERSION 3.0)
project(quirc)

set(CURR_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include")

set_property(GLOBAL PROPERTY QUIRC_INCLUDE_DIR ${CURR_INCLUDE_DIR})
include_directories(${CURR_INCLUDE_DIR})

file(GLOB_RECURSE quirc_headers RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "include/*.h")
file(GLOB_RECURSE quirc_sources RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "src/*.c")

add_library(${PROJECT_NAME} STATIC ${quirc_headers} ${quirc_sources})
