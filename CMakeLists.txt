cmake_minimum_required(VERSION 2.8.3)
project(wVio)


set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
find_package(Pangolin 0.4 REQUIRED)
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")
include_directories("/usr/local/include/eigen3")
find_package(OpenCV REQUIRED)
#find_package(Ceres REQUIRED)

add_executable(sfm sfm_test.cpp)
target_link_libraries(sfm ${OpenCV_LIBS} 
#${CERES_LIBRARIES} 
${Pangolin_LIBRARIES}
)
