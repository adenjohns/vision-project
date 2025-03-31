cmake_minimum_required(VERSION 3.4)

project(cpp_example)

find_package(ArducamDepthCamera REQUIRED)
set(CLIBS ArducamDepthCamera::ArducamDepthCamera4C)
set(CXXLIBS ArducamDepthCamera::ArducamDepthCamera)

include_directories("/usr/include/eigen3")
find_package(Eigen3 3.4 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(PIGPIO_LIBS pigpio rt pthread)

add_executable(preview_depth_IMU preview_depth_IMU.cpp RPI_BNO055.cpp) 
target_link_libraries(preview_depth_IMU ${CXXLIBS} ${OpenCV_LIBS} ${PIGPIO_LIBS})

# add_executable(preview_depth preview_depth.cpp)
# target_link_libraries(preview_depth ${CXXLIBS} ${OpenCV_LIBS})

# add_executable(capture_raw capture_raw.cpp)
# target_link_libraries(capture_raw ${CXXLIBS} ${OpenCV_LIBS})

# add_executable(preview_usb preview_by_usb.cpp)
# target_link_libraries(preview_usb ${CXXLIBS} ${OpenCV_LIBS})

# aux_source_directory(example DIR_SRCS)
# add_executable(example ${DIR_SRCS})
# target_link_libraries(example ${CXXLIBS} ${OpenCV_LIBS})

# add_executable(hello_world hello_world.cpp)