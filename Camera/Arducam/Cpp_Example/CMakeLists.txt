cmake_minimum_required(VERSION 3.4)

project(cpp_example)

# Arducam
find_package(ArducamDepthCamera REQUIRED)
set(CLIBS ArducamDepthCamera::ArducamDepthCamera4C)
set(CXXLIBS ArducamDepthCamera::ArducamDepthCamera)

# Eigen
include_directories("/usr/include/eigen3")
find_package(Eigen3 3.4 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include Directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# Libraries 
set(PIGPIO_LIBS pigpiod_if2 rt pthread)

# Add executable from src/ folder 
add_executable(preview_depth_IMU src/preview_depth_IMU.cpp src/RPI_BNO055.cpp) 
target_link_libraries(preview_depth_IMU ${CXXLIBS} ${OpenCV_LIBS} ${PIGPIO_LIBS})

# add_executable(preview_depth src/preview_depth.cpp)
# target_link_libraries(preview_depth ${CXXLIBS} ${OpenCV_LIBS})

# aux_source_directory(example DIR_SRCS)
# add_executable(example ${DIR_SRCS})
# target_link_libraries(example ${CXXLIBS} ${OpenCV_LIBS})

# add_executable(hello_world src/hello_world.cpp)
