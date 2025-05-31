cmake_minimum_required(VERSION 3.4)
project( Vid2Img )
project(VisionIntegration)

# Find and Configure all Dependencies

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

# ALSA for I2S audio
find_package(ALSA REQUIRED)
include_directories(${ALSA_INCLUDE_DIRS})

# Include Directory
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

# Libraries 
set(PIGPIO_LIBS pigpio rt pthread)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_FLAGS "-g -std=c++11")

# Add executable from src/ folder 
#add_executable(VisionApp src/final_integration.cpp src/RPI_BNO055.cpp) 
#target_link_libraries(VisionApp ${CXXLIBS} ${OpenCV_LIBS} ${PIGPIO_LIBS} ${ALSA_LIBRARIES})

add_executable( detector src/object_detection_yolo.cpp )
target_link_libraries(detector  ${OpenCV_LIBS} )


#### Old test executable files
#add_executable(VisionApp src/camera_integration.cpp ) 
#target_link_libraries(VisionApp ${CXXLIBS} ${OpenCV_LIBS})

#add_executable(detector src/object_detection_save_frame.cpp )
#target_link_libraries(detector ${OpenCV_LIBS})
	
#add_executable(preview_depth_IMU src/preview_depth_IMU.cpp src/RPI_BNO055.cpp) 
#target_link_libraries(preview_depth_IMU ${CXXLIBS} ${OpenCV_LIBS} ${PIGPIO_LIBS} ${ALSA_LIBRARIES})

#add_executable(preview_depth src/preview_depth.cpp)
#target_link_libraries(preview_depth ${CXXLIBS} ${OpenCV_LIBS})

#add_executable(hello_world src/hello_world.cpp)
