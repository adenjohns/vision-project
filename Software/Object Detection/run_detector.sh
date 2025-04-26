#!/bin/bash
# set -e

# figure out where script lives (project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR"
# BUILD_DIR="$PROJECT_DIR/build"

echo "Project root: $PROJECT_DIR"
# echo "Build dir: $BUILD_DIR"

# building project
echo "Cleaning and rebuilding project..."
rm -rf build
mkdir build
cd build
cmake ..
make -j"$(nproc)"


# camera setup
echo "Applying camera settings..."
v4l2-ctl -d /dev/video0 -c auto_exposure=1 -c exposure_time_absolute=100 -c gain=10 -c brightness=0 -c contrast=32

# run the detector
echo "Launching detector..."
# cd build
./detector --input "v4l2src device=/dev/video0 ! videoconvert ! video/x-raw,format=BGR,width=320,height=240 ! appsink"
# "$PROJECT_DIR/build/detector" --cfg "$PROJECT_DIR/cfg/yolov3-tiny.cfg" --weights "$PROJECT_DIR/weights/yolov3-tiny.weights" --input "v4l2src device=/dev/video0 ! videoconvert ! video/x-raw,format=BGR,width=320,height=240 ! appsink"



