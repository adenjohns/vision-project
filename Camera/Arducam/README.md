# Arducam Depth Camera

## Overview

This project is a use example based on arducam's depth camera which includes basic image rendering using opencv.
The depth camera is the depth data obtained by calculating the phase difference based on the transmitted modulated pulse. The resolution of the camera is 240*180. Currently, it has two range modes: 2 meters and 4 meters. The measurement error is within 2 cm.
The depth camera supports CSI and USB two connection methods, and needs an additional 5V 2A current power supply for the camera.

## Quick Start

### Clone the entire repository for additional features (not required). 

Clone this repository and enter the directory. The current repository used within this project only has depth processing files in Python and C++. The original has features such as a point cloud function as well as compability with ROS2.

```shell
  git clone https://github.com/ArduCAM/Arducam_tof_camera.git
  cd Arducam_tof_camera
```

### Install dependencies for Raspberry Pi

> Run in the Arducam folder
> Whatever you want to run the C/C++ examples or Python examples, you need to install the dependencies.

```shell
  ./Install_dependencies.sh
```

## Run Examples

> Platform: Raspberry Pi

### Depth Examples

#### Python

##### Run

###### Python Example

> Run in the Python_Example folder

```shell
  cd Python_Example
```
> Turn on the Python virtual environment

```shell
  source venv/bin/activate
```

```shell
  python3 follow_the_gap.py
  #or
  python3 preview_depth.py
```

#### C++

##### Compile

> Run in the Cpp_Example/build folder. If there are any changes within the build file, run both commands, otherwise just skip to the second command. 

```shell
  cmake ..
```
```shell
  make 
```

##### Run

> Run in the Cpp_Example/build folder

```shell
  ./preview_depth
  #or
  ./capture_raw
```

##### Remove Object and Executable Files

> Run in the Cpp_Example/build folder. Sometimes the compiler will link or compile files incorrectly and the only way to get a fresh start is to remove all the object and executable files which the following command does (make clean doesn't work for some reason). Removing and rebuilding the build directory also works.

```shell
  rm -r CMake*
``` 
> After clearing out the CMake files run the follow command to remake and link the proper files. 

```shell
  cmake ..
```
