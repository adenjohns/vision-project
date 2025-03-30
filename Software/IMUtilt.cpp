#include <unistd.h>
#include <csignal>
#include <pigpio.h>
#include "RPI_BNO055.h"
#include <cmath>
#include <cstdlib>
#include <chrono>


std::cout << "BNO055 Sensor Test using pigpio\n";

// Initialize pigpio early (for precise timing and I2C)
gpioInitialise();

// Quick I2C test on bus 3 to ensure proper bus access
int handle = i2cOpen(3, 0x28, 0);
if (handle < 0) {
    std::cerr << "Direct i2cOpen test failed with error: " << handle << std::endl;
} else {
    std::cout << "Direct i2cOpen test succeeded with handle: " << handle << std::endl;
    i2cClose(handle);
}

// Create the sensor instance using bus 1 and the default address 0x28
RPI_BNO055 bno(-1, BNO055_ADDRESS_A, 1);

// Initialize the sensor
if (!bno.begin()) {
    std::cerr << "Failed to initialize BNO055 sensor!\n";
    return -1;
}
std::cout << "BNO055 sensor initialized successfully!\n";

// Enable external crystal for improved accuracy (if available)
bno.setExtCrystalUse(true);

// Retrieve and display sensor revision information
bno055_rev_info_t revInfo;
bno.getRevInfo(&revInfo);
std::cout << "Sensor Information:\n";
std::cout << "- System Rev: " << (int)revInfo.bl_rev << "\n";
std::cout << "- Accelerometer Rev: " << (int)revInfo.accel_rev << "\n";
std::cout << "- Gyroscope Rev: " << (int)revInfo.gyro_rev << "\n";
std::cout << "- Magnetometer Rev: " << (int)revInfo.mag_rev << "\n";
std::cout << "- Software Rev: " << revInfo.sw_rev << "\n";

std::cout << "\nReading sensor data...\n";
std::cout << "Press Ctrl+C to exit\n\n";

// Variables for motion detection (to trigger shutdown if no motion is detected)
imu::Vector<3> previousGyro;
bool firstReading = true;
std::chrono::steady_clock::time_point lastMotionTime;
const float MOTION_THRESHOLD = 5.0f;  // 5° threshold for detecting significant motion changes
const int SHUTDOWN_TIMEOUT = 30;      // 30 seconds timeout

// Main loop
while (running) {
    // Get calibration status for system, gyroscope, accelerometer, and magnetometer (values 0-3)
    uint8_t sysCal, gyroCal, accelCal, magCal;
    bno.getCalibration(&sysCal, &gyroCal, &accelCal, &magCal);
    
    // Clear the current line and output calibration status
    std::cout << "\033[2K\r";
    std::cout << "Calibration: Sys=" << (int)sysCal << " Gyro=" << (int)gyroCal 
              << " Accel=" << (int)accelCal << " Mag=" << (int)magCal << " | ";
    
    // Read Euler angles (for general orientation display)
    imu::Vector<3> euler = bno.getVector(VECTOR_EULER);
    
    // Read accelerometer and gyroscope data
    imu::Vector<3> accelData = bno.getVector(VECTOR_ACCELEROMETER);
    imu::Vector<3> gyroData = bno.getVector(VECTOR_GYROSCOPE);
    
    // Motion detection: compare current gyroscope readings with previous values
    bool motionDetected = false;
    if (!firstReading) {
        motionDetected = hasChangedBy(gyroData.x(), previousGyro.x(), MOTION_THRESHOLD) ||
                         hasChangedBy(gyroData.y(), previousGyro.y(), MOTION_THRESHOLD) ||
                         hasChangedBy(gyroData.z(), previousGyro.z(), MOTION_THRESHOLD);
        
        if (motionDetected) {
            lastMotionTime = std::chrono::steady_clock::now();
        } else {
            // If no significant motion, check if timeout has elapsed
            auto currentTime = std::chrono::steady_clock::now();
            auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(
                currentTime - lastMotionTime).count();
            
            std::cout << " | No motion for " << elapsedSeconds << "s";
            if (elapsedSeconds >= SHUTDOWN_TIMEOUT) {
                std::cout << "\nNo significant motion detected for " << SHUTDOWN_TIMEOUT 
                          << " seconds. Shutting down...\n";
                std::system("sudo shutdown -h now");
                running = false; // Flag for going into low power mode
                break;
            }
        }
    } else {
        firstReading = false;
        lastMotionTime = std::chrono::steady_clock::now();
    }
    previousGyro = gyroData;
    
    // Display orientation (Euler angles)
    std::cout << "    Orientation: ";
    std::cout << "X=" << std::fixed << std::setprecision(2) << euler.x() << "° ";
    std::cout << "Y=" << std::fixed << std::setprecision(2) << euler.y() << "° ";
    std::cout << "Z=" << std::fixed << std::setprecision(2) << euler.z() << "°";
    
    // Read and display temperature (optional)
    // int8_t temp = bno.getTemp();
    // std::cout << " | Temp=" << (int)temp << "°C";
    
    // Head Tilt Calculation:
    // With the sensor mounted so that the z-axis points upward (toward the sky)
    // and the y-axis points forward from the glasses, a head tilt downward will yield
    // a change in the accelerometer's y and z readings.
    // The tilt angle is computed as:
    //     angle = atan2(accelData.y, -accelData.z)
    // A positive angle indicates that the user is looking downward.
    double headTiltRad = atan2(accelData.y(), -accelData.z());
    double headTiltDeg = headTiltRad * 180.0 / M_PI;
    std::cout << " | Head Tilt: " << std::fixed << std::setprecision(2) << headTiltDeg << "°";
    
    std::cout << std::flush;  // Flush output immediately
    
    // Delay for 100ms (adjustable for your sampling rate)
    gpioDelay(100000);  // Delay in microseconds
}

std::cout << "\nExiting...\n";
return 0;
