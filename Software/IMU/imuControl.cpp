#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <csignal>
#include <pigpio.h>
#include "RPI_BNO055.h"
#include <cmath>
#include <cstdlib>
#include <chrono>

// Flag to control program execution
volatile bool running = true;

// Function to check if two values differ by at least threshold (will use this for termination)
bool hasChangedBy(float current, float previous, float threshold) {
    return std::abs(current - previous) >= threshold;
}

// Signal handler for Ctrl+C
void signalHandler(int signum) {
    std::cout << "\nInterrupt signal (" << signum << ") received.\n";
    running = false;
}

int main() {
    // Register signal handler
    signal(SIGINT, signalHandler);

    std::cout << "BNO055 Sensor Test using pigpio\n";
    

    gpioInitialise();  // Initialize pigpio early

// This whole section is just to check if the i2c is working (and device is on correct bus)
    int handle = i2cOpen(3, 0x28, 0);
    if (handle < 0) {
        std::cerr << "Direct i2cOpen test failed with error: " << handle << std::endl;
    } else {
        std::cout << "Direct i2cOpen test succeeded with handle: " << handle << std::endl;
        i2cClose(handle);
    }
    
    // Create sensor instance with bus 1 and address 0x28
    RPI_BNO055 bno(-1, BNO055_ADDRESS_A, 1);  // Try using bus 1 instead (should be the case)
    
    // Initialize the sensor 
    if (!bno.begin()) {
        std::cerr << "Failed to initialize BNO055 sensor!\n";
        return -1;
    }
    
    std::cout << "BNO055 sensor initialized successfully!\n";
    
    // Set external crystal for better accuracy (if available)
    bno.setExtCrystalUse(true);
    
    // Get chip information
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
    
    // Variables for motion detection
    imu::Vector<3> previousGyro;
    bool firstReading = true;
    std::chrono::steady_clock::time_point lastMotionTime;
    const float MOTION_THRESHOLD = 5.0f;  // 5 degrees threshold for low power standby purposes
    const int SHUTDOWN_TIMEOUT = 30;      // 30 seconds timeout (both of these can be changed depending on needs)
    
    // Main loop
    while (running) {
        // Get calibration status
        uint8_t system, gyro, accel, mag;
        bno.getCalibration(&system, &gyro, &accel, &mag);
        
        // Display calibration status (0-3, where 3 is fully calibrated)
        std::cout << "\033[2K\r"; // Clear current line
        std::cout << "Calibration: Sys=" << (int)system << " Gyro=" << (int)gyro 
                  << " Accel=" << (int)accel << " Mag=" << (int)mag << " | ";
        
        // Read Euler angles (in degrees)
        imu::Vector<3> euler = bno.getVector(VECTOR_EULER);
        
        // Read acceleration values (in m/s^2)
        imu::Vector<3> accelData = bno.getVector(VECTOR_ACCELEROMETER);
        
        // Read gyroscope values (in deg/s)
        imu::Vector<3> gyroData = bno.getVector(VECTOR_GYROSCOPE);
        
        // Check for significant motion
        bool motionDetected = false;
        
        if (!firstReading) {
            motionDetected = hasChangedBy(gyroData.x(), previousGyro.x(), MOTION_THRESHOLD) ||
                             hasChangedBy(gyroData.y(), previousGyro.y(), MOTION_THRESHOLD) ||
                             hasChangedBy(gyroData.z(), previousGyro.z(), MOTION_THRESHOLD);
            
            if (motionDetected) {
                lastMotionTime = std::chrono::steady_clock::now();
            } else {
                // Check if timeout has elapsed
                auto currentTime = std::chrono::steady_clock::now();
                auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(
                    currentTime - lastMotionTime).count();
                
                // Display remaining time before shutdown
                std::cout << " | No motion for " << elapsedSeconds << "s";
                
                if (elapsedSeconds >= SHUTDOWN_TIMEOUT) {
                    std::cout << "\nNo significant motion detected for " << SHUTDOWN_TIMEOUT 
                              << " seconds. Shutting down...\n";
                    std::system("sudo shutdown -h now");
                    running = false;
                    break;
                }
            }
        } else {
            // Initialize for first iteration
            firstReading = false;
            lastMotionTime = std::chrono::steady_clock::now();
        }
        
        // Save values for next comparison
        previousGyro = gyroData;
        
        // Display orientation data (euler)
        std::cout << "Orientation: ";
        std::cout << "X=" << std::fixed << std::setprecision(2) << euler.x() << "° ";
        std::cout << "Y=" << std::fixed << std::setprecision(2) << euler.y() << "° ";
        std::cout << "Z=" << std::fixed << std::setprecision(2) << euler.z() << "°";
        
        // Get temp (idk if we need this but just in case)
        int8_t temp = bno.getTemp();
        std::cout << " | Temp=" << (int)temp << "°C";
        
        std::cout << std::flush; // Ensure output is displayed
        
        // Wait a bit - using gpioDelay for more precise timing
        gpioDelay(100000); // 100ms (can be changed depending on desired sampling rate)
    }
    
    std::cout << "\nExiting...\n";
    
    return 0;
} 
