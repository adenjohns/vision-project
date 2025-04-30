#include "ArducamTOFCamera.hpp"
#include <chrono>
#include <fstream>
#include <iostream>

#include <unistd.h> // IMU libraries
#include <iomanip>
#include <csignal>
#include <pigpio.h>
#include "RPI_BNO055.h"
#include <cmath>
#include <cstdlib>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Dense>
#include <vector> 
#include <algorithm> 
#include <thread>


// Flag to control IMU program execution
volatile bool running = true;

using namespace Arducam;

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;
using std::sort;

// MAX_DISTANCE value modifiable  is 2 or 4
#define MAX_DISTANCE 4000

cv::Rect seletRect(0, 0, 0, 0);
cv::Rect followRect(0, 0, 0, 0);
int max_width = 240;
int max_height = 180;
int max_range = 0;
int confidence_value = 60; // original value of 30

void on_confidence_changed(int pos, void *userdata)
{
    //
}

// Function to check if two (IMU) values differ by at least threshold (will use this for termination)
bool hasChangedBy(float current, float previous, float threshold) {
    return std::abs(current - previous) >= threshold;
}

// Signal handler for Ctrl+C (IMU CODE INTEGRATION)
void signalHandler(int signum) {
    std::cout << "\nInterrupt signal (" << signum << ") received.\n";
    running = false;
}

void display_fps(void)
{
    using std::chrono::high_resolution_clock;
    using namespace std::literals;
    static int count = 0;
    static auto time_beg = high_resolution_clock::now();
    auto time_end = high_resolution_clock::now();
    ++count;
    auto duration_ms = (time_end - time_beg) / 1ms;
    if (duration_ms >= 1000)
    {
        std::cout << "fps:" << count << std::endl;
        count = 0;
        time_beg = time_end;
    }
}

void save_image(float *image, int width, int height)
{
    using namespace std::literals;
    // filename = "depth_$width$_$height$_f32_$time.raw"
    auto now = std::chrono::system_clock::now().time_since_epoch() / 1ms;
    std::string filename =
        "depth_" + std::to_string(width) + "_" + std::to_string(height) + "_f32_" + std::to_string(now) + ".raw";
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<char *>(image), width * height * sizeof(float));
    file.close();
}

cv::Mat matRotateClockWise180(cv::Mat src)
{
    if (src.empty())
    {
        std::cerr << "RorateMat src is empty!";
    }

    flip(src, src, 0);
    flip(src, src, 1);
    return src;
}

void getPreview(cv::Mat preview_ptr, cv::Mat amplitude_image_ptr)
{
    auto len = preview_ptr.rows * preview_ptr.cols;
    for (int line = 0; line < preview_ptr.rows; line++)
    {
        for (int col = 0; col < preview_ptr.cols; col++)
        {
            if (amplitude_image_ptr.at<float>(line, col) < confidence_value)
                preview_ptr.at<uint8_t>(line, col) = 255;
        }
    }
}

void getPreviewRGB(cv::Mat preview_ptr, cv::Mat amplitude_image_ptr)
{
    preview_ptr.setTo(cv::Scalar(0, 0, 0), amplitude_image_ptr < confidence_value);
    // cv::GaussianBlur(preview_ptr, preview_ptr, cv::Size(7, 7), 0);
}

void onMouse(int event, int x, int y, int flags, void *param)
{
    if (x < 4 || x > (max_width - 4) || y < 4 || y > (max_height - 4))
        return;
    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:

        break;

    case cv::EVENT_LBUTTONUP:
        seletRect.x = x - 4 ? x - 4 : 0;
        seletRect.y = y - 4 ? y - 4 : 0;
        seletRect.width = 8;
        seletRect.height = 8;
        break;
    default:
        followRect.x = x - 4 ? x - 4 : 0;
        followRect.y = y - 4 ? y - 4 : 0;
        followRect.width = 8;
        followRect.height = 8;
        break;
    }
}

struct Gap {
    int start;                                                                              // the starting value of the gap
    int end;                                                                                // the ending value of the gap
    int length() const {return end - start + 1;}                                            // function that returns the length of the gap
};

bool compareGapLength(const Gap& a, const Gap& b)
{
    return a.length() > b.length();                                                         // longer gaps first
}

/**
 * @brief Find largest gap(s). 
 *
 * Finds the largest traversable gap by counting the largest number of consecutive
 * numbers (indices) as well as their starting and ending indices. The gaps in data_indices (breaks in consecutive numbers) occurs from
 * reset_closest_points() which sets the points greater than the threshold to 0, and returns the indices
 * of the rest of the non-zero values in the array.
 *
 * @param data_indices : A 1D array of numbers that are the indices of all the furthest data points for the find the gap algorithm.
 * @param topN : The number of gap vectors to record.
 * @return allGaps : A vector of all the gaps.
        start : The start of the gaps.
        end : The end of the gaps. 
        length : The total length of a gap.
 */
std::vector<Gap> find_largest_gaps(VectorXd data_indices, int topN = 2)
{ 
    vector<int> data_vec(data_indices.data(), data_indices.data() + data_indices.size());    // copies Eigen vector to normal vector 
    vector<Gap> allGaps;                                                                     // place to store all the existing gaps
    
    int start = data_vec[0];
    for (int i = 1; i < data_vec.size(); ++i)                                                // iterate through the indices of data 
    { 
        if (data_vec[i] != data_vec[i - 1] + 1)                                              // if the current index is not consecutive with the previous one, the gap ended
        {
            if (start != data_vec[i - 1])                                                    // if start and previous index are not the same, save the gap
            {
                allGaps.push_back({start, data_vec[i - 1]});                                 // creates a gap from the start to just before the current number 
            }
            start = data_vec[i];                                                             // start tracking next potential gap
        }
    }
    
    if (start != data_vec.back())                                                            // Track potential last gap in range
    {
        allGaps.push_back({start, data_vec.back()});
    }
    
    sort(allGaps.begin(), allGaps.end(), compareGapLength);                                  // Sorts gaps by length, from laragest to smallest
    
    if (allGaps.size() > static_cast<size_t>(topN))                                          // Shrink the vector to keep only the top N largest gaps
    {
        allGaps.resize(topN);
    }
    
    return allGaps;
}

/**
 * @brief Resets closes points and returns indices of rest. 
 *
 * Sets data less than threshold to zero and returns the indices of all data points larger than zero.
 *
 * @param data : The 1D array that needs to be parsed.
 * @param threshold : An experimental value that is the limit to how close an object should be distance wise.
 * @return void : 1D array of all the indices of the data points larger than zero is copied into dataIndices.
 */
void reset_closest_points(VectorXd& data, int threshold, VectorXd& dataIndices)
{
    int count = 0;

    // Iterate through the vector and apply thresholding
    for (int i = 0; i < data.size(); ++i)
    {
        if (data(i) < threshold)
        {
            data(i) = 0; // Set values greater than the threshold to zero
        }
        if (data(i) > 0)
        {
            // Store indices of values that are still nonzero after thresholding
            dataIndices(count++) = i; // Store index
        }
    }
}

/**
 * @brief Returns the average distances of select rows per column. 
 *
 * Takes a 2D array and first slices it based on the necessary rows needed given by
 * row_start and row_end, and then finds the average value of all the columns for n rows between the
 * given starting and ending rows.
 *
 * @param array : The 2D array of distance data.
 * @param row_start : An experimental value of the beginning of the rows that need to be parsed.
 * @param row_end : An experimental value of the end of the rows that need to be parsed.
 * @return void : A 1D array of the min distance out of each col between row_start and row_end is copied into col_avg_val.
 */
void avg_data_rows(MatrixXd array, int row_start, int row_end, int window_size, VectorXd& col_avg_val)
{
    int num_rows = row_end - row_start;
    int num_cols = array.cols();
    
    // Extract the submatrix (slicing rows from the given start row to the end row)
    MatrixXd sliced_arr = array.block(row_start, 0, num_rows, num_cols);
    
    // Compute average along each column    
    col_avg_val = sliced_arr.colwise().mean();
    
    // MOVING AVERAGE IMPLEMENTATION (may be too slow for our needs fps jumped to 17)
    //// New matrix to store values of averages 
    //MatrixXd smoothed = MatrixXd::Zero(num_rows - window_size + 1, num_cols); 
    //for (int i = 0; i <= num_rows - window_size; ++i){
        //smoothed.row(i) = sliced_arr.block(i, 0, window_size, num_cols).colwise().mean();
    //}
    //// Compute average along each column    
    //col_avg_val = smoothed.colwise().mean();
}

/**
 * @brief Convert to an Eigen matrix.
 *
 * Converts the depth matrix to the Eigen matrix.
 *
 * @param depth_mat OpenCv's 2D depth data.
 * @return void. A 2D matrix of depth data converted to an Eigen matrix.
 */
void convertMatToEigen(cv::Mat& depth_mat, MatrixXd& depth_matrix)
{
    for (int i = 0; i < depth_mat.rows; ++i)
    {
        for (int j = 0; j < depth_mat.cols; ++j)
        {
            depth_matrix(i, j) = depth_mat.at<float>(i, j);
        }
    }
}


    // ###########################################################################################################
    // AUDIO FEEDBACK START
    // ###########################################################################################################

// Audio parameters
const int AUDIO_PIN_LEFT = 18;  // GPIO pin for left speaker
const int AUDIO_PIN_RIGHT = 19; // GPIO pin for right speaker
const int AUDIO_FREQ = 1000;    // Base frequency in Hz
const int AUDIO_RANGE = 1000;   // PWM range (0-1000)
const int BASE_VOLUME = 200;    // Base volume level (out of 1000)




class AudioFeedback {
private:
    int leftPin;
    int rightPin;
    bool initialized;

public:
    AudioFeedback(int leftPin = AUDIO_PIN_LEFT, int rightPin = AUDIO_PIN_RIGHT)
        : leftPin(leftPin), rightPin(rightPin), initialized(false) {}

    bool initialize() {
        if (gpioInitialise() < 0) {
            std::cerr << "Failed to initialize pigpio" << std::endl;
            return false;
        }

        // Set pins as output
        gpioSetMode(leftPin, PI_OUTPUT);
        gpioSetMode(rightPin, PI_OUTPUT);

        // Set PWM frequency
        gpioSetPWMfrequency(leftPin, AUDIO_FREQ);
        gpioSetPWMfrequency(rightPin, AUDIO_FREQ);

        // Set PWM range
        gpioSetPWMrange(leftPin, AUDIO_RANGE);
        gpioSetPWMrange(rightPin, AUDIO_RANGE);

        initialized = true;
        return true;
    }

    void updateAudio(const std::vector<Gap>& gaps, int imageWidth) {
        if (!initialized) {
            std::cerr << "AudioFeedback not initialized" << std::endl;
            return;
        }

        // If no gaps found, play base volume on both speakers
        if (gaps.empty()) {
            gpioPWM(leftPin, BASE_VOLUME);
            gpioPWM(rightPin, BASE_VOLUME);
            return;
        }

        // Find the largest gap (most likely path)
        const Gap& mainGap = gaps[0];
        
        // Calculate the center of the gap
        float gapCenter = (mainGap.start + mainGap.end) / 2.0f;
        
        // Calculate the center of the image
        float imageCenter = imageWidth / 2.0f;
        
        // Calculate distance from center (normalized to [0, 1])
        float distanceFromCenter = std::abs(gapCenter - imageCenter) / imageCenter;
        
        // Calculate volume based on distance from center
        // The farther from center, the louder the volume
        int maxVolume = AUDIO_RANGE;
        int volume = BASE_VOLUME + static_cast<int>(distanceFromCenter * (maxVolume - BASE_VOLUME));
        
        // Set volumes based on which side of center the gap is
        if (gapCenter < imageCenter) {
            // Gap is on the left side
            gpioPWM(leftPin, volume);
            gpioPWM(rightPin, BASE_VOLUME);
        } else {
            // Gap is on the right side
            gpioPWM(leftPin, BASE_VOLUME);
            gpioPWM(rightPin, volume);
        }
    }

    void stop() {
        if (initialized) {
            gpioPWM(leftPin, 0);
            gpioPWM(rightPin, 0);
            gpioTerminate();
            initialized = false;
        }
    }

    ~AudioFeedback() {
        stop();
    }
};



    
    // ###########################################################################################################
    // AUDIO FEEDBACK END
    // ###########################################################################################################


int main()
{
    gpioInitialise();
    // Initialize audio feedback
    AudioFeedback audio;
    if (!audio.initialize()) {
        std::cerr << "Failed to initialize audio feedback" << std::endl;
        return -1;
    }
    
    // ###########################################################################################################
    // IMU SETUP 
    // ###########################################################################################################
    
    // Register signal handler
    signal(SIGINT, signalHandler);
    std::cout << "Activating BNO055 Sensor using pigpio\n";
    gpioInitialise();  // Initialize pigpio early

    // This whole section is just to check if the i2c is working (and device is on correct bus)
    int handle = i2cOpen(4, 0x28, 0);
    if (handle < 0) {
        std::cerr << "Direct i2cOpen test failed with error: " << handle << std::endl;
    } else {
        std::cout << "Direct i2cOpen test succeeded with handle: " << handle << std::endl;
        i2cClose(handle);
    }

    // Create sensor instance with bus 1 and address 0x28
    RPI_BNO055 bno(-1, BNO055_ADDRESS_A, 4);  // Try using bus 1 instead (should be the case)
    
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
    
    // ###########################################################################################################
    // IMU SETUP END
    // ###########################################################################################################
    


    // ###########################################################################################################
    // ARDUCAM SETUP 
    // ###########################################################################################################

    ArducamTOFCamera tof;
    ArducamFrameBuffer *frame;
    if (tof.open(Connection::CSI, 0))
    {
        std::cerr << "Failed to open camera" << std::endl;
        return -1;
    }

    if (tof.start(FrameType::DEPTH_FRAME))
    {
        std::cerr << "Failed to start camera" << std::endl;
        return -1;
    }
    //  Modify the range also to modify the MAX_DISTANCE
    tof.setControl(Control::RANGE, MAX_DISTANCE);
    tof.getControl(Control::RANGE, &max_range);
    auto info = tof.getCameraInfo();
    std::cout << "open camera with (" << info.width << "x" << info.height << ")" << std::endl;

    uint8_t *preview_ptr = new uint8_t[info.width * info.height * 2];
    cv::namedWindow("preview", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("preview", onMouse);
    
    // ###########################################################################################################
    // ARDUCAM SETUP END
    // ###########################################################################################################

    while (running)
    {
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
        std::cout << "    Orientation: ";
        std::cout << "X=" << std::fixed << std::setprecision(2) << euler.x() << "° ";
        std::cout << "Y=" << std::fixed << std::setprecision(2) << euler.y() << "° ";
        std::cout << "Z=" << std::fixed << std::setprecision(2) << euler.z() << "°";
        
        // Get temp (idk if we need this but just in case)
        // int8_t temp = bno.getTemp();
        // std::cout << " | Temp=" << (int)temp << "°C";
        
        // Head Tilt Calculation:
        // With the sensor mounted so that the z-axis points upward (toward the sky)
        // and the y-axis points forward from the glasses, a head tilt downward will yield
        // a change in the accelerometer's y and z readings.
        // A positive angle indicates that the user is looking downward.
        double headTiltRad = atan2(accelData.y(), -accelData.z()); // Tilt angle computation
        double headTiltDeg = headTiltRad * 180.0 / M_PI;
        std::cout << " | Head Tilt: " << std::fixed << std::setprecision(2) << headTiltDeg << "°";

        std::cout << std::flush; // Ensure output is displayed
        
        // Wait a bit - using gpioDelay for more precise timing
        gpioDelay(100000); // 100ms (can be changed depending on desired sampling rate)




        Arducam::FrameFormat format;
        frame = tof.requestFrame(200);
        if (frame == nullptr)
        {
            continue;
        }
        frame->getFormat(FrameType::DEPTH_FRAME, format);
        // std::cout << "frame: (" << format.width << "x" << format.height << ")" << std::endl;
        max_height = format.height;
        max_width = format.width;

        float *depth_ptr = (float *)frame->getData(FrameType::DEPTH_FRAME);
        float *confidence_ptr = (float *)frame->getData(FrameType::CONFIDENCE_FRAME);
        // getPreview(preview_ptr, depth_ptr, confidence_ptr);

        cv::Mat result_frame(format.height, format.width, CV_8U, preview_ptr);
        cv::Mat depth_frame(format.height, format.width, CV_32F, depth_ptr);
        cv::Mat confidence_frame(format.height, format.width, CV_32F, confidence_ptr);

        // depth_frame = matRotateClockWise180(depth_frame);
        // result_frame = matRotateClockWise180(result_frame);
        // confidence_frame = matRotateClockWise180(confidence_frame);
        depth_frame.convertTo(result_frame, CV_8U, 255.0 / 7000, 0);

        cv::applyColorMap(result_frame, result_frame, cv::COLORMAP_RAINBOW);
        getPreviewRGB(result_frame, confidence_frame);

        confidence_frame.convertTo(confidence_frame, CV_8U, 255.0 / 1024, 0);

        cv::imshow("confidence", confidence_frame);

        cv::rectangle(result_frame, seletRect, cv::Scalar(0, 0, 0), 2);
        cv::rectangle(result_frame, followRect, cv::Scalar(255, 255, 255), 1);

        // std::cout << "select Rect distance: " << cv::mean(depth_frame(seletRect)).val[0] << std::endl;

        cv::imshow("preview", result_frame);
        
        // #######################################################################################
        // PATH PLANNING CODE
        // #######################################################################################
        
        MatrixXd depth_matrix(depth_frame.rows, depth_frame.cols);                      
        convertMatToEigen(depth_frame, depth_matrix);

        int threshold = 2999; 
        int base_row_start = 60;   
        int base_row_end = 120;    
        int window_size = 5;

        double scale_factor = 0.5;
        int offset = static_cast<int>(scale_factor * headTiltDeg);
        
        int row_start = std::max(0, base_row_start + offset);
        int row_end = std::min(max_height - 1, base_row_end + offset);

        VectorXd col_max_val(depth_matrix.cols());                                     
        avg_data_rows(depth_matrix, row_start, row_end, window_size, col_max_val);

        VectorXd data_indices(col_max_val.size());                                     
        reset_closest_points(col_max_val, threshold, data_indices);

        auto gaps = find_largest_gaps(data_indices, 2);
        
        // Update audio feedback based on the gaps
        audio.updateAudio(gaps, format.width);
        
        if (gaps.empty()) 
        { 
            cout << "No gaps found.\n"; 
        }
        else 
        {
            cout << "Gaps: ";
            for (size_t i = 0; i < gaps.size(); ++i) 
            {
                cout << "[Gap " << i + 1 
                     << " -> Start: " << gaps[i].start << " End: " << gaps[i].end << ""
                     << ", Len: " << gaps[i].length() << "] ";
            }
            cout << "\n";
        }

        // #######################################################################################
        // #######################################################################################

        auto key = cv::waitKey(1);
        if (key == 27 || key == 'q')
        {
            break;
        }
        else if (key == 's')
        {
            save_image(depth_ptr, format.width, format.height);
        }
        display_fps();
        tof.releaseFrame(frame);
    }

    // Clean up
    if (tof.stop())
    {
        return -1;
    }

    if (tof.close())
    {
        return -1;
    }
    
    // Stop audio feedback
    audio.stop();
    
    std::cout << "\nExiting...\n"; 
    return 0;
}
