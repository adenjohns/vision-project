#include "ArducamTOFCamera.hpp"
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <regex> 

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Dense>
#include <vector> 
#include <algorithm> 

#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>  // Add OpenCL header
#include <opencv2/optflow.hpp>  // Add optical flow header

const char* keys =
"{help h usage ? | | Usage examples: \n\t\t./object_detection_yolo.out --image=dog.jpg \n\t\t./object_detection_yolo.out --video=run_sm.mp4}"
"{image i        |<none>| input image   }"
"{video v       |<none>| input video   }"
"{device d         |0| input device index (for webcam) }"
;
using namespace cv;
using namespace dnn;
using namespace std;
using namespace std::chrono;
using namespace Arducam;

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;
using std::sort;

int max_width = 240;
int max_height = 180;
int max_range = 0;
int confidence_value = 60;

// MAX_DISTANCE value modifiable  is 2 or 4
#define MAX_DISTANCE 4000

cv::Rect seletRect(0, 0, 0, 0);
cv::Rect followRect(0, 0, 0, 0);


int getVidIdxFromSymlink(const std::string &symlinkPath) {
 
    namespace fs = std::filesystem; 
    try { 
        fs::path realPath = fs::read_symlink(symlinkPath);
        std::string filename = realPath.filename(); // e.g., "video1"
        std::smatch match; 
        if (std::regex_search(filename, match, std::regex("video(\\d+)"))) {
            return std::stoi(match[1]);
        }
    } catch (const fs::filesystem_error &e) {
        std::cerr << "Error resolving symlink: " << e.what() << std::endl;
    }
    return -1; // error
}

void on_confidence_changed(int pos, void *userdata)
{
    //
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

int main()
{
    int videoIndex = getVidIdxFromSymlink("/dev/video-arducam");
    if (videoIndex == -1) {
        std::cerr << "Failed to resolve video index for Arducam" << std::endl;
        return -1;
    } 
    
    ArducamTOFCamera tof;
    ArducamFrameBuffer *frame;
    if (tof.open(Connection::CSI, videoIndex))
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

    for (;;)
    {
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
        
        MatrixXd depth_matrix(depth_frame.rows, depth_frame.cols);                      // Matrix output for convertMatToEigen(). A 2D array of depth data converted to an Eigen matrix.
        convertMatToEigen(depth_frame, depth_matrix);

        int threshold = 800; // EXPERIMENTAL VALUE, depth values of object at closest limit to user
        int row_start = 60;   // EXPERIMANTAL VALUE, depth value to first row from frame to parse  
        int row_end = 120;    // EXPERIMENTAL VALUE, depth value of last row from frame to parse 
        int window_size = 5;

        VectorXd col_max_val(depth_matrix.cols());                                     // Vector output for min_data_rows(). A 1D array of the min distance out of each col between row_start and row_end.
        avg_data_rows(depth_matrix, row_start, row_end, window_size, col_max_val);

        VectorXd data_indices(col_max_val.size());                                     // Vector output for reset_closet_points(). 1D array of all the indices of the data points larger than zero.
        reset_closest_points(col_max_val, threshold, data_indices);
        
        auto gaps = find_largest_gaps(data_indices, 2);
        
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

    if (tof.stop())
    {
        return -1;
    }

    if (tof.close())
    {
        return -1;
    }

    return 0;
}
