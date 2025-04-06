#include "ArducamTOFCamera.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Dense>

using namespace Arducam;

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

// MAX_DISTANCE value modifiable  is 2 or 4
#define MAX_DISTANCE 4000

cv::Rect seletRect(0, 0, 0, 0);
cv::Rect followRect(0, 0, 0, 0);
int max_width = 240;
int max_height = 180;
int max_range = 0;
int confidence_value = 30;

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

/**
 * @brief std::tuple<int, int, int> find_largest_gap(VectorXi data_indices)
 *
 * Finds the largest traversable gap by counting the largest number of consecutive
 * numbers (indices) as well as their starting and ending indeces. The gaps in data_indices occurs from
 * reset_closest_points() which sets the points greater than the threshold to 0, and returns the indices
 * of the rest of the non-zero values in the array.
 *
 * @param data_indices a 1D array of numbers that are the indices of all the furthest data points for the find the gap algorithm.
 * @return max_length, max_start_idx, and max_end_idx.
        The max length is the length of the widest gap that exists.
        The max_start_idx is the starting index of the largest gap.
        The max_end_idx isthe ending index of the largest gap.
 */
std::tuple<int, int, int> find_largest_gap(VectorXd data_indices)
{
    int max_length = 1;
    int current_length = 1;

    int start_idx = 0;
    int max_start_idx = 0;
    int max_end_idx = 0;

    for (int i = 1; i < data_indices.size(); ++i)
    {
        if (data_indices(i) - data_indices(i - 1) == 1)
        { // Checking for increasing order
            current_length += 1;
        }
        else
        {
            if (current_length > max_length)
            {                                // If current length is greater than max length
                max_length = current_length; // Set max length to current length
                max_start_idx = start_idx;
                max_end_idx = i - 1;
            }

            current_length = 1; // Reset count
            start_idx = i;      // Update start index
        }
    }

    // Final check of the last value
    if (current_length > max_length)
    {
        max_length = current_length;
        max_start_idx = start_idx;
        max_end_idx = data_indices.size() - 1;
    }

    return std::make_tuple(max_length, max_start_idx, max_end_idx);
}

/**
 * @brief min_data_rows(MatrixXd array, int row_start, int row_end)
 *
 * Sets data great than threshold to zero and returns the indices of all data points larger than zero.
 *
 * @param data The 1D array that needs to be parsed.
 * @param threshold An experimental value that is the limit to how close an object should be distance wise.
 * @return void. 1D array of all the indices of the data points larger than zero.
 */
void reset_closest_points(VectorXd data, int threshold, VectorXd dataIndices)
{
    int count = 0;

    // Iterate through the vector and apply thresholding
    for (int i = 0; i < data.size(); ++i)
    {
        if (data(i) > threshold)
        {
            data(i) = 0; // Set values greater than the threshold to zero
        }
        else if (data(i) > 0)
        {
            // Store indices of values that are still nonzero after thresholding
            dataIndices(count++) = i; // Store index
        }
    }

    // Resize output vector to match the actual number of nonzero elements
    dataIndices.head(count);
}

/**
 * @brief min_data_rows(MatrixXd array, int row_start, int row_end)
 *
 * Takes a 2D array and first slices it based on the necessary rows needed given by
 * row_start and row_end, and then finds the min value of all the columns for n rows between the
 * given starting and ending rows.
 *
 * @param array The 2D array of distance data.
 * @param row_start An experimental value of the beginning of the rows that need to be parsed.
 * @param row_end An experimental value of the end of the rows that need to be parsed.
 * @return void. A 1D array of the min distance out of each col between row_start and row_end.
 */
void min_data_rows(MatrixXd array, int row_start, int row_end, VectorXd col_max_val)
{
    // Extract the submatrix (slicing rows from the given start row to the end row)
    MatrixXd sliced_arr = array.block(row_start, 0, row_end - row_start, array.cols());

    // Compute max along each column
    col_max_val = sliced_arr.colwise().maxCoeff();
}

/**
 * @brief MatrixXf convertMatToEigen(cv::Mat depth_mat)
 *
 * Converts the depth matrix to the Eigen matrix.
 *
 * @param depth_mat OpenCv's 2D depth data.
 * @return void. A 2D array of depth data converted to an Eigen matrix.
 */
void convertMatToEigen(cv::Mat depth_mat, MatrixXd depth_matrix)
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
        getPreview(preview_ptr, depth_ptr, confidence_ptr);

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

        // PATH PLANNING CODE
        // #######################################################################################
        MatrixXd depth_matrix(depth_frame.rows, depth_frame.cols); // Matrix output for convertMatToEigen(). A 2D array of depth data converted to an Eigen matrix.
        convertMatToEigen(depth_frame, depth_matrix);
        
        std::cout << "Matrix:" << std::end1 << depth_matrix << std::end1;

        int threshold = 2999; // EXPERIMENTAL VALUE, depth values of object at closest limit to user
        int row_start = 60;   // EXPERIMANTAL VALUE, depth value to first row from frame to parse
        int row_end = 120;    // EXPERIMENTAL VALUE, depth value of last row from frame to parse

        VectorXd col_max_val(depth_matrix.cols()); // Vector output for min_data_rows(). A 1D array of the min distance out of each col between row_start and row_end.
        min_data_rows(depth_matrix, row_start, row_end, col_max_val);

        VectorXd data_indices(col_max_val.size()); // Vector output for reset_closet_points(). 1D array of all the indices of the data points larger than zero.
        reset_closest_points(col_max_val, threshold, data_indices);

        auto [max_length, max_start_idx, max_end_idx] = find_largest_gap(data_indices);

        cout << "Max Length: " << max_length << endl;
        cout << "Max Start Index: " << max_start_idx << endl;
        cout << "Max End Index: " << max_end_idx << endl;

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
