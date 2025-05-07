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

// Initialize the parameters
float confThreshold = 0.4; // Confidence threshold
float nmsThreshold = 0.3;  // Non-maximum suppression threshold
float motionThreshold = 0.5;  // Motion detection threshold
int inpWidth = 128;  // Reduced from 128 for faster processing
int inpHeight = 96; // Reduced from 96 for faster processing
int minFrameSkip = 2;  // Minimum frames to skip
int maxFrameSkip = 4;  // Maximum frames to skip
int currentFrameSkip = minFrameSkip;  // Dynamic frame skip
int frameCounter = 0;
bool skipFrame = false;
vector<string> classes;

// Add these global variables after the existing ones
Mat prevFrame, prevGray;
bool firstFrame = true;

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, Mat& oldframe, const vector<Mat>& out);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);

Size letterbox(Mat &frame, int new_shape=416) {
    // Mat frame = inputframe.clone();
    int width = frame.cols;
    int height = frame.rows;
    // cout<<width << " hain? " << height<< endl;
    float ratio = float(new_shape)/max(width, height);
    float ratiow = ratio;
    float ratioh = ratio;
    // cout<<width*ratio << " " << height*ratio << endl;
    int new_unpad0 = int(round(width*ratio));
    int new_unpad1 = int(round(height * ratio));
    int dw = ((new_shape - new_unpad0) % 32 )/2;
    int dh = ((new_shape - new_unpad1) % 32 )/2;
    int top = int(round(dh - 0.1));
    int bottom = int(round(dh+0.1));
    int left = int(round(dw - 0.1));
    int right = int(round(dw + 0.1));

    // cout<<" ---- "<< new_unpad0 <<  " " << new_unpad1<<endl;
    cv::resize(frame, frame, cv::Size(new_unpad0, new_unpad1), 0, 0, 1); //CV_INTER_LINEAR = 1
    Scalar value(127.5, 127.5, 127.5);
    cv::copyMakeBorder(frame, frame, top, bottom, left, right, cv::BORDER_CONSTANT, value);
    return frame.size();
    
}

void scale_coords(Size img1, Size img0, Rect &box) {
    int img00 = img0.height;
    int img01 = img0.width;
    int img10 = img1.height;
    int img11 = img1.width;
    //  cout<<"im1 ki shape " << img10 << " " << img11 << endl;
    //  cout<<"im0 ki shape " << img00 << " " << img01 << endl;

    int max0  = max(img00, img01);
    int max1 = max(img10, img11);
    double gain = double(max1)/double(max0);
    // cout<<"Gain = " << gain << " " << max0 << " " << max1 << endl;

    box.x = box.x - (img11 - (img01*gain))/2;
    box.width = box.width - (img11 - (img01*gain))/2;
    box.y = box.y - (img10 - (img00*gain))/2;
    box.height = box.height - (img10 - (img00*gain))/2;

    // cout<<"subtractions = " << box.x << " " << box.y << " " << box.width << " " << box.height << endl;

    box.x = box.x/gain;
    box.y = box.y/gain;
    box.width = box.width/gain;
    box.height = box.height/gain;

    if (box.x < 0)
        box.x = 0;
    if (box.y < 0)
        box.y = 0;
    if (box.width < 0)
        box.width = 0;
    if (box.height < 0)
        box.height = 0;
    
    // cout<<"after gain = " << box.x << " " << box.y << " " << box.width << " " << box.height << endl;

}

// Add this function before main()
bool isKeyFrame(const Mat& currentFrame) {
    if (firstFrame) {
        firstFrame = false;
        currentFrame.copyTo(prevFrame);
        cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);
        return true;
    }

    Mat currentGray;
    cvtColor(currentFrame, currentGray, COLOR_BGR2GRAY);
    
    // Calculate absolute difference between frames
    Mat diff;
    absdiff(prevGray, currentGray, diff);
    
    // Calculate mean difference
    Scalar meanDiff = mean(diff);
    float motionScore = (meanDiff[0] + meanDiff[1] + meanDiff[2]) / 3.0;
    
    // Adjust frame skip based on motion intensity
    if (motionScore > motionThreshold) {
        currentFrameSkip = minFrameSkip;  // Process more frames when motion is high
    } else {
        currentFrameSkip = maxFrameSkip;  // Skip more frames when motion is low
    }
    
    // Update previous frame
    currentFrame.copyTo(prevFrame);
    currentGray.copyTo(prevGray);
    
    return motionScore > motionThreshold;
}

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

int main(int argc, char** argv)
{
	// ###########################################################################################################
    // ARDUCAM SETUP 
    // ###########################################################################################################
	
	int videoIndex = getVidIdxFromSymlink("/dev/video-arducam");
    if (videoIndex == -1) {
        std::cerr << "Failed to resolve video index for Arducam" << std::endl;
        return -1;
    } 
    
    ArducamTOFCamera tof;
    ArducamFrameBuffer *frame0;
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
	
	// ###########################################################################################################
    // ARDUCAM SETUP END
    // ###########################################################################################################
	
	// ###########################################################################################################
    // WEBCAM SETUP 
    // ###########################################################################################################
    
    // Check OpenCL availability
    bool haveOpenCL = cv::ocl::haveOpenCL();
    bool useOpenCL = false;
    
    if (haveOpenCL) {
        cv::ocl::Context context;
        if (context.create(cv::ocl::Device::TYPE_GPU)) {
            cout << "OpenCL GPU context created successfully" << endl;
            cv::ocl::Device device = context.device(0);
            cout << "OpenCL device: " << device.name() << endl;
            useOpenCL = true;
        } else {
            cout << "OpenCL GPU context creation failed" << endl;
        }
    } else {
        cout << "OpenCL not available" << endl;
    }

    // Enable OpenCL if available
    cv::ocl::setUseOpenCL(useOpenCL);
    
    // Pre-allocate memory buffers
    cv::Mat frame_buffer(inpHeight, inpWidth, CV_8UC3);
    cv::Mat blob_buffer(1, inpWidth * inpHeight * 3, CV_32F);
    cv::Mat display_buffer(inpHeight, inpWidth, CV_8UC3);
    
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    // Load names of classes
    string classesFile = "../data/coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);
    
    // Give the configuration and weight files for the model
    String modelConfiguration = "/home/vision-rpi/Desktop/Tiny-Yolov3-OpenCV-Cpp/cfg/yolov3-tiny.cfg";
    // String modelWeights = "/home/omair/workspace/CNN/hazen.ai/ultralytics/yolov3/weights/latest_retail.weights";
    String modelWeights = "/home/vision-rpi/Desktop/Tiny-Yolov3-OpenCV-Cpp/weights/yolov3-tiny.weights";

    // Load the network with optimized settings
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    
    // Try to use OpenCL backend if available
    if (useOpenCL) {
        cout << "Attempting to use OpenCL backend..." << endl;
        try {
            net.setPreferableBackend(DNN_BACKEND_OPENCV);
            net.setPreferableTarget(DNN_TARGET_OPENCL);
            cout << "Successfully set OpenCL backend and target" << endl;
        } catch (const cv::Exception& e) {
            cout << "Failed to set OpenCL backend/target: " << e.what() << endl;
            cout << "Falling back to CPU" << endl;
            net.setPreferableBackend(DNN_BACKEND_OPENCV);
            net.setPreferableTarget(DNN_TARGET_CPU);
            useOpenCL = false;
        }
    } else {
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }
    
    // Open a video file or an image file or a camera stream.
    string str, outputFile;
    VideoCapture cap;
    VideoWriter video;
    Mat frame, blob;
    
    try {
        
        outputFile = "yolo_out_cpp.avi";
        if (parser.has("image"))
        {
            // Open the image file
            str = parser.get<String>("image");
            ifstream ifile(str);
            if (!ifile) throw("error");
            cap.open(str);
            str.replace(str.end()-4, str.end(), "_yolo_out_cpp.jpg");
            outputFile = str;
        }
        else if (parser.has("video"))
        {
            // Open the video file
            str = parser.get<String>("video");
            ifstream ifile(str);
            if (!ifile) throw("error");
            cap.open(str);
            str.replace(str.end()-4, str.end(), "_yolo_out_cpp.avi");
            outputFile = str;
        }
        // Open the webcaom
        else cap.open(parser.get<int>("device"));
        
    }
    catch(...) {
        cout << "Could not open the input image/video stream" << endl;
        return 0;
    }
    
    // Get the video writer initialized to save the output video
    if (!parser.has("image")) {
        video.open(outputFile, VideoWriter::fourcc('M','J','P','G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
    }
    
    // Create a window
    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);
    double alltimes = 0.0;
    double count = 0;
    Mat oldframe;
    Size sz;
    
    // ###########################################################################################################
    // WEBCAM SETUP END
    // ###########################################################################################################

    // Process frames.
    while (waitKey(1) < 0)
    {
		
		Arducam::FrameFormat formatFrame;
        frame0 = tof.requestFrame(200);
        if (frame0 == nullptr)
        {
            continue;
        }
        frame0->getFormat(FrameType::DEPTH_FRAME, formatFrame);
        // std::cout << "frame: (" << formatFrame.width << "x" << formatFrame.height << ")" << std::endl;
        max_height = formatFrame.height;
        max_width = formatFrame.width;

        float *depth_ptr = (float *)frame0->getData(FrameType::DEPTH_FRAME);
        float *confidence_ptr = (float *)frame0->getData(FrameType::CONFIDENCE_FRAME);
        // getPreview(preview_ptr, depth_ptr, confidence_ptr);

        cv::Mat result_frame(formatFrame.height, formatFrame.width, CV_8U, preview_ptr);
        cv::Mat depth_frame(formatFrame.height, formatFrame.width, CV_32F, depth_ptr);
        cv::Mat confidence_frame(formatFrame.height, formatFrame.width, CV_32F, confidence_ptr);

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
            save_image(depth_ptr, formatFrame.width, formatFrame.height);
        }
        display_fps();
        tof.releaseFrame(frame0);
		
		
		
        // Start timing for frame capture
        auto capture_start = high_resolution_clock::now();
        
        // get frame from the video
        cap >> frame;
        
        if (frame.empty()) {
            cout << "Done processing !!!" << endl;
            cout << "Output file is stored as " << outputFile << endl;
            waitKey(3000);
            break;
        }

        auto capture_end = high_resolution_clock::now();
        auto capture_time = duration_cast<microseconds>(capture_end - capture_start).count();
        
        // Check if this is a keyframe based on motion
        bool isKey = isKeyFrame(frame);
        
        // Use dynamic frame skipping
        frameCounter++;
        skipFrame = !isKey && (frameCounter % currentFrameSkip != 0);
        
        if (skipFrame) {
            // Just display the frame without processing
            auto display_start = high_resolution_clock::now();
            
            string label = format(
                "Frame %d:\n"
                "Display FPS: %.1f\n"
                "Skip: %d\n"
                "Motion: %s",
                frameCounter,
                1000000.0 / capture_time,
                currentFrameSkip,
                isKey ? "Yes" : "No"
            );
            
            // Display on frame
            int y = 15;
            std::stringstream ss(label);
            std::string line;
            while (std::getline(ss, line)) {
                putText(frame, line, Point(0, y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
                y += 20;
            }
            
            imshow(kWinName, frame);
            
            auto display_end = high_resolution_clock::now();
            auto display_time = duration_cast<microseconds>(display_end - display_start).count();
            
            // Update statistics for skipped frames
            alltimes += capture_time + display_time;
            count += 1;
            
            continue;
        }

        // Start timing for YOLO processing
        auto process_start = high_resolution_clock::now();

        // Process frame with YOLO
        // Use pre-allocated buffers
        cv::resize(frame, frame_buffer, cv::Size(inpWidth, inpHeight));
        cv::cvtColor(frame_buffer, frame_buffer, COLOR_BGR2RGB);
        
        // Create blob using pre-allocated buffer
        blobFromImage(frame_buffer, blob_buffer, 1/255.0, Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
        
        // Set input and run forward pass
        net.setInput(blob_buffer);
        
        // Time the forward pass
        auto forward_start = high_resolution_clock::now();
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));
        auto forward_end = high_resolution_clock::now();
        
        // Process detections
        postprocess(frame_buffer, frame_buffer, outs);
        
        // End timing for YOLO processing
        auto process_end = high_resolution_clock::now();
        auto process_time = duration_cast<microseconds>(process_end - process_start).count();
        
        // Calculate timing information
        auto total_time = capture_time + process_time;
        
        // Update statistics
        alltimes += total_time;
        count += 1;
        
        // Display timing information
        string label = format(
            "Frame %d:\n"
            "Display FPS: %.1f\n"
            "Process FPS: %.1f\n"
            "Total Time: %.1f ms\n"
            "Motion: %s",
            frameCounter,
            1000000.0 / capture_time,
            1000000.0 / process_time,
            total_time / 1000.0,
            isKey ? "Yes" : "No"
        );
        
        // Print to console
        cout << "\r" << label << flush;
        
        // Display on frame
        int y = 15;
        std::stringstream ss(label);
        std::string line;
        while (std::getline(ss, line)) {
            putText(frame_buffer, line, Point(0, y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
            y += 20;
        }
        
        // Write the frame with the detection boxes
        cv::cvtColor(frame_buffer, display_buffer, COLOR_RGB2BGR);
        if (parser.has("image")) imwrite(outputFile, display_buffer);
        else if (!skipFrame) video.write(display_buffer);
        
        imshow(kWinName, frame_buffer);
    }
    
    if (tof.stop())
    {
        return -1;
    }

    if (tof.close())
    {
        return -1;
    }
    
    // Calculate and display final statistics
    double mean_time = (alltimes/count)/1000.0;
    double effective_fps = 1000.0/mean_time;
    cout << "\n\nFinal Statistics:" << endl;
    cout << "MEAN TIME PER PROCESSED FRAME = " << mean_time << " ms" << endl;
    cout << "EFFECTIVE FPS (including skipped frames) = " << effective_fps/currentFrameSkip << endl;
    cout << "TOTAL PROCESSED FRAMES = " << count << endl;
    
    cap.release();
    if (!parser.has("image")) video.release();

    return 0;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, Mat& oldframe, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height /2;
                cout
                << "Raw detection: class=" << classIdPoint.x
                << " conf=" << confidence
                << " bbox=[" << left << "," << top
                << "," << width << "," << height << "]" << std::endl;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence); 
                boxes.push_back(Rect(left, top, width, height));
                //boxes.push_back(newbox);
                
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        cout
        << "Drawing box: class=" << classIds[idx]
        << " conf=" << confidences[idx]
        << " rect=(" << box.x <<","<< box.y
        <<","<< box.width <<","<< box.height
        <<")" << std::endl;
        // box.width = box.x + box.width;
        // box.height = box.y + box.height;

        // cout<<"Before = " << box.x << " " << box.y << " " << box.width << " " << box.height << endl;
        
        // scale_coords(frame.size(), oldframe.size(), box);
        // cout<<"After = " << box.x << " " << box.y << " " << box.width << " " << box.height << endl;

        drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
        //drawPred(classIds[idx], confidences[idx], box.x, box.y, box.width, box.height, oldframe);
    }
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box/
    
    // cout<<"draww = " << left << " " << top << " " << right << " " << bottom << endl;
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}
