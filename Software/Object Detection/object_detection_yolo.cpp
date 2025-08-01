// This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

// Usage example:  ./object_detection_yolo.out --video=run.mp4
//                 ./object_detection_yolo.out --image=bird.jpg
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>  // Add this for setprecision

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <opencv2/features2d.hpp>  // For ORB feature detection

const char* keys =
"{help h usage ? | | Usage examples: \n\t\t./object_detection_yolo.out --image=dog.jpg \n\t\t./object_detection_yolo.out --video=run_sm.mp4}"
"{image i        |<none>| input image   }"
"{video v       |<none>| input video   }"
"{device d         |0| input device index (for webcam) }"
"{save-video s  | | save output video to file }"
;
using namespace cv;
using namespace dnn;
using namespace std;
using namespace std::chrono;

// Initialize the parameters
float confThreshold = 0.4; // Confidence threshold
float nmsThreshold = 0.3;  // Non-maximum suppression threshold
float motionThreshold = 7;  // Motion detection threshold
int inpWidth = 128;  // Reduced from 224 for faster processing
int inpHeight = 96;  // Reduced from 160 for faster processing
int minFrameSkip = 5;  // Increased from 2
int maxFrameSkip = 10;  // Increased from 4
int currentFrameSkip = minFrameSkip;  // Dynamic frame skip
int frameCounter = 0;
bool skipFrame = false;
vector<string> classes;

// Add these global variables after the existing ones
Mat prevFrame, prevGray;
bool firstFrame = true;
int orbMaxFeatures = 50;  // Reduced from 100 for faster processing
float orbScaleFactor = 1.2f;  // Standard scale factor
int orbLevels = 3;  // Reduced from 4 for faster processing
int orbEdgeThreshold = 31;  // Standard edge threshold
int orbPatchSize = 31;  // Standard patch size
int orbFastThreshold = 30;  // Increased from 20 for faster processing
Ptr<ORB> orb = ORB::create(
    orbMaxFeatures,  // max features
    orbScaleFactor,  // scale factor
    orbLevels,       // pyramid levels
    orbEdgeThreshold,// edge threshold
    0,               // first level
    2,               // WTA_K
    ORB::HARRIS_SCORE,// score type
    orbPatchSize,    // patch size
    orbFastThreshold // FAST threshold
);
vector<KeyPoint> prevKeypoints;
Mat prevDescriptors;
float contentThreshold = 0.3;  // Threshold for content change
int minFeatures = 10;  // Minimum number of features to consider

// Define a struct to hold detection information
struct Detection {
    int classId;
    float confidence;
    Rect bbox;
    string className;
};

// Global vector to store current frame detections
vector<Detection> currentDetections;

// Update function declaration
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
bool isContentKeyFrame(const Mat& currentFrame) {
    if (firstFrame) {
        firstFrame = false;
        currentFrame.copyTo(prevFrame);
        // Detect features in first frame
        orb->detectAndCompute(prevFrame, noArray(), prevKeypoints, prevDescriptors);
        return true;
    }

    // Detect features in current frame
    vector<KeyPoint> currentKeypoints;
    Mat currentDescriptors;
    orb->detectAndCompute(currentFrame, noArray(), currentKeypoints, currentDescriptors);

    // If not enough features, consider it a keyframe
    if (currentKeypoints.size() < minFeatures || prevKeypoints.size() < minFeatures) {
        currentFrame.copyTo(prevFrame);
        prevKeypoints = currentKeypoints;
        prevDescriptors = currentDescriptors;
        return true;
    }

    // Match features between frames
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(prevDescriptors, currentDescriptors, matches);

    // Calculate match ratio
    float matchRatio = float(matches.size()) / float(min(prevKeypoints.size(), currentKeypoints.size()));
    
    // Update previous frame data
    currentFrame.copyTo(prevFrame);
    prevKeypoints = currentKeypoints;
    prevDescriptors = currentDescriptors;

    // Return true if content change is significant
    return matchRatio < contentThreshold;
}

// Modify isKeyFrame to use both motion and content
bool isKeyFrame(const Mat& currentFrame) {
    bool hasMotion = false;
    bool hasContentChange = false;

    // Check motion
    if (firstFrame) {
        firstFrame = false;
        currentFrame.copyTo(prevFrame);
        cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);
        return true;
    }

    // Convert to grayscale
    Mat currentGray;
    cvtColor(currentFrame, currentGray, COLOR_BGR2GRAY);
    
    // Calculate motion
    Mat diff;
    absdiff(prevGray, currentGray, diff);
    
    Scalar meanDiff = mean(diff);
    float motionScore = meanDiff[0];
    
    // Adjust frame skip based on motion intensity
    if (motionScore > motionThreshold) {
        currentFrameSkip = minFrameSkip;  // Process more frames when motion is high
        hasMotion = true;
    } else {
        currentFrameSkip = maxFrameSkip;  // Skip more frames when motion is low
        hasMotion = false;
    }

    // Check content change using the same frame
    hasContentChange = isContentKeyFrame(currentFrame);

    // Update previous frame
    currentFrame.copyTo(prevFrame);
    currentGray.copyTo(prevGray);

    return hasMotion || hasContentChange;
}

// Add these parameters back for FPS calculations
const int FPS_WINDOW_SIZE = 30;  // Number of frames to average FPS over
vector<double> fps_history;      // Store recent FPS values

int main(int argc, char** argv)
{
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
    String modelWeights = "/home/vision-rpi/Desktop/Tiny-Yolov3-OpenCV-Cpp/weights/yolov3-tiny.weights";

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    
    // Set CPU as backend
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    
    // Open a video file or an image file or a camera stream.
    string outputFile = "yolo_out_cpp.avi";
    VideoCapture cap;
    VideoWriter video;
    Mat frame, blob;
    
    try {
        // Open the webcam
        cap.open(parser.get<int>("device"));
        if (!cap.isOpened()) {
            throw runtime_error("Could not open webcam");
        }
    }
    catch(const exception& e) {
        cout << "Error: " << e.what() << endl;
        return 0;
    }
    
    // Initialize video writer only if save-video flag is set
    if (parser.has("save-video")) {
        video.open(outputFile, VideoWriter::fourcc('M','J','P','G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
        if (!video.isOpened()) {
            cout << "Could not open video writer. Output will not be saved." << endl;
        }
    }
    
    // Create a window
    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);
    
    // FPS tracking variables
    double capture_times = 0.0;
    double process_times = 0.0;
    int total_frames = 0;
    int processed_frames = 0;
    Mat oldframe;
    Size sz;

    // Process frames.
    while (waitKey(1) < 0)
    {
        // Start timing for frame capture
        auto capture_start = high_resolution_clock::now();
        
        // get frame from the video buffer
        cap >> frame;
        
        if (frame.empty()) {
            cout << "Done processing !!!" << endl;
            if (parser.has("save-video")) {
                cout << "Output file is stored as " << outputFile << endl;
            }
            waitKey(3000);
            break;
        }

        auto capture_end = high_resolution_clock::now();
        auto capture_time = duration_cast<microseconds>(capture_end - capture_start).count();
        capture_times += capture_time;
        total_frames++;
        
        // Calculate instant FPS
        double instant_fps = 1000000.0 / capture_time;
        
        // Add to history and maintain window size
        fps_history.push_back(instant_fps);
        if (fps_history.size() > FPS_WINDOW_SIZE) {
            fps_history.erase(fps_history.begin());
        }
        
        // Calculate average FPS over window
        double avg_fps = 0.0;
        for (double fps : fps_history) {
            avg_fps += fps;
        }
        avg_fps /= fps_history.size();
        
        // Calculate total FPS (including skipped frames)
        double total_fps = 1000000.0 / (capture_times / total_frames);
        
        // Check if this is a keyframe based on motion
        bool isKey = isKeyFrame(frame);
        
        // Use dynamic frame skipping
        frameCounter++;
        skipFrame = !isKey && (frameCounter % currentFrameSkip != 0);
        
        if (skipFrame) {
            // Just display the frame without processing
            imshow(kWinName, frame);
            
            // Print FPS information to console
            cout << "\rFrame " << frameCounter 
                 << " - Instant FPS: " << fixed << setprecision(1) << instant_fps
                 << " - Avg FPS: " << fixed << setprecision(1) << avg_fps
                 << " - Total FPS: " << fixed << setprecision(1) << total_fps
                 << " - Motion: " << (isKey ? "Yes" : "No") << flush;
            
            continue;
        }

        // Start timing for YOLO processing
        auto process_start = high_resolution_clock::now();

        // Process frame with YOLO
        // Use pre-allocated buffers
        cv::resize(frame, frame_buffer, cv::Size(inpWidth, inpHeight), 0, 0, INTER_NEAREST);  // Faster resize
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
        process_times += process_time;
        processed_frames++;
        
        // Calculate process FPS (time per processed frame)
        double process_fps = 1000000.0 / (process_times / processed_frames);
        
        // Calculate effective FPS (total frames / total time including processing)
        double total_time = (capture_times + process_times) / 1000000.0;  // Convert to seconds
        double effective_fps = total_frames / total_time;
        
        // Print FPS information to console
        cout << "\rFrame " << frameCounter 
             << " - Instant FPS: " << fixed << setprecision(1) << instant_fps
             << " - Avg FPS: " << fixed << setprecision(1) << avg_fps
             << " - Total FPS: " << fixed << setprecision(1) << total_fps
             << " - Process FPS: " << fixed << setprecision(1) << process_fps
             << " - Effective FPS: " << fixed << setprecision(1) << effective_fps
             << " - Motion: " << (isKey ? "Yes" : "No") << flush;
        
        // Write the frame with the detection boxes
        cv::cvtColor(frame_buffer, display_buffer, COLOR_RGB2BGR);
        if (parser.has("save-video") && video.isOpened()) {
            video.write(display_buffer);
        }
        
        imshow(kWinName, frame_buffer);
    }
    
    // Calculate and display final statistics
    double mean_capture_time = (capture_times/total_frames)/1000.0;
    double mean_process_time = (process_times/processed_frames)/1000.0;
    double total_fps = 1000.0/mean_capture_time;
    double process_fps = 1000.0/mean_process_time;
    double effective_fps = min(total_fps, process_fps * (processed_frames / total_frames));
    
    cout << "\n\nFinal Statistics:" << endl;
    cout << "MEAN CAPTURE TIME PER FRAME = " << mean_capture_time << " ms" << endl;
    cout << "MEAN PROCESS TIME PER FRAME = " << mean_process_time << " ms" << endl;
    cout << "TOTAL FPS = " << total_fps << endl;
    cout << "PROCESS FPS = " << process_fps << endl;
    cout << "EFFECTIVE FPS = " << effective_fps << endl;
    cout << "TOTAL FRAMES = " << total_frames << endl;
    cout << "PROCESSED FRAMES = " << processed_frames << endl;
    if (parser.has("save-video")) {
        cout << "Output video saved as: " << outputFile << endl;
    }
    
    cap.release();
    if (parser.has("save-video")) video.release();

    return 0;
}

// Update postprocess function
void postprocess(Mat& frame, Mat& oldframe, const vector<Mat>& out)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
    // Clear previous detections
    currentDetections.clear();
    
    for (size_t i = 0; i < out.size(); ++i)
    {
        float* data = (float*)out[i].data;
        for (int j = 0; j < out[i].rows; ++j, data += out[i].cols)
        {
            Mat scores = out[i].row(j).colRange(5, out[i].cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height /2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence); 
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        
        // Create and store detection
        Detection det;
        det.classId = classIds[idx];
        det.confidence = confidences[idx];
        det.bbox = box;
        det.className = classes[classIds[idx]];
        currentDetections.push_back(det);

        drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
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
