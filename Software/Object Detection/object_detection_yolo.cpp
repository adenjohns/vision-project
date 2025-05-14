// This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

// Usage example:  ./object_detection_yolo.out --video=run.mp4
//                 ./object_detection_yolo.out --image=bird.jpg
#include <fstream>
#include <sstream>
#include <iostream>

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
;
using namespace cv;
using namespace dnn;
using namespace std;
using namespace std::chrono;

// Initialize the parameters
float confThreshold = 0.4; // Confidence threshold
float nmsThreshold = 0.3;  // Non-maximum suppression threshold
float motionThreshold = 0.5;  // Motion detection threshold
int inpWidth = 64;  // Reduced from 128 for faster processing
int inpHeight = 48; // Reduced from 96 for faster processing
int minFrameSkip = 2;  // Minimum frames to skip
int maxFrameSkip = 4;  // Maximum frames to skip
int currentFrameSkip = minFrameSkip;  // Dynamic frame skip
int frameCounter = 0;
bool skipFrame = false;
vector<string> classes;

// Add these global variables after the existing ones
Mat prevFrame, prevGray;
bool firstFrame = true;
Ptr<ORB> orb = ORB::create();
vector<KeyPoint> prevKeypoints;
Mat prevDescriptors;
float contentThreshold = 0.3;  // Threshold for content change
int minFeatures = 10;  // Minimum number of features to consider

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

    Mat currentGray;
    cvtColor(currentFrame, currentGray, COLOR_BGR2GRAY);
    
    Mat diff;
    absdiff(prevGray, currentGray, diff);
    
    // For grayscale images, meanDiff[0] contains the average pixel difference
    Scalar meanDiff = mean(diff);
    float motionScore = meanDiff[0];  // Just use the first channel for grayscale
    
    // Adjust frame skip based on motion intensity
    if (motionScore > motionThreshold) {
        currentFrameSkip = minFrameSkip;  // Process more frames when motion is high
        hasMotion = true;
    } else {
        currentFrameSkip = maxFrameSkip;  // Skip more frames when motion is low
        hasMotion = false;
    }

    // Check content change
    hasContentChange = isContentKeyFrame(currentFrame);

    // Update previous frame
    currentFrame.copyTo(prevFrame);
    currentGray.copyTo(prevGray);

    // Return true if either motion or content change is significant
    return hasMotion || hasContentChange;
}

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

    // Process frames.
    while (waitKey(1) < 0)
    {
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
