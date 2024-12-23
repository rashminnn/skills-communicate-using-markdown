# This is a header1 
## This is a header2
### This is a header3
#### This is a header4
##### This is a header5
###### This is a header6

Checking header sizes 

![Github Wallpaper](https://c4.wallpaperflare.com/wallpaper/607/551/626/code-github-logo-open-source-wallpaper-preview.jpg)

```
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

// Helper function to preprocess the input image
cv::Mat preprocess(const cv::Mat& img, const cv::Size& input_size) {
    cv::Mat resized, float_img;
    cv::resize(img, resized, input_size);
    resized.convertTo(float_img, CV_32F, 1.0 / 255.0);
    return float_img;
}

// Helper function to postprocess the output of the model
void postprocess(const cv::Mat& img, const std::vector<float>& output, float conf_threshold, float nms_threshold) {
    int num_classes = 80; // Assuming COCO dataset
    int num_detections = output.size() / (num_classes + 5);

    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> confidences;

    for (int i = 0; i < num_detections; ++i) {
        float conf = output[i * (num_classes + 5) + 4];
        if (conf < conf_threshold) continue;

        float* class_scores = &output[i * (num_classes + 5) + 5];
        cv::Mat scores(1, num_classes, CV_32F, class_scores);
        cv::Point class_id_point;
        double max_class_score;
        minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);
        if (max_class_score < conf_threshold) continue;

        int center_x = static_cast<int>(output[i * (num_classes + 5) + 0] * img.cols);
        int center_y = static_cast<int>(output[i * (num_classes + 5) + 1] * img.rows);
        int width = static_cast<int>(output[i * (num_classes + 5) + 2] * img.cols);
        int height = static_cast<int>(output[i * (num_classes + 5) + 3] * img.rows);
        int left = center_x - width / 2;
        int top = center_y - height / 2;

        boxes.emplace_back(left, top, width, height);
        class_ids.push_back(class_id_point.x);
        confidences.push_back(static_cast<float>(max_class_score));
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        int class_id = class_ids[idx];
        float confidence = confidences[idx];

        cv::rectangle(img, box, cv::Scalar(0, 255, 0), 2);
        std::string label = cv::format("Class %d: %.2f", class_id, confidence);
        int base_line;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);
        cv::rectangle(img, cv::Point(box.x, box.y - label_size.height),
            cv::Point(box.x + label_size.width, box.y + base_line),
            cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(img, label, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
}

int main() {
    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv5");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Load the ONNX model
    const char* model_path = "model.onnx";
    Ort::Session session(env, model_path, session_options);

    // Get input and output info
    Ort::AllocatorWithDefaultOptions allocator;
    const char* input_name = session.GetInputName(0, allocator);
    const char* output_name = session.GetOutputName(0, allocator);

    // Open the webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam" << std::endl;
        return -1;
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Could not read frame" << std::endl;
            break;
        }

        // Preprocess the frame
        cv::Size input_size(640, 640); // Assuming YOLOv5 input size
        cv::Mat input_image = preprocess(frame, input_size);

        // Prepare input tensor
        std::vector<int64_t> input_shape = {1, 3, input_size.height, input_size.width};
        std::vector<float> input_tensor_values(input_image.begin<float>(), input_image.end<float>());
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

        // Run inference
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);

        // Process output
        std::vector<float> output(output_tensors.front().GetTensorMutableData<float>(), output_tensors.front().GetTensorMutableData<float>() + output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount());
        postprocess(frame, output, 0.5, 0.4); // Adjust thresholds as needed

        // Display the frame with bounding boxes
        cv::imshow("YOLOv5 Object Detection", frame);
        if (cv::waitKey(1) == 27) { // Press 'Esc' to exit
            break;
        }
    }

    return 0;
}
```

- [x] Turn on GitHub Pages
- [ ] Outline my portfolio
- [x] Introduce myself to the world
