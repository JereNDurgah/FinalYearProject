#include <cuda_runtime.h>
#include <cublas_v2.h>  
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cstring>
#include <NvInfer.h>

int main(int argc, char** argv)
{
    // Load pre-trained Faster R-CNN model
    auto runtime = nvinfer1::createInferRuntime(gLogger);
    auto engine = runtime->deserializeCudaEngine("faster_rcnn.engine");
    auto context = engine->createExecutionContext();

    // Load input image
    cv::Mat image = cv::imread("input.jpg");

    // Resize image to the size expected by the model
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0, cv::Size(600, 800), cv::Scalar(), true, false);

    // Copy input to GPU memory
    float* input_data;
    cudaMalloc((void**) &input_data, blob.total() * sizeof(float));
    cudaMemcpy(input_data, blob.data, blob.total() * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for output on GPU
    float* detection_data;
    cudaMalloc((void**) &detection_data, 1000 * 7 * sizeof(float));

    // Run forward pass on GPU using TensorRT
    context->enqueue(1, &input_data, detection_data, nullptr);

    // Copy output from GPU memory to CPU memory
    cv::Mat detection(1000, 7, CV_32FC1);
    cudaMemcpy(detection.data, detection_data, 1000 * 7 * sizeof(float), cudaMemcpyDeviceToHost);

    // Extract and display results
    std::vector<float> scores, boxes;
    for (int i = 0; i < detection.rows; i++) 
    {
        float confidence = detection.at<float>(i, 2);
        if (confidence > 0.5) {
            int x1 = static_cast<int>(detection.at<float>(i, 3) * image.cols);
            int y1 = static_cast<int>(detection.at<float>(i, 4) * image.rows);
            int x2 = static_cast<int>(detection.at<float>(i, 5) * image.cols);
            int y2 = static_cast<int>(detection.at<float>(i, 6) * image.rows);
            cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);
        }
    }
    cv::imshow("Detection Results", image);
    cv::waitKey();

    //There are results being stored on the GPU memory buffer 

    // Clean up
    cudaFree(input_data);
    cudaFree(detection_data);
    context->destroy();
    engine
}