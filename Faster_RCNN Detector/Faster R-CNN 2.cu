#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <time.h>
#include <chrono>

#define IMAGE_HEIGHT 28
#define IMAGE_WIDTH 28
#define NUM_CHANNELS 1
#define NUM_CLASSES 10
#define BATCH_SIZE 128

using namespace std;
using namespace std::chrono;

// Function to convert h5 weights to arrays
void convert_weights_to_arrays(const string& weight_file, float** weights, int& num_weights) 
{
    ifstream file(weight_file, ios::binary | ios::ate);
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);
    vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        cerr << "Error reading weight file" << endl;
    }
    num_weights = size / sizeof(float);
    *weights = new float[num_weights];
    memcpy(*weights, buffer.data(), size);
}

// Function to perform forward pass on the GPU
void forward_pass(const float* input, float* output, float** weights, const int& num_weights, cudnnHandle_t& cudnn, cublasHandle_t& cublas) {
    // Initialize tensors
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, NUM_CLASSES, 1, 1);

    // Allocate memory on GPU
    float* gpu_input;
    float* gpu_output;
    cudaMalloc((void**)&gpu_input, BATCH_SIZE * NUM_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH * sizeof(float));
    cudaMalloc((void**)&gpu_output, BATCH_SIZE * NUM_CLASSES * sizeof(float));

    // Copy input data to GPU
    cudaMemcpy(gpu_input, input, BATCH_SIZE * NUM_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    // Perform forward pass on GPU
    // Create a linear layer on GPU
    cudnnFilterDescriptor_t weight, bias_desc;
    cudnnCreateFilterDescriptor(&weight);
    cudnnCreateFilterDescriptor(&bias_desc);
    cudnnSetFilter4dDescriptor(weight, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, NUM_CHANNELS, NUM_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH);
    cudnnSetFilter4dDescriptor(bias_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, NUM_CLASSES, 1, 1, 1);

    // Allocate memory on GPU for weights and biases
    float* gpu_weights;
    float* gpu_biases;
    cudaMalloc((void**)&gpu_weights, num_weights * sizeof(float));
    cudaMalloc((void**)&gpu_biases, NUM_CLASSES * sizeof(float));

    // Copy weights and biases to GPU
    cudaMemcpy(gpu_weights, *weights, num_weights * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_biases, *weights + num_weights - NUM_CLASSES, NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice);

    // Perform convolution
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn, &alpha, input_desc, gpu_input, weight, gpu_weights, CUDNN_CONVOLUTION_FWD_NO_WORKSPACE, &beta, output_desc, gpu_output);

    // Perform bias addition
    cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, NUM_CLASSES, BATCH_SIZE, 1, &alpha, gpu_biases, NUM_CLASSES, gpu_output, NUM_CLASSES, &beta, gpu_output, NUM_CLASSES);

    // Copy output data back to host
    cudaMemcpy(output, gpu_output, BATCH_SIZE * NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(weight);
    cudnnDestroyFilterDescriptor(bias_desc);
    cudaFree(gpu_input);
    cudaFree(gpu_output);
    cudaFree(gpu_weights);
    cudaFree(gpu_biases);

}

int main(int argc, char** argv) 
{
    if (argc != 2) 
    {
        cerr << "Usage: " << argv[0] << " <weight file>" << endl;
        return 1;
    }

    // Load weights
    float* weights;
    int num_weights;
    convert_weights_to_arrays(argv[1], &weights, num_weights);

    // Initialize CUDA and cuDNN
    cudaSetDevice(0);
    cudnnHandle_t cudnn;

    cublasHandle_t cublas;
    cudnnCreate(&cudnn);
    cublasCreate(&cublas);

    // Initialize input and output arrays
    float* input = new float[BATCH_SIZE * NUM_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH];
    float* output = new float[BATCH_SIZE * NUM_CLASSES];

    // Perform image classification
    classify_images(cudnn, cublas, weights, num_weights, input, output);

    // Clean up
    cudnnDestroy(cudnn);
    cublasDestroy(cublas);
    delete[] input;
    delete[] output;
    delete[] weights;

    return 0;

}
    
    
