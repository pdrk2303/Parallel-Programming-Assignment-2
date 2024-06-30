#include <cuda_runtime.h>
#include <vector>
#include <iostream>

using namespace std;

/* vector<vector<vector<float> > > convolutionWithoutPadding(vector<vector<vector<float> > >& input, vector<vector<vector<vector<float> > > >& kernels, vector<float> biases) {
    
    int input_channels = input.size();
    int inputSize = input[0].size();
    int kernelSize = kernels[0][0].size(); // Assuming all kernels have the same size
    int numKernels = kernels.size();
    int outputSize = inputSize - kernelSize + 1;

    // Perform convolution
    
    vector<vector<vector<float> > > output(numKernels, vector<vector<float> >(outputSize, vector<float>(outputSize, 0.0f)));

    for (int k=0; k<numKernels; k++) {
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                for (int m = 0; m < kernelSize; ++m) {
                    for (int n = 0; n < kernelSize; ++n) {
                        for (int o=0; o<input_channels; o++) {
                            output[k][i][j] += input[o][i + m][j + n] * kernels[k][o][m][n];
                        }
                    }
                }
            }
        }
        
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                output[k][i][j] += biases[k];
            }
        }

    }

    return output;
} */

__global__ void convolutionWithoutPaddingKernel(float* input, float* kernels, float* biases, float* output, int input_channels, int inputSize, int kernelSize, int numKernels) {

    int k = threadIdx.z + blockIdx.z * blockDim.z; // Kernel index
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Row index
    int j = threadIdx.y + blockIdx.y * blockDim.y; // Column index

    
    if (i < inputSize - kernelSize + 1 && j < inputSize - kernelSize + 1) {
        float sum = 0.0f;
        for (int m = 0; m < kernelSize; ++m) {
            for (int n = 0; n < kernelSize; ++n) {
                for (int o = 0; o < input_channels; o++) {
                    int input_index = o * inputSize * inputSize + (i + m) * inputSize + (j + n);
                    int kernel_index = k * input_channels * kernelSize * kernelSize + o * kernelSize * kernelSize + m * kernelSize + n;
                    sum += input[input_index] * kernels[kernel_index];
                }
            }
        }
        output[k * (inputSize - kernelSize + 1) * (inputSize - kernelSize + 1) + i * (inputSize - kernelSize + 1) + j] = sum + biases[k];
    }
}


void convolutionWithoutPadding(float * input,float *kernels, float *biases, float * output,int input_channels,int inputSize, int numKernels, int kernelSize) {
    
    int outputSize = inputSize - kernelSize +1;
    
    float *d_input, *d_kernels, *d_biases, *d_output;
    cudaMalloc(&d_input, input_channels * inputSize * inputSize * sizeof(float));
    cudaMalloc(&d_kernels, numKernels * input_channels * kernelSize * kernelSize * sizeof(float));
    cudaMalloc(&d_biases, numKernels * sizeof(float));
    cudaMalloc(&d_output, numKernels * outputSize * outputSize * sizeof(float));

    
    cudaMemcpy(d_input, input, input_channels * inputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernels, kernels, numKernels * input_channels * kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, numKernels * sizeof(float), cudaMemcpyHostToDevice);


    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((inputSize - kernelSize + 1 + threadsPerBlock.x - 1) / threadsPerBlock.x, (inputSize - kernelSize + 1 + threadsPerBlock.y - 1) / threadsPerBlock.y, numKernels);


    convolutionWithoutPaddingKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_kernels, d_biases, d_output, input_channels, inputSize, kernelSize, numKernels);


    cudaMemcpy(output, d_output, numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    
    cudaFree(d_input);
    cudaFree(d_kernels);
    cudaFree(d_biases);
    cudaFree(d_output);

    return;
}


/*int main() {
    /* vector<vector<vector<float> > > input(2, vector<vector<float> >(2, vector<float>(2, 0.0f)));
    input[0][0][0] = 1.0f;
    input[0][0][1] = 0.0f;
    input[0][1][0] = 0.0f;
    input[0][1][1] = 1.0f;
    
    input[1][0][0] = 0.0f;
    input[1][0][1] = 1.0f;
    input[1][1][0] = 1.0f;
    input[1][1][1] = 0.0f;
    
    vector<vector<vector<vector<float> > > > kernels(3, vector<vector<vector<float> > >(2, vector<vector<float> >(1, vector<float>(1, 0.0f))));
    
    kernels[0][0][0][0] = 1.0f;
    kernels[0][1][0][0] = 2.0f;
    
    kernels[1][0][0][0] = 3.0f;
    kernels[1][1][0][0] = 4.0f;
    
    kernels[2][0][0][0] = 5.0f;
    kernels[2][1][0][0] = 6.0f;
    
    vector<float> biases(3, 1.0f); */
    
    /* float input[] = { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f };
    float kernels[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float biases[] = { 1.0f, 1.0f, 1.0f };
    
    int input_channels = 2;
    int inputSize = 2;
    int numKernels = 3;
    int kernelSize = 1; 
    
    float input[] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    float kernels[] = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    float biases[] = {2.5f, 2.5f};
    
    int input_channels = 2;
    int inputSize = 3;
    int numKernels = 2;
    int kernelSize = 2; 
    
    
    int outputSize = inputSize - kernelSize + 1;
    
    float* output = new float[numKernels * outputSize * outputSize];
    
    convolutionWithoutPadding(input, kernels, biases, output, input_channels, inputSize, numKernels, kernelSize);
    
    cout << "Printing output:\n";
    for (int i = 0; i < numKernels; i++) {
        for (int j = 0; j < outputSize; j++) {
            for (int k = 0; k < outputSize; k++) {
                cout << output[i * outputSize * outputSize + j * outputSize + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    
    // Free memory
    delete[] output;
    
    
    /* vector<vector<vector<float> > > output = convolutionWithoutPadding(input, kernels, biases);
    
    cout << "Number of output channels: " << output.size() << endl;
    cout << "Output dimension: " << output[0].size() << " * " << output[0][0].size() << endl;
    
    cout << "Printing output:\n";
    for (int i=0; i<output.size(); i++) {
        for (int j=0; j<output[0].size(); j++) {
            for (int k=0; k<output[0][0].size(); k++) {
                cout << output[i][j][k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    } 
    
    return 0;
}*/



















