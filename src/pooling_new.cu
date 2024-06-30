#include <iostream>
#include <vector>
#include <limits>
#include <cfloat>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;

/* vector<vector<vector<float> > > maxPooling(const vector<vector<vector<float> > >& input, int poolSize) {
    
    int input_channels = input.size();
    int rows = input[0].size();
    int cols = input[0][0].size();
    int outputRows = rows / poolSize;
    int outputCols = cols / poolSize;

    vector<vector<vector<float> > > output(input_channels, vector<vector<float> >(outputRows, vector<float>(outputCols, 0.0f)));
    
    for (int k=0; k<input_channels; k++) {
        for (int i = 0; i < outputRows; ++i) {
            for (int j = 0; j < outputCols; ++j) {
                //float maxVal = numeric_limits<float>::lowest();
                float maxVal = -1e9;
                for (int m = 0; m < poolSize; ++m) {
                    for (int n = 0; n < poolSize; ++n) {
                        maxVal = max(maxVal, input[k][i * poolSize + m][j * poolSize + n]);
                    }
                }
                output[k][i][j] = maxVal;
            }
        }
    }

    return output;
} */

__global__ void maxPoolingKernel(float* input, float* output, int input_channels, int inputSize, int poolSize) {
    
    int k = threadIdx.z + blockIdx.z * blockDim.z; // Kernel index
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Row index
    int j = threadIdx.y + blockIdx.y * blockDim.y; // Column index
    
    int outputSize = inputSize/poolSize;
    
    if (i < inputSize/poolSize && j < inputSize / poolSize) {
        float maxVal = -1e9;
        for (int m = 0; m < poolSize; ++m) {
            for (int n = 0; n < poolSize; ++n) {
                float val = input[k * inputSize * inputSize + (i * poolSize + m) * inputSize + j * poolSize + n];
                maxVal = max(maxVal, val);
            }
        }
        
        output[k*outputSize*outputSize+ i*outputSize + j] = maxVal;
    }
    
}

void maxPooling(float* input, float* output, int input_channels, int inputSize, int poolSize) {

    int outputSize = inputSize / poolSize;
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_channels * inputSize * inputSize * sizeof(float));
    cudaMalloc(&d_output, input_channels * outputSize * outputSize * sizeof(float));

    
    cudaMemcpy(d_input, input, input_channels * inputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((outputSize + threadsPerBlock.x - 1) / threadsPerBlock.x, (outputSize + threadsPerBlock.y - 1) / threadsPerBlock.y, input_channels);


    maxPoolingKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, input_channels, inputSize, poolSize);


    cudaMemcpy(output, d_output, input_channels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    
    cudaFree(d_input);
    cudaFree(d_output);

    return;
    
}


/*int main() {
    
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    
    int input_channels = 5;
    int inputSize = 2;
    int poolSize = 2;
    
    int outputSize = inputSize / poolSize;
    
    float* output = new float[input_channels * outputSize * outputSize];
    
    maxPooling(input, output, input_channels, inputSize, poolSize);
    
    cout << "Printing output:\n";
    for (int i = 0; i < input_channels; i++) {
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
    
    return 0;
}*/





