#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

/* float relu(float x) {
    return max(0.0f, x);
}


// Apply ReLU activation function element-wise on a matrix
vector<vector<float>> applyReLU(const vector<vector<vector<float> > >& matrix) {
    int input_channels = matrix.size();
    int rows = matrix[0].size();
    int cols = matrix[0][0].size();

    vector<vector<vector<float> > > result(input_channels, vector<vector<float> >(rows, vector<float>(cols, 0.0f)));
    for (int k=0; k<input_channels; k++) {
        for (int i = 0; i < rows; ++i) {
          for (int j = 0; j < cols; ++j) {
              result[k][i][j] = relu(matrix[k][i][j]);
          }
      }
    }
    

    return result;
} */

__device__ float relu(float x) {
    return max(0.0f, x);
}

__global__ void applyReLU_kernel(float* input, float* output, int rows, int cols) {
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < rows && j < cols) {
        output[k*rows*cols + i*cols + j] = relu(input[k*rows*cols + i*cols + j]);
    }
    
}

void applyReLU(float* input, float* output, int input_channels, int rows, int cols) {

    
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_channels * rows * cols * sizeof(float));
    cudaMalloc(&d_output, input_channels * rows * cols * sizeof(float));

    
    cudaMemcpy(d_input, input, input_channels * rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((rows + threadsPerBlock.x - 1) / threadsPerBlock.x, (cols + threadsPerBlock.y - 1) / threadsPerBlock.y, input_channels);


    applyReLU_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, rows, cols);


    cudaMemcpy(output, d_output, input_channels * rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    
    cudaFree(d_input);
    cudaFree(d_output);

    return;
    
}


/*int main() {
    
    float input[] = {1.0f, -2.0f, -3.0f, 4.0f, -5.0f, 6.0f, 7.0f, -8.0f};
    
    int input_channels = 2;
    int rows = 2;
    int cols = 2;
    
    
    float* output = new float[input_channels * rows * cols];
    
    applyReLU(input, output, input_channels, rows, cols);
    
    cout << "Printing output:\n";
    for (int i = 0; i < input_channels; i++) {
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {
                cout << output[i * rows * cols + j * cols + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    
    // Free memory
    delete[] output;
    
    return 0;
}
*/






