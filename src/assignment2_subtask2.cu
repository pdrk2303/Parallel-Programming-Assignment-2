#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <string>
#include <cstdlib>
#include <limits>
#include <cfloat>
using namespace std;

// CUDA kernel for computing softmax probabilities
__global__ void sumVector(float *vector, float *result, int n) {
    // Shared memory for storing partial sums
    __shared__ float partialSum[256];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads one element from global to shared memory
    if (i < n) {
        partialSum[tid] = exp(vector[i]);
    } else {
        partialSum[tid] = 0;
    }
    __syncthreads();

    // Reduction phase
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            partialSum[tid] += partialSum[tid + stride];
        }
        __syncthreads();
    }

    // Write the result of this block to global memory
    if (tid == 0) {
        result[blockIdx.x] = partialSum[0];
    }
}

// CUDA kernel for computing softmax probabilities
__global__ void softmax_gpu(float* input, float* probabilities, float* sum, int cols) {
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols) {
        probabilities[col] = exp(input[col]) / (*sum);
    }
}

float* softmax(std::vector<float>& input) {
    int n = input.size();
    float *d_input, *d_result;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    // Copy vector from host to device
    cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim((n + 255) / 256, 1, 1);
    dim3 blockDim(256, 1, 1);

    // Launch kernel for summing the vector elements
    sumVector<<<gridDim, blockDim>>>(d_input, d_result, n);

    // Compute sum of elements on GPU
    float h_result;
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Allocate memory for softmax probabilities
    float* probabilities;
    cudaMalloc(&probabilities, n * sizeof(float));

    // Launch softmax kernel
    dim3 threadsPerBlock7(16, 16);
    dim3 numBlocks7((1 + threadsPerBlock7.x - 1) / threadsPerBlock7.x, (n + threadsPerBlock7.y - 1) / threadsPerBlock7.y);
    softmax_gpu<<<numBlocks7, threadsPerBlock7>>>(d_input, probabilities, d_result, n);

    // Allocate memory for output
    float* output = new float[n];

    // Copy softmax probabilities from device to host
    cudaMemcpy(output, probabilities, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_result);
    cudaFree(probabilities);

    return output;
}

__global__ void sigmoid_gpu(float* input, float* probabilities, int cols) {
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols) {
        probabilities[col] = 1.0f / (1.0f + exp(-input[col]));
    }
}
float* sigmoid( std::vector<float>& input) {
    int n = input.size();
    float *d_input;
    cudaMalloc(&d_input, n * sizeof(float));
    //cudaMalloc(&d_result, sizeof(float));

    // Copy vector from host to device
    cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    float* probabilities;
    cudaMalloc(&probabilities, n * sizeof(float));

    // Launch softmax kernel
    dim3 threadsPerBlock7(16, 16);
    dim3 numBlocks7((1 + threadsPerBlock7.x - 1) / threadsPerBlock7.x, (n + threadsPerBlock7.y - 1) / threadsPerBlock7.y);
    sigmoid_gpu<<<numBlocks7, threadsPerBlock7>>>(d_input, probabilities,  n);

    // Allocate memory for output
    float* output = new float[n];

    // Copy softmax probabilities from device to host
    cudaMemcpy(output, probabilities, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    //cudaFree(d_result);
    cudaFree(probabilities);

    return output;
}

__global__ void convolutionWithPaddingKernel(float** paddedInput, float** kernel, float** output, int inputSize, int kernelSize, int paddingSize, int outputSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < outputSize && j < outputSize) {
        for (int m = 0; m < kernelSize; ++m) {
            for (int n = 0; n < kernelSize; ++n) {
                if (i + m >= 0 && j + n >= 0 && i + m < (inputSize+2*paddingSize) && j + n < (inputSize+2*paddingSize)) {
                    atomicAdd(&output[i][j], paddedInput[i + m][j + n] * kernel[m][n]);
                }
            }
        }
    }
}



vector<vector<float> > convolutionWithPadding(const vector<vector<float> >& input, const vector<vector<float> >& kernel, int paddingSize) {
    int inputSize = input.size();
    int kernelSize = kernel.size();
    //int paddingSize = (kernelSize - 1) / 2;

    // Create padded input matrix
    vector<vector<float> > paddedInput(inputSize + 2 * paddingSize, vector<float>(inputSize + 2 * paddingSize, 0.0));
    for (int i = paddingSize; i < inputSize + paddingSize; ++i) {
        for (int j = paddingSize; j < inputSize + paddingSize; ++j) {
            paddedInput[i][j] = input[i - paddingSize][j - paddingSize];
        }
    }
    //cout<<"conv 1\n";
    int outputSize = inputSize + 2 * paddingSize - kernelSize + 1;

    // Allocate memory on GPU
    // float **d_paddedInput, **d_kernel, **d_output;
    // cudaMalloc((void**)&d_paddedInput, (inputSize + 2 * paddingSize) * (inputSize + 2 * paddingSize) * sizeof(float));
    // cudaMalloc((void**)&d_kernel, kernelSize * kernelSize * sizeof(float));
    // cudaMalloc((void**)&d_output, (inputSize-kernelSize+1) * (inputSize-kernelSize+1) * sizeof(float));
    
    // float **d_paddedinput, **d_output;
    float **d_paddedInput, **d_kernel, **d_output;
    cudaMalloc((void**)&d_paddedInput, (inputSize+ (2*paddingSize)) * sizeof(float*));
    cudaMalloc((void**)&d_kernel,kernelSize*sizeof(float*));
    cudaMalloc((void**)&d_output, (outputSize) * sizeof(float*));

    //cout<<"conv 2\n";
    float *temp_paddedInput[inputSize+ (2*paddingSize)],*temp_kernel[kernelSize], *temp_output[inputSize];
    for (int i = 0; i < (inputSize+ (2*paddingSize)); ++i) {
        cudaMalloc((void**)&temp_paddedInput[i], (inputSize+ (2*paddingSize)) * sizeof(float));
        // cudaMalloc((void**)&temp_output[i], cols * sizeof(float));
        cudaMemcpy(temp_paddedInput[i], paddedInput[i].data(), (inputSize+ (2*paddingSize)) * sizeof(float), cudaMemcpyHostToDevice);
    }
    for (int i = 0; i < kernelSize; ++i) {
        cudaMalloc((void**)&temp_kernel[i], (kernelSize) * sizeof(float));
        // cudaMalloc((void**)&temp_output[i], cols * sizeof(float));
        cudaMemcpy(temp_kernel[i], kernel[i].data(), (kernelSize) * sizeof(float), cudaMemcpyHostToDevice);
    }
    for (int i = 0; i < (outputSize); ++i) {
        cudaMalloc((void**)&temp_output[i], (outputSize) * sizeof(float));
        // cudaMalloc((void**)&temp_output[i], cols * sizeof(float));
        // cudaMemcpy(temp_paddedInput[i], paddedInput[i].data(), (inputSize+ (2*paddingSize)) * sizeof(float), cudaMemcpyHostToDevice);
    }



    //cout<<"conv 3\n";

    // Copy input and kernel data from host to device
    cudaMemcpy(d_paddedInput, temp_paddedInput, (inputSize + 2 * paddingSize) * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, temp_kernel, kernelSize * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, temp_output, outputSize * sizeof(float*), cudaMemcpyHostToDevice);


    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((inputSize + blockDim.x - 1) / blockDim.x, (inputSize + blockDim.y - 1) / blockDim.y);

    // Launch CUDA kernel
    //cout<<"conv 4\n";
    convolutionWithPaddingKernel<<<gridDim, blockDim>>>(d_paddedInput, d_kernel, d_output, inputSize, kernelSize, paddingSize, outputSize);
    //cout<<"conv 5\n";
    // Copy output data from device to host
    vector<vector<float> > output(outputSize, vector<float>(outputSize, 0.0));
    // cudaMemcpy(output.data(), d_output, inputSize * inputSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_output, d_output, (outputSize) * sizeof(float*), cudaMemcpyDeviceToHost);
    for (int i = 0; i < outputSize; ++i) {
        cudaMemcpy(output[i].data(), temp_output[i], outputSize * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(temp_paddedInput[i]);
        // cudaFree(temp_kernel)
        cudaFree(temp_output[i]);
    }
    //cout<<"conv 6\n";
    // Free device memory
    cudaFree(d_paddedInput);
    cudaFree(d_kernel);
    cudaFree(d_output);

    //cout<<"conv 7\n";
    return output;
}

// CUDA kernel for max pooling
__global__ void maxPooling(float* input, float* output, int rows, int cols, int poolSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows / poolSize && col < cols / poolSize) {
        float maxVal = -FLT_MAX;
        for (int m = 0; m < poolSize; ++m) {
            for (int n = 0; n < poolSize; ++n) {
                float val = input[(row * poolSize + m) * cols + col * poolSize + n];
                maxVal = max(maxVal, val);
            }
        }
        output[row * (cols / poolSize) + col] = maxVal;
    }
}
// CUDA kernel for average pooling
__global__ void averagePooling(float* input, float* output, int rows, int cols, int poolSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows / poolSize && col < cols / poolSize) {
        float sum = 0.0f;
        for (int m = 0; m < poolSize; ++m) {
            for (int n = 0; n < poolSize; ++n) {
                sum += input[(row * poolSize + m) * cols + col * poolSize + n];
            }
        }
        output[row * (cols / poolSize) + col] = sum / (poolSize * poolSize);
    }
}

__device__ float relu(float x) {

    return max(0.0f, x);

}



__device__ float tanh_activation(float x) {

    return tanh(x);

}

__global__ void applyReLU_kernel(float** input, float** output, int rows, int cols) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    int col = blockIdx.y * blockDim.y + threadIdx.y;

    

    if (row < rows && col < cols) {

        output[row][col] = relu(input[row][col]);

    }

}



__global__ void applyTanh_kernel(float** input, float** output, int rows, int cols) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    int col = blockIdx.y * blockDim.y + threadIdx.y;

    

    if (row < rows && col < cols) {

        output[row][col] = tanh_activation(input[row][col]);

    }

}



void applyReLU(const vector < vector < float > >& input, vector<vector<float > >& output) {

    int rows = input.size();

    int cols = input[0].size();



    // Allocate memory on GPU

    float **d_input, **d_output;

    cudaMalloc((void**)&d_input, rows * sizeof(float*));

    cudaMalloc((void**)&d_output, rows * sizeof(float*));



    float *temp_input[rows], *temp_output[rows];

    for (int i = 0; i < rows; ++i) {

        cudaMalloc((void**)&temp_input[i], cols * sizeof(float));

        cudaMalloc((void**)&temp_output[i], cols * sizeof(float));

        cudaMemcpy(temp_input[i], input[i].data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    }



    cudaMemcpy(d_input, temp_input, rows * sizeof(float*), cudaMemcpyHostToDevice);

    cudaMemcpy(d_output, temp_output, rows * sizeof(float*), cudaMemcpyHostToDevice);



    // Launch kernel

    dim3 threadsPerBlock(16, 16);

    dim3 numBlocks((rows + threadsPerBlock.x - 1) / threadsPerBlock.x, (cols + threadsPerBlock.y - 1) / threadsPerBlock.y);

    applyReLU_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, rows, cols);



    // Copy output data from device to host

    cudaMemcpy(temp_output, d_output, rows * sizeof(float*), cudaMemcpyDeviceToHost);

    for (int i = 0; i < rows; ++i) {

        cudaMemcpy(output[i].data(), temp_output[i], cols * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(temp_input[i]);

        cudaFree(temp_output[i]);

    }



    // Free device memory

    cudaFree(d_input);

    cudaFree(d_output);

}



void applyTanh(const vector<vector<float > >& input, vector<vector<float > >& output) {

    int rows = input.size();

    int cols = input[0].size();



    // Allocate memory on GPU

    float **d_input, **d_output;

    cudaMalloc((void**)&d_input, rows * sizeof(float*));

    cudaMalloc((void**)&d_output, rows * sizeof(float*));



    float *temp_input[rows], *temp_output[rows];

    for (int i = 0; i < rows; ++i) {

        cudaMalloc((void**)&temp_input[i], cols * sizeof(float));

        cudaMalloc((void**)&temp_output[i], cols * sizeof(float));

        cudaMemcpy(temp_input[i], input[i].data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    }



    cudaMemcpy(d_input, temp_input, rows * sizeof(float*), cudaMemcpyHostToDevice);

    cudaMemcpy(d_output, temp_output, rows * sizeof(float*), cudaMemcpyHostToDevice);



    // Launch kernel

    dim3 threadsPerBlock(16, 16);

    dim3 numBlocks((rows + threadsPerBlock.x - 1) / threadsPerBlock.x, (cols + threadsPerBlock.y - 1) / threadsPerBlock.y);

    applyTanh_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, rows, cols);



    // Copy output data from device to host

    cudaMemcpy(temp_output, d_output, rows * sizeof(float*), cudaMemcpyDeviceToHost);

    for (int i = 0; i < rows; ++i) {

        cudaMemcpy(output[i].data(), temp_output[i], cols * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(temp_input[i]);

        cudaFree(temp_output[i]);

    }



    // Free device memory

    cudaFree(d_input);

    cudaFree(d_output);

}


// Function to print a matrix
void printMatrix(const vector<vector<float> >& matrix) {
    for(int i=0;i<matrix.size();i++){
        for(int j=0;j<matrix[0].size();j++){
            cout<<matrix[i][j]<<" ";
        }
        cout<<endl;
    }
}

int main(int argv, char* argc[]){
    //cout<<"Input done\n"<<atoi(argc[24])<<endl;
    int task = atos(argc[1]);
    
    //cout<<"Input done\n";
    if(task==1){
        //cout<<"Input done\n";
        int n=atos(argc[2]);
        int m=atos(argc[3]);
        int p=atos(argc[4]);
        //cout<<"Input done\n";
        vector<vector<float> >input(n,vector<float>(n,0.0));
        vector<vector<float> >kernel(m,vector<float>(m,0.0));
        //cout<<"Input done\n";
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                input[i][j]=atos(argc[5+i*n+j]);
            }
        }
        //cout<<"Input done\n";
        for(int i=0;i<m;i++){
            for(int j=0;j<m;j++){
                //cout<<"5+n*n+i "<<5+n*n+i*m+j<<" "<< atoi(argc[5+n*n+i*m+j])<<endl;
                kernel[i][j]=atos(argc[5+n*n+i*m+j]);
            }
        }

        //cout<<"Input done\n";
        vector<vector<float> > output = convolutionWithPadding(input,kernel,p);
        printMatrix(output);
        
   
    }
    else if(task==2){
        int sub_task = atos(argc[2]);
        int rows = atos(argc[3]);
        int cols = atos(argc[4]);
        vector<vector<float> >input(rows,vector<float>(cols,0.0));
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                input[i][j]=atos(argc[5+i*cols+j]);
            }
        }
        if(sub_task==0){
            vector<vector<float> > reluResult(input.size());
            for (size_t i = 0; i < input.size(); ++i) {
        reluResult[i].resize(input[i].size());
        }
            applyReLU(input, reluResult);
            printMatrix(reluResult);
        }
        else if(sub_task==1){
            vector<vector<float> > tanhResult(input.size());
            for (size_t i = 0; i < input.size(); ++i) {
                tanhResult[i].resize(input[i].size());
    }
        applyTanh(input, tanhResult);
        printMatrix(tanhResult);
        }
    }
    else if(task==3){
        int sub_task = atos(argc[2]);
        int poolSize = atos(argc[3]);
        int n = atos(argc[4]);
        vector<vector<float> >inputMatrix(n,vector<float>(n,0.0));
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                inputMatrix[i][j]=atos(argc[5+i*n+j]);
            }
        }
	int rows = inputMatrix.size();
    int cols = inputMatrix[0].size();
    vector<float> flatInput(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flatInput[i * cols + j] = inputMatrix[i][j];
        }
    }
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_output, (rows / poolSize) * (cols / poolSize) * sizeof(float));
    cudaMemcpy(d_input, flatInput.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);
        if(sub_task==0){
            //vector<vector<float>> output;
	    // Pool size
    //int poolSize = 2;

    // Allocate memory for input and output matrices on the GPU
    

    // Launch max pooling kernel
    maxPooling<<<gridDim, blockDim>>>(d_input, d_output, rows, cols, poolSize);
    // averagePooling<<<gridDim, blockDim>>>(d_input, d_output, rows, cols, poolSize);
    // Copy the result back to the host
    vector<vector<float> > maxPoolResult(rows / poolSize, vector<float>(cols / poolSize));
    vector<float> flatOutput((rows / poolSize) * (cols / poolSize));
    cudaMemcpy(flatOutput.data(), d_output, (rows / poolSize) * (cols / poolSize) * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < rows / poolSize; ++i) {
        for (int j = 0; j < cols / poolSize; ++j) {
            maxPoolResult[i][j] = flatOutput[i * (cols / poolSize) + j];
        }
    }
	    //maxPooling(input,output,pool_size);
            printMatrix(maxPoolResult);
        }
        else if(sub_task==1){
            //vector<vector<float>> output = averagePooling(input,pool_size);
	    averagePooling<<<gridDim, blockDim>>>(d_input, d_output, rows, cols, poolSize);

    // Copy the result back to the host
    vector<vector<float> > avgPoolResult(rows / poolSize, vector<float>(cols / poolSize));
    vector<float> flatOutput((rows / poolSize) * (cols / poolSize));
    cudaMemcpy(flatOutput.data(), d_output, (rows / poolSize) * (cols / poolSize) * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < rows / poolSize; ++i) {
        for (int j = 0; j < cols / poolSize; ++j) {
            avgPoolResult[i][j] = flatOutput[i * (cols / poolSize) + j];
        }
    }
            printMatrix(avgPoolResult);
        }
    }
    else if(task==4){
        int sub_task = atos(argc[2]);
        int n = atos(argc[3]);
        vector<vector<float> >input(1,vector<float>(n,0.0));
        for(int i=0;i<1;i++){
            for(int j=0;j<n;j++){
                input[i][j]=atos(argc[4+i*n+j]);
            }
        }
        if(sub_task==0){
            // float* in1 = new float(n);
            // for(int i=0;i<n;i++){
            //     in1[i]=input[1][i];
            // }
            float* output = sigmoid(input[0]);
            for(int i=0; i<n;i++){
                cout<<output[i]<<" ";
            }
            cout<<endl;
        }
        else if(sub_task==1){
            // float* in1 = new float(n);
            // for(int i=0;i<n;i++){
            //     in1[i]=input[1][i];
            // }
            float* output = softmax(input[0]);
            for(int i=0; i<n;i++){
                cout<<output[i]<<" ";
            }
            cout<<endl;
    
        }
    }
    else{
        cout<<"Invalid task"<<endl;
    }
}












