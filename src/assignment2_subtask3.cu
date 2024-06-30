#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <ctime>
#include <dirent.h>
#include "relu_new.cu"
// #include "softmax_cuda.cu"
#include "pooling_new.cu"
#include "convolution_new.cu"

using namespace std;


void loadWeights(const string& filename, int num_input_channels, int num_output_channels, int num_filters, int kernel_size, float* input_kernels, float* biases) {
    ifstream file(filename.c_str());
    float val;
    int kernel_index = 0;
    int bias_index = 0;
    while (file >> val) {
        if (kernel_index < num_output_channels * num_input_channels * kernel_size * kernel_size) {
            input_kernels[kernel_index++] = val;
        } else {
            biases[bias_index++] = val;
        }
    }
}

__global__ void softmax(float* input, float* probabilities, float sum, int rows, int cols, int start) {
    // int k = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row<rows && col<cols) {
        probabilities[start*10 + rows*row*cols + col] = exp(input[rows*row*cols + col]) / (sum);
    }
}

string intToString(int value) {
    stringstream ss;
    ss << value;
    return ss.str();
}

void writeProbabilitiesToFile(float* probabilities, int num_images, vector<string> fileNames) {
    
    string outputFolderPath = "../output/";
    
    
    for (int c=0; c<num_images; c++) {
        string inputFileName = fileNames[c] + ".txt";;
    
        string outputFilePath = outputFolderPath + inputFileName;
    
        // Open the output file
        std::ofstream outputFile(outputFilePath.c_str());
        if (!outputFile.is_open()) {
            std::cerr << "Failed to open output file: " << outputFilePath << std::endl;
            return;
        }
        
        vector<float> temp;
        
        for (int i=0; i<10; i++) {
            temp.push_back(probabilities[c*10 + i]);
        }
        
        vector<float> sortedProbabilities = temp;
        sort(sortedProbabilities.begin(), sortedProbabilities.end(), greater<float>());

        
        for (int i = 0; i < 5; ++i) {
            float probability = sortedProbabilities[i];
            int classNumber = find(temp.begin(), temp.end(), probability) - temp.begin();
            
            std::stringstream ss;
            ss << fixed << probability*100 << " class " << classNumber << endl;
            outputFile << ss.str();
        }
    
        outputFile.close();
    }
    

    // cout << "Probabilities written to file" << endl;
}


void read_filenames(const char* folderPath, vector<string>& fileNames) {


    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(folderPath)) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            // Check if the file name ends with ".txt"
            if (strlen(ent->d_name) > 4 && std::string(ent->d_name).substr(strlen(ent->d_name) - 4) == ".txt") {
                // Add the file name without the extension to the vector
                fileNames.push_back(std::string(ent->d_name).substr(0, strlen(ent->d_name) - 4));
            }
        }
        closedir(dir);
    } else {
        // Error opening directory
        std::cerr << "Error opening directory" << std::endl;
        return;
    }

}


int main() {

    clock_t start = clock();
    
    float* conv1Kernels = new float[20*1*5*5];
    float* conv1Biases = new float[20];
    // Input channels 1
    // Output channels 20
    loadWeights("../weights/conv1.txt", 1, 20, 20, 5, conv1Kernels, conv1Biases);
    
    /* for (int i=0; i<20; i++) {
        for (int j=0; j<5; j++) {
            for (int k=0; k<5; k++) {
                cout << conv1Kernels[i*5*5 + j*5 + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << endl; */
    
    float* conv2Kernels = new float[50*20*5*5];
    float* conv2Biases = new float[50];
    // Input channels 20
    // Output channels 50 
    loadWeights("../weights/conv2.txt", 20, 50, 50, 5, conv2Kernels, conv2Biases);
    
    float* fc1Kernels = new float[500*50*4*4];
    float* fc1Biases = new float[500];
    // Input channels 50
    // Output channels 500 
    loadWeights("../weights/fc1.txt", 50, 500, 500, 4, fc1Kernels, fc1Biases);
    
    float* fc2Kernels = new float[10*500*1*1];
    float* fc2Biases = new float[10];
    // Input channels 500
    // Output channels 10 
    loadWeights("../weights/fc2.txt", 500, 10, 10, 1, fc2Kernels, fc2Biases);
    
    
    
    const char* folderPath = "../pre-proc-img";
    vector<string> filenames;
    read_filenames(folderPath, filenames);
    // cout << "Weights Loaded\n";
    
    int num_images = filenames.size();
    // cout<<"num of images is "<<num_images<<endl;
    
    // CONVOLUTION 1
        
    int input_channels = 1;
    int inputSize = 28;
    int numKernels = 20;
    int kernelSize = 5;
    int outputSize = inputSize - kernelSize + 1;
    
    float *c1_kernels, *c1_biases;
    cudaMalloc(&c1_kernels, numKernels * input_channels * kernelSize * kernelSize * sizeof(float));
    cudaMalloc(&c1_biases, numKernels * sizeof(float));
    
    cudaMemcpy(c1_kernels, conv1Kernels, numKernels * input_channels * kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c1_biases, conv1Biases, numKernels * sizeof(float), cudaMemcpyHostToDevice);
    
    // CONVOLUTION 2
    
    input_channels = 20;
    inputSize = 12;
    numKernels = 50;
    kernelSize = 5;
    outputSize = inputSize -kernelSize+1;
    
    float *c2_kernels, *c2_biases;
    cudaMalloc(&c2_kernels, numKernels * input_channels * kernelSize * kernelSize * sizeof(float));
    cudaMalloc(&c2_biases, numKernels * sizeof(float));

    cudaMemcpy(c2_kernels, conv2Kernels, numKernels * input_channels * kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c2_biases, conv2Biases, numKernels * sizeof(float), cudaMemcpyHostToDevice);
    
    // FC 1
    
    input_channels = 50;
    inputSize = 4;
    numKernels = 500;
    kernelSize = 4;
    outputSize = inputSize -kernelSize+1;
    
    float *fc1_kernels, *fc1_biases;
    cudaMalloc(&fc1_kernels, numKernels * input_channels * kernelSize * kernelSize * sizeof(float));
    cudaMalloc(&fc1_biases, numKernels * sizeof(float));

    cudaMemcpy(fc1_kernels, fc1Kernels, numKernels * input_channels * kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(fc1_biases, fc1Biases, numKernels * sizeof(float), cudaMemcpyHostToDevice);
    
    // FC 2
    
    input_channels = 500;
    inputSize = 1;
    numKernels = 10;
    kernelSize = 1;
    outputSize = inputSize -kernelSize+1;
    
    float *fc2_kernels, *fc2_biases;
    cudaMalloc(&fc2_kernels, numKernels * input_channels * kernelSize * kernelSize * sizeof(float));
    cudaMalloc(&fc2_biases, numKernels * sizeof(float));

    cudaMemcpy(fc2_kernels, fc2Kernels, numKernels * input_channels * kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(fc2_biases, fc2Biases, numKernels * sizeof(float), cudaMemcpyHostToDevice);
    
    // SOFTMAX output
    
    float* probabilities;
    cudaMalloc(&probabilities, num_images * 10 * sizeof(float));
    
    string filepath_prefix = "../pre-proc-img/";
    
    for (int c = 0; c < num_images; ++c) {
    
        // string str = filepath_prefix + intToString(c) + ".txt";
        
        string str = filepath_prefix + filenames[c] + ".txt";
      
        ifstream file(str.c_str());
        // cout << str << endl;
        
        float val;
        
        int idx=0;
        float* input_matrix = new float[28*28];
        
        while (file >> val) {
            input_matrix[idx++] = val;
        }
        
        
        
        
        float *d_input, *c1_output, *p1_output, *c2_output, *p2_output, *fc1_output, *r_output, *fc2_output;
        
        // CONVOLUTION 1
        
        input_channels = 1;
        inputSize = 28;
        numKernels = 20;
        kernelSize = 5;
        outputSize = inputSize - kernelSize + 1;
        
        cudaMalloc(&d_input, input_channels * inputSize * inputSize * sizeof(float));
        cudaMalloc(&c1_output, numKernels * outputSize * outputSize * sizeof(float));
        
        cudaMemcpy(d_input, input_matrix, input_channels * inputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);
        
        
        
        delete[] input_matrix;
        
        // POOLING 1
    
        input_channels = 20;
        inputSize = 24;
        numKernels = 20;
        int poolSize = 2;
        outputSize = inputSize / poolSize;

        cudaMalloc(&p1_output, input_channels * outputSize * outputSize * sizeof(float));
        
        // CONVOLUTION 2
        
        input_channels = 20;
        inputSize = 12;
        numKernels = 50;
        kernelSize = 5;
        outputSize = inputSize -kernelSize+1;
        
        cudaMalloc(&c2_output, numKernels * outputSize * outputSize * sizeof(float));
        
        // POOLING 2
        
        input_channels = 50;
        inputSize = 8;
        numKernels = 50;
        poolSize = 2;
        outputSize = inputSize / poolSize;
        
        cudaMalloc(&p2_output, input_channels * outputSize * outputSize * sizeof(float));
        
        // FC 1
        
        input_channels = 50;
        inputSize = 4;
        numKernels = 500;
        kernelSize = 4;
        outputSize = inputSize -kernelSize+1;
        
        cudaMalloc(&fc1_output, numKernels * outputSize * outputSize * sizeof(float));
        
        // ReLU Layer
        
        input_channels = 500;
        int rows = 1;
        int cols = 1;
        
        cudaMalloc(&r_output, input_channels * rows * cols * sizeof(float));
        
        // FC 2
        
        input_channels = 500;
        inputSize = 1;
        numKernels = 10;
        kernelSize = 1;
        outputSize = inputSize -kernelSize+1;
        
        cudaMalloc(&fc2_output, numKernels * outputSize * outputSize * sizeof(float));
        
        // SOFTMAX
        
        // cudaMalloc(&probabilities, 10 * sizeof(float));
        
        // MALLOC DONE
        
        // EXECUTION STARTS
        
        // CONVOLUTION 1
        
        input_channels = 1;
        inputSize = 28;
        numKernels = 20;
        kernelSize = 5;
        outputSize = inputSize - kernelSize + 1;
        
        dim3 threadsPerBlock(16, 16, 1);
        dim3 numBlocks((outputSize + threadsPerBlock.x - 1) / threadsPerBlock.x, (outputSize + threadsPerBlock.y - 1) / threadsPerBlock.y, numKernels);
    
    
        convolutionWithoutPaddingKernel<<<numBlocks, threadsPerBlock>>>(d_input, c1_kernels, c1_biases, c1_output, input_channels, inputSize, kernelSize, numKernels);
        
    
        cudaFree(d_input);
        
        // POOLING 1
    
        input_channels = 20;
        inputSize = 24;
        numKernels = 20;
        poolSize = 2;
        outputSize = inputSize / poolSize;
        
        dim3 threadsPerBlock1(16, 16, 1);
        dim3 numBlocks1((outputSize + threadsPerBlock1.x - 1) / threadsPerBlock1.x, (outputSize + threadsPerBlock1.y - 1) / threadsPerBlock1.y, input_channels);
    
    
        maxPoolingKernel<<<numBlocks1, threadsPerBlock1>>>(c1_output, p1_output, input_channels, inputSize, poolSize);
        
        /*float* temp1 = new float[numKernels * outputSize * outputSize];
            
            cudaMemcpyAsync(temp1, p1_output, numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
            
            cout << "Printing pool output:\n";
          
            for (int i=0; i<1; i++) {
                for (int j=0; j<outputSize; j++) {
                    for (int k=0; k<outputSize; k++) {
                        cout << temp1[j*outputSize + k];
                    }
                    cout << endl;
                }
                cout << endl;
            }
            
            cout << endl;
            delete[] temp1;*/
    
        
        cudaFree(c1_output);
        
        // CONVOLUTION 2
        
        input_channels = 20;
        inputSize = 12;
        numKernels = 50;
        kernelSize = 5;
        outputSize = inputSize -kernelSize+1;
    
        dim3 threadsPerBlock2(16, 16, 1);
        dim3 numBlocks2((outputSize + threadsPerBlock2.x - 1) / threadsPerBlock2.x, (outputSize + threadsPerBlock2.y - 1) / threadsPerBlock2.y, numKernels);
    
    
        convolutionWithoutPaddingKernel<<<numBlocks2, threadsPerBlock2>>>(p1_output, c2_kernels, c2_biases, c2_output, input_channels, inputSize, kernelSize, numKernels);
        
        /*float* temp1 = new float[numKernels * outputSize * outputSize];
            
            cudaMemcpyAsync(temp1, c2_output, numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
            
            cout << "Printing c2 output:\n";
          
            for (int i=0; i<1; i++) {
                for (int j=0; j<outputSize; j++) {
                    for (int k=0; k<outputSize; k++) {
                        cout << temp1[j*outputSize + k];
                    }
                    cout << endl;
                }
                cout << endl;
            }
            
            cout << endl;
            delete[] temp1;*/
        
        cudaFree(p1_output);
        
        // POOLING 2
        
        input_channels = 50;
        inputSize = 8;
        numKernels = 50;
        poolSize = 2;
        outputSize = inputSize / poolSize;
    
        dim3 threadsPerBlock3(16, 16, 1);
        dim3 numBlocks3((outputSize + threadsPerBlock3.x - 1) / threadsPerBlock3.x, (outputSize + threadsPerBlock3.y - 1) / threadsPerBlock3.y, input_channels);
    
    
        maxPoolingKernel<<<numBlocks3, threadsPerBlock3>>>(c2_output, p2_output, input_channels, inputSize, poolSize);
        /*float* temp1 = new float[numKernels * outputSize * outputSize];
            
            cudaMemcpyAsync(temp1, p2_output, numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
            
            cout << "Printing c2 output:\n";
          
            for (int i=0; i<1; i++) {
                for (int j=0; j<outputSize; j++) {
                    for (int k=0; k<outputSize; k++) {
                        cout << temp1[j*outputSize + k];
                    }
                    cout << endl;
                }
                cout << endl;
            }
            
            cout << endl;
            delete[] temp1*/;

        cudaFree(c2_output);
        
        // FC 1
        
        input_channels = 50;
        inputSize = 4;
        numKernels = 500;
        kernelSize = 4;
        outputSize = inputSize -kernelSize+1;
    
        dim3 threadsPerBlock4(16, 16, 1);
        dim3 numBlocks4((outputSize + threadsPerBlock4.x - 1) / threadsPerBlock4.x, (outputSize + threadsPerBlock4.y - 1) / threadsPerBlock4.y, numKernels);
    
    
        convolutionWithoutPaddingKernel<<<numBlocks4, threadsPerBlock4>>>(p2_output, fc1_kernels, fc1_biases, fc1_output, input_channels, inputSize, kernelSize, numKernels);
        
        /*float* temp1 = new float[numKernels * outputSize * outputSize];
            
            cudaMemcpyAsync(temp1, fc1_output, numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
            
            cout << "Printing fc1 output:\n";
          
            for (int i=0; i<1; i++) {
                for (int j=0; j<outputSize; j++) {
                    for (int k=0; k<outputSize; k++) {
                        cout << temp1[j*outputSize + k];
                    }
                    cout << endl;
                }
                cout << endl;
            }
            
            cout << endl;
            delete[] temp1;*/

        cudaFree(p2_output);
        
        // ReLU Layer
        
        input_channels = 500;
        rows = 1;
        cols = 1;
    
        dim3 threadsPerBlock5(16, 16);
        dim3 numBlocks5((rows + threadsPerBlock5.x - 1) / threadsPerBlock5.x, (cols + threadsPerBlock5.y - 1) / threadsPerBlock5.y, input_channels);
    
    
        applyReLU_kernel<<<numBlocks5, threadsPerBlock5>>>(fc1_output, r_output, rows, cols);
        
        /*float* temp1 = new float[numKernels * outputSize * outputSize];
            
            cudaMemcpyAsync(temp1, r_output, numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
            
            cout << "Printing relu output:\n";
          
            for (int i=0; i<1; i++) {
                for (int j=0; j<outputSize; j++) {
                    for (int k=0; k<outputSize; k++) {
                        cout << temp1[j*outputSize + k];
                    }
                    cout << endl;
                }
                cout << endl;
            }
            
            cout << endl;
            delete[] temp1;*/

        cudaFree(fc1_output);
        
        // FC 2
        
        input_channels = 500;
        inputSize = 1;
        numKernels = 10;
        kernelSize = 1;
        outputSize = inputSize -kernelSize+1;
    
        dim3 threadsPerBlock6(16, 16, 1);
        dim3 numBlocks6((outputSize + threadsPerBlock6.x - 1) / threadsPerBlock6.x, (outputSize + threadsPerBlock6.y - 1) / threadsPerBlock6.y, numKernels);
    
    
        convolutionWithoutPaddingKernel<<<numBlocks6, threadsPerBlock6>>>(r_output, fc2_kernels, fc2_biases, fc2_output, input_channels, inputSize, kernelSize, numKernels);
        
        /*float* temp1 = new float[numKernels * outputSize * outputSize];
            
            cudaMemcpyAsync(temp1, fc2_output, numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
            
            cout << "Printing fc2 output:\n";
          
            for (int i=0; i<1; i++) {
                for (int j=0; j<outputSize; j++) {
                    for (int k=0; k<outputSize; k++) {
                        cout << temp1[j*outputSize + k];
                    }
                    cout << endl;
                }
                cout << endl;
            }
            
            cout << endl;
            delete[] temp1;*/
        
        cudaFree(r_output);
        
        // SOFTMAX
        
        float* temp = new float[10*1*1];
        cudaMemcpy(temp, fc2_output, numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
        
        float sum = 0.0f;
        
        for (int i=0; i<10; i++) {
            sum += exp(temp[i]);
            //cout<<temp[i]<<" ";
        }
        
        //cout<<"\nsum is "<< sum <<"\n";
        
        dim3 threadsPerBlock7(16, 16);
        dim3 numBlocks7((1 + threadsPerBlock7.x - 1) / threadsPerBlock7.x, (10 + threadsPerBlock7.y - 1) / threadsPerBlock7.y);
    
    
        softmax<<<numBlocks7, threadsPerBlock7>>>(fc2_output, probabilities, sum, 1, 10, c);
        /*float* temp1 = new float[numKernels * outputSize * outputSize];
            
            cudaMemcpyAsync(temp1, fc2_output, numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
            
            cout << "Printing fc2 output:\n";
          
            for (int i=0; i<1; i++) {
                for (int j=0; j<outputSize; j++) {
                    for (int k=0; k<outputSize; k++) {
                        cout << temp1[j*outputSize + k];
                    }
                    cout << endl;
                }
                cout << endl;
            }
            
            cout << endl;
            delete[] temp1;*/
        // probabilities += 10;
        
        cudaFree(fc2_output);
        
        
    }
    
    
    float* output_prob = new float[num_images*10];
    
    cudaMemcpyAsync(output_prob, probabilities, num_images * 10 * sizeof(float), cudaMemcpyDeviceToHost);
    
    clock_t end = clock();
    
    /* cout << "Printing output:\n";
    
    for (int i=0; i<2; i++) {
        for (int j=0; j<10; j++) {
            cout << output_prob[i*10 + j] << " ";
        }
        cout << endl;
    }
    
    cout << endl; */
    
    
    double elapsed_time = double(end - start) / CLOCKS_PER_SEC;

    cout << "Time taken by LeNet5 function: " << elapsed_time * 1000 << " milliseconds" << endl;
    writeProbabilitiesToFile(output_prob, num_images, filenames);
    // cout << "Done\n";
    
    delete[] output_prob;

    return 0;
}