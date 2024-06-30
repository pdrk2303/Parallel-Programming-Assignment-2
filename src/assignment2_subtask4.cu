#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <dirent.h>
#include "relu_new.cu"
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


int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [1 - with streams, 0 - without streams]" << std::endl;
        return 1;
    }
    
    int flag = atoi(argv[1]);
    // cout << flag << endl;

    clock_t start = clock();
    
        float* conv1Kernels = new float[20*1*5*5];
    float* conv1Biases = new float[20];
    // Input channels 1
    // Output channels 20
    loadWeights("../weights/conv1.txt", 1, 20, 20, 5, conv1Kernels, conv1Biases);
    
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
    
    // cout << "Weights Loaded\n";
    
    const char* folderPath = "../pre-proc-img";
    vector<string> filenames;
    read_filenames(folderPath, filenames);
    
    
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
    
    int num_images = filenames.size();
    
    float* probabilities;
    cudaMalloc(&probabilities, num_images * 10 * sizeof(float));
    
    vector<float*> all_input_matrix;
    
    
    string filepath_prefix = "../pre-proc-img/";
    
    for (int c = 0; c < num_images; ++c) {
    
        // stringstream ss;
        // ss << "../pre-proc-img/matrix_" << c << ".txt";
        // string str = ss.str();
        
        string str = filepath_prefix + filenames[c] + ".txt";
      
        ifstream file(str.c_str());
        // cout << str << endl;
        
        float val;
        
        int idx=0;
        float* input_matrix = new float[28*28];
        
        while (file >> val) {
            input_matrix[idx++] = val;
        }
        
        // all_input_matrix.push_back(input_matrix);
        /* for (int i=0; i<28; i++) {
            for (int j=0; j<28; j++) {
                cout << input_matrix[i*28 + j] << " ";
            }
            cout << endl;
        }
        cout << endl; */
        
        
        if (flag == 1) {
            all_input_matrix.push_back(input_matrix);
            // delete[] input_matrix;
            continue;
        } 
        else if (flag == 0) {
        
            float *d_input, *c1_output, *p1_output, *c2_output, *p2_output, *fc1_output, *r_output, *fc2_output;
        
            // CONVOLUTION 1
            
            input_channels = 1;
            inputSize = 28;
            numKernels = 20;
            kernelSize = 5;
            outputSize = inputSize - kernelSize + 1;
            
            cudaMalloc(&d_input, input_channels * inputSize * inputSize * sizeof(float));
            cudaMalloc(&c1_output, numKernels * outputSize * outputSize * sizeof(float));
            
            cudaMemcpyAsync(d_input, input_matrix, input_channels * inputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);
            
            
            
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

    
            cudaFree(p2_output);
            
            // ReLU Layer
            
            input_channels = 500;
            rows = 1;
            cols = 1;
        
            dim3 threadsPerBlock5(16, 16);
            dim3 numBlocks5((rows + threadsPerBlock5.x - 1) / threadsPerBlock5.x, (cols + threadsPerBlock5.y - 1) / threadsPerBlock5.y, input_channels);
        
        
            applyReLU_kernel<<<numBlocks5, threadsPerBlock5>>>(fc1_output, r_output, rows, cols);
            

    
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
            

            
            cudaFree(r_output);
            
            // SOFTMAX
            
            float* temp = new float[10*1*1];
            cudaMemcpy(temp, fc2_output, numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
            
            float sum = 0.0f;
            
            for (int i=0; i<10; i++) {
                sum += exp(temp[i]);
                //cout<<temp[i]<<" ";
            }
            
            
            dim3 threadsPerBlock7(16, 16);
            dim3 numBlocks7((1 + threadsPerBlock7.x - 1) / threadsPerBlock7.x, (10 + threadsPerBlock7.y - 1) / threadsPerBlock7.y);
        
        
            softmax<<<numBlocks7, threadsPerBlock7>>>(fc2_output, probabilities, sum, 1, 10, c);

            
            cudaFree(fc2_output);
        }
        
    }
    
    if (flag == 1) {
        const int num_streams = min(25, num_images);  // Using 8 streams
    
        cudaStream_t streams[num_streams];
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamCreate(&streams[i]);
        }
        
        int batch_num = 0;
        
        // cout<<"Num streams is "<<num_streams<<endl;
        
        for(int c=0; c<num_images; c+=num_streams){
            //int stream_idx = c % num_streams; 
            
            vector<float*> all_d_input(num_streams);
            vector<float*> all_c1_output(num_streams);
            vector<float*> all_p1_output(num_streams);
            vector<float*> all_c2_output(num_streams);
            vector<float*> all_p2_output(num_streams);
            vector<float*> all_fc1_output(num_streams);
            vector<float*> all_r_output(num_streams);
            vector<float*> all_fc2_output(num_streams);
            //float *d_input, *c1_output, *p1_output, *c2_output, *p2_output, *fc1_output, *r_output, *fc2_output;
            
            // CONVOLUTION 1
            
            input_channels = 1;
            inputSize = 28;
            numKernels = 20;
            kernelSize = 5;
            outputSize = inputSize - kernelSize + 1;
            
            for(int i=0;i<num_streams;i++) cudaMalloc(&all_d_input[i], input_channels * inputSize * inputSize * sizeof(float));
            for(int i=0;i<num_streams;i++) cudaMalloc(&all_c1_output[i], numKernels * outputSize * outputSize * sizeof(float));
            
            for(int i=0;i<num_streams;i++) {
                cudaMemcpyAsync(all_d_input[i], all_input_matrix[c+i], input_channels * inputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);
                
            }
            
            
            //delete[] input_matrix;
            
            // POOLING 1
        
            input_channels = 20;
            inputSize = 24;
            numKernels = 20;
            int poolSize = 2;
            outputSize = inputSize / poolSize;
    
            for(int i=0;i<num_streams;i++)cudaMalloc(&all_p1_output[i], input_channels * outputSize * outputSize * sizeof(float));
            
            // CONVOLUTION 2
            
            input_channels = 20;
            inputSize = 12;
            numKernels = 50;
            kernelSize = 5;
            outputSize = inputSize -kernelSize+1;
            
            for(int i=0;i<num_streams;i++)cudaMalloc(&all_c2_output[i], numKernels * outputSize * outputSize * sizeof(float));
            
            // POOLING 2
            
            input_channels = 50;
            inputSize = 8;
            numKernels = 50;
            poolSize = 2;
            outputSize = inputSize / poolSize;
            
            for(int i=0;i<num_streams;i++)cudaMalloc(&all_p2_output[i], input_channels * outputSize * outputSize * sizeof(float));
            
            // FC 1
            
            input_channels = 50;
            inputSize = 4;
            numKernels = 500;
            kernelSize = 4;
            outputSize = inputSize -kernelSize+1;
            
            for(int i=0;i<num_streams;i++)cudaMalloc(&all_fc1_output[i], numKernels * outputSize * outputSize * sizeof(float));
            
            // ReLU Layer
            
            input_channels = 500;
            int rows = 1;
            int cols = 1;
            
            for(int i=0;i<num_streams;i++)cudaMalloc(&all_r_output[i], input_channels * rows * cols * sizeof(float));
            
            // FC 2
            
            input_channels = 500;
            inputSize = 1;
            numKernels = 10;
            kernelSize = 1;
            outputSize = inputSize -kernelSize+1;
            
            for(int i=0;i<num_streams;i++)cudaMalloc(&all_fc2_output[i], numKernels * outputSize * outputSize * sizeof(float));
            
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
        
        
            for(int i=0;i<num_streams;i++){
                convolutionWithoutPaddingKernel<<<numBlocks, threadsPerBlock, 0, streams[i]>>>(all_d_input[i], c1_kernels, c1_biases, all_c1_output[i], input_channels, inputSize, kernelSize, numKernels);
             
                /*float* temp = new float[numKernels * outputSize * outputSize];
                
                cudaMemcpyAsync(temp, all_c1_output[i], numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
                
                cout << "Printing c1 output:\n";
              
                for (int i=0; i<1; i++) {
                    for (int j=0; j<outputSize; j++) {
                        for (int k=0; k<outputSize; k++) {
                            cout << temp[j*outputSize + k] ;
                        }
                        cout << endl;
                    }
                    cout << endl;
                }
                
                cout << endl;
                delete[] temp;*/
                
            }
        
            //cudaFree(all_d_input);
            for (int i = 0; i < num_streams; ++i) {
                cudaFree(all_d_input[i]);
            }
            all_d_input.clear();
            
            /* if(c==1){
              //Printing CONV1 output:
              float* output_conv1 = new float[ numKernels * outputSize * outputSize];
              
              cudaMemcpyAsync(output_conv1, all_c1_output[0],  numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
              
              cout << "Printing CONV1 output:\n";
              
              for (int i=0; i<outputSize; i++) {
                  for (int j=0; j<outputSize; j++) {
                      cout << output_conv1[i*outputSize + j] << " ";
                  }
                  cout << endl;
              }
              
              cout << endl;
            } */
                   
            // POOLING 1
        
            input_channels = 20;
            inputSize = 24;
            numKernels = 20;
            poolSize = 2;
            outputSize = inputSize / poolSize;
            
            dim3 threadsPerBlock1(16, 16, 1);
            dim3 numBlocks1((outputSize + threadsPerBlock1.x - 1) / threadsPerBlock1.x, (outputSize + threadsPerBlock1.y - 1) / threadsPerBlock1.y, input_channels);
        
        
            for(int i=0;i<num_streams;i++){maxPoolingKernel<<<numBlocks1, threadsPerBlock1, 0, streams[i]>>>(all_c1_output[i], all_p1_output[i], input_channels, inputSize, poolSize);
            /*float* temp = new float[numKernels * outputSize * outputSize];
                
                cudaMemcpyAsync(temp, all_p1_output[i], numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
                
                cout << "Printing pool output:\n";
              
                for (int i=0; i<1; i++) {
                    for (int j=0; j<outputSize; j++) {
                        for (int k=0; k<outputSize; k++) {
                            cout << temp[j*outputSize + k] ;
                        }
                        cout << endl;
                    }
                    cout << endl;
                }
                
                cout << endl;
                delete[] temp;*/
            }
            //cudaFree(c1_output);
            for (int i = 0; i < num_streams; ++i) {
                cudaFree(all_c1_output[i]);
            }
            all_c1_output.clear();
            // CONVOLUTION 2
            
            input_channels = 20;
            inputSize = 12;
            numKernels = 50;
            kernelSize = 5;
            outputSize = inputSize -kernelSize+1;
        
            dim3 threadsPerBlock2(16, 16, 1);
            dim3 numBlocks2((outputSize + threadsPerBlock2.x - 1) / threadsPerBlock2.x, (outputSize + threadsPerBlock2.y - 1) / threadsPerBlock2.y, numKernels);
        
        
            for(int i=0;i<num_streams;i++){convolutionWithoutPaddingKernel<<<numBlocks2, threadsPerBlock2, 0, streams[i]>>>(all_p1_output[i], c2_kernels, c2_biases, all_c2_output[i], input_channels, inputSize, kernelSize, numKernels);
            
            /*float* temp = new float[numKernels * outputSize * outputSize];
                
                cudaMemcpyAsync(temp, all_c2_output[i], numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
                
                cout << "Printing c2 output:\n";
              
                for (int i=0; i<1; i++) {
                    for (int j=0; j<outputSize; j++) {
                        for (int k=0; k<outputSize; k++) {
                            cout << temp[j*outputSize + k] ;
                        }
                        cout << endl;
                    }
                    cout << endl;
                }
                
                cout << endl;
                delete[] temp;*/
            }
    
            //cudaFree(p1_output);
            for (int i = 0; i < num_streams; ++i) {
                cudaFree(all_p1_output[i]);
            }
            all_p1_output.clear();
            // POOLING 2
            
            input_channels = 50;
            inputSize = 8;
            numKernels = 50;
            poolSize = 2;
            outputSize = inputSize / poolSize;
        
            dim3 threadsPerBlock3(16, 16, 1);
            dim3 numBlocks3((outputSize + threadsPerBlock3.x - 1) / threadsPerBlock3.x, (outputSize + threadsPerBlock3.y - 1) / threadsPerBlock3.y, input_channels);
        
        
            for(int i=0;i<num_streams;i++){maxPoolingKernel<<<numBlocks3, threadsPerBlock3, 0, streams[i]>>>(all_c2_output[i], all_p2_output[i], input_channels, inputSize, poolSize);
            /*float* temp = new float[numKernels * outputSize * outputSize];
                
                cudaMemcpyAsync(temp, all_p2_output[i], numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
                
                cout << "Printing c2 output:\n";
              
                for (int i=0; i<1; i++) {
                    for (int j=0; j<outputSize; j++) {
                        for (int k=0; k<outputSize; k++) {
                            cout << temp[j*outputSize + k] ;
                        }
                        cout << endl;
                    }
                    cout << endl;
                }
                
                cout << endl;
                delete[] temp;*/}
    
            //cudaFree(c2_output);
            for (int i = 0; i < num_streams; ++i) {
                cudaFree(all_c2_output[i]);
            }
            all_c2_output.clear();
            
            // FC 1
            
            input_channels = 50;
            inputSize = 4;
            numKernels = 500;
            kernelSize = 4;
            outputSize = inputSize -kernelSize+1;
        
            dim3 threadsPerBlock4(16, 16, 1);
            dim3 numBlocks4((outputSize + threadsPerBlock4.x - 1) / threadsPerBlock4.x, (outputSize + threadsPerBlock4.y - 1) / threadsPerBlock4.y, numKernels);
        
        
            for(int i=0;i<num_streams;i++){convolutionWithoutPaddingKernel<<<numBlocks4, threadsPerBlock4, 0, streams[i]>>>(all_p2_output[i], fc1_kernels, fc1_biases, all_fc1_output[i], input_channels, inputSize, kernelSize, numKernels);
            /*float* temp = new float[numKernels * outputSize * outputSize];
                
                cudaMemcpyAsync(temp, all_fc1_output[i], numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
                
                cout << "Printing fc1 output:\n";
              
                for (int i=0; i<1; i++) {
                    for (int j=0; j<outputSize; j++) {
                        for (int k=0; k<outputSize; k++) {
                            cout << temp[j*outputSize + k] ;
                        }
                        cout << endl;
                    }
                    cout << endl;
                }
                
                cout << endl;
                delete[] temp;*/}
    
            //cudaFree(p2_output);
            for (int i = 0; i < num_streams; ++i) {
                cudaFree(all_p2_output[i]);
            }
            all_p2_output.clear();
            // ReLU Layer
            
            input_channels = 500;
            rows = 1;
            cols = 1;
        
            dim3 threadsPerBlock5(16, 16);
            dim3 numBlocks5((rows + threadsPerBlock5.x - 1) / threadsPerBlock5.x, (cols + threadsPerBlock5.y - 1) / threadsPerBlock5.y, input_channels);
        
        
            for(int i=0;i<num_streams;i++){applyReLU_kernel<<<numBlocks5, threadsPerBlock5, 0, streams[i]>>>(all_fc1_output[i], all_r_output[i], rows, cols);
            /*float* temp = new float[numKernels * outputSize * outputSize];
                
                cudaMemcpyAsync(temp, all_r_output[i], numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
                
                cout << "Printing relu output:\n";
              
                for (int i=0; i<1; i++) {
                    for (int j=0; j<outputSize; j++) {
                        for (int k=0; k<outputSize; k++) {
                            cout << temp[j*outputSize + k] ;
                        }
                        cout << endl;
                    }
                    cout << endl;
                }
                
                cout << endl;
                delete[] temp*/;}
    
            //cudaFree(fc1_output);
            for (int i = 0; i < num_streams; ++i) {
                cudaFree(all_fc1_output[i]);
            }
            all_fc1_output.clear();
            // FC 2
            
            input_channels = 500;
            inputSize = 1;
            numKernels = 10;
            kernelSize = 1;
            outputSize = inputSize -kernelSize+1;
        
            dim3 threadsPerBlock6(16, 16, 1);
            dim3 numBlocks6((outputSize + threadsPerBlock6.x - 1) / threadsPerBlock6.x, (outputSize + threadsPerBlock6.y - 1) / threadsPerBlock6.y, numKernels);
        
        
            for(int i=0;i<num_streams;i++){convolutionWithoutPaddingKernel<<<numBlocks6, threadsPerBlock6, 0, streams[i]>>>(all_r_output[i], fc2_kernels, fc2_biases, all_fc2_output[i], input_channels, inputSize, kernelSize, numKernels);
            /*float* temp = new float[numKernels * outputSize * outputSize];
                
                cudaMemcpyAsync(temp, all_fc2_output[i], numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
                
                cout << "Printing fc2 output:\n";
              
                for (int i=0; i<1; i++) {
                    for (int j=0; j<outputSize; j++) {
                        for (int k=0; k<outputSize; k++) {
                            cout << temp[j*outputSize + k] ;
                        }
                        cout << endl;
                    }
                    cout << endl;
                }
                
                cout << endl;
                delete[] temp;*/
            }
            
            //cudaFree(r_output);
            for (int i = 0; i < num_streams; ++i) {
                cudaFree(all_r_output[i]);
            }
            all_r_output.clear();
            // SOFTMAX
            
            float* all_temp = new float[num_streams*10*1*1];
            for(int i=0; i<num_streams; i++){
              cudaMemcpy(all_temp+i*10, all_fc2_output[i], numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
            }
            
            vector<float> all_sum(num_streams,0.0f);
            for(int i=0; i<num_streams; i++){
              for(int j=0; j<10; j++){
                  all_sum[i] += exp(all_temp[i*10 + j]);
                  //cout<<all_temp[i*10+j]<<" ";
                  
              }
              //cout<<all_sum[i]<<" ";
            }
            //cout<<endl;
            
            dim3 threadsPerBlock7(16, 16);
            dim3 numBlocks7((1 + threadsPerBlock7.x - 1) / threadsPerBlock7.x, (10 + threadsPerBlock7.y - 1) / threadsPerBlock7.y);
        
        
            for(int i=0;i<num_streams;i++)softmax<<<numBlocks7, threadsPerBlock7, 0, streams[i]>>>(all_fc2_output[i], probabilities, all_sum[i], 1, 10, c + i);
            
            
            /*vector<float*> all_temp(num_streams,new float[10*1*1]);
            //float* temp = new float[10*1*1];
            for(int i=0;i<num_streams;i++){cudaMemcpy(all_temp[i], all_fc2_output[i], numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
            
            //float* temp = new float[numKernels * outputSize * outputSize];
                
                //cudaMemcpyAsync(temp, all_temp[i], numKernels * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
                
                cout << "Printing temp output:\n";
              
                for (int i=0; i<10; i++) {
                    for (int j=0; j<outputSize; j++) {
                        for (int k=0; k<outputSize; k++) {
                            cout << all_temp[i*numKernels + j][ k] <<" " ;
                        }
                        cout << endl;
                    }
                    cout << endl;
                }
                
                cout << endl;
                //delete[] temp;
                
                }
            
            vector<float> all_sum(num_streams,0.0f);
            //float sum = 0.0f;
            
            for(int i=0;i<num_streams;i++){for (int j=0; j<10; j++) {
                all_sum[i] += exp(all_temp[i][j]);
              }
              cout<<"sum is "<<i<<" "<<all_sum[i]<<"\n";
            }
            
            dim3 threadsPerBlock7(16, 16);
            dim3 numBlocks7((1 + threadsPerBlock7.x - 1) / threadsPerBlock7.x, (10 + threadsPerBlock7.y - 1) / threadsPerBlock7.y);
        
        
            for(int i=0;i<num_streams;i++)softmax<<<numBlocks7, threadsPerBlock7, 0, streams[i]>>>(all_fc2_output[i], probabilities, all_sum[i], 1, 10, c + i);*/
            
            // probabilities += 10;
            
            //cudaFree(fc2_output);
            
            
            //SOFTMAX_NEW
            
            
            
            
            for (int i = 0; i < num_streams; ++i) {
                cudaFree(all_fc2_output[i]);
            }
            all_fc2_output.clear();
            batch_num++;
            
        }
        
    }
    
    
    float* output_prob = new float[num_images*10];
    
    cudaMemcpyAsync(output_prob, probabilities, num_images * 10 * sizeof(float), cudaMemcpyDeviceToHost);
    
    clock_t end = clock();
    double elapsed_time = double(end - start) / CLOCKS_PER_SEC;

    cout << "Time taken by LeNet5 function: " << elapsed_time * 1000 << " milliseconds" << endl;
    
    writeProbabilitiesToFile(output_prob, num_images, filenames);
    
    // cout << "Done\n";
    
    delete[] output_prob;

    return 0;
}