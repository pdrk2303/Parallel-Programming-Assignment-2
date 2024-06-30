#include <stdlib.h>
#include <iostream>
#include <math.h>
#include<vector>
#include<limits>
using namespace std;
vector<vector<float> > convolutionWithPadding(const vector<vector<float> >& input, const vector<vector<float> >& kernel, int paddingSize) {
    int inputSize = input.size();
    int kernelSize = kernel.size();
    // int paddingSize = (kernelSize-1) / 2;
    //cout<<"Size calculated "<<endl;
    // Create padded input matrix
    vector<vector<float> > paddedInput(inputSize + 2 * paddingSize, vector<float>(inputSize + 2 * paddingSize, 0.0));
    for (int i = paddingSize; i < inputSize + paddingSize; ++i) {
        for (int j = paddingSize; j < inputSize + paddingSize; ++j) {
            paddedInput[i][j] = input[i - paddingSize][j - paddingSize];
        }
    }
    //cout<<"Padded input set"<<endl;
    inputSize = inputSize + 2 * paddingSize;
    // Perform convolution
    vector<vector<float> > output(inputSize - kernelSize + 1, vector<float>(inputSize - kernelSize + 1, 0.0));
    for (int i = 0; i < inputSize - kernelSize + 1; ++i) {
        for (int j = 0; j < inputSize - kernelSize + 1; ++j) {
            for (int m = 0; m < kernelSize; ++m) {
                for (int n = 0; n < kernelSize; ++n) {
                    if(i+m>=0 && j+n>=0 && i+m<paddedInput.size() && j+n<paddedInput.size()){
                    output[i][j] += paddedInput[i+m][j+n] * kernel[m][n];
                    }
                }
            }
        }
    }

    return output;
}

vector<vector<float> > convolutionWithoutPadding(const vector<vector<float> >& input, const vector<vector<float> >& kernel) {
    int inputSize = input.size();
    int kernelSize = kernel.size();

    // Perform convolution
    //cout<<"Input size and kernel size are "<<inputSize<<" "<<kernelSize<<endl;
    vector<vector<float> > output(inputSize - kernelSize + 1, vector<float>(inputSize - kernelSize + 1, 0.0));
    for (int i = 0; i < inputSize - kernelSize + 1; ++i) {
        for (int j = 0; j < inputSize - kernelSize + 1; ++j) {
            for (int m = 0; m < kernelSize; ++m) {
                for (int n = 0; n < kernelSize; ++n) {
                    output[i][j] += input[i + m][j + n] * kernel[m][n];
                }
            }
        }
    }

    return output;
}
vector<vector<float> > maxPooling(const vector<vector<float> >& input, int poolSize) {
    int rows = input.size();
    int cols = input[0].size();
    int outputRows = rows / poolSize;
    int outputCols = cols / poolSize;

    vector<vector<float> > output(outputRows, vector<float>(outputCols, 0.0f));

    for (int i = 0; i < outputRows; ++i) {
        for (int j = 0; j < outputCols; ++j) {
            float maxVal = std::numeric_limits<float>::min();
            for (int m = 0; m < poolSize; ++m) {
                for (int n = 0; n < poolSize; ++n) {
                    maxVal = max(maxVal, input[i * poolSize + m][j * poolSize + n]);
                }
            }
            output[i][j] = maxVal;
        }
    }

    return output;
}

// Average pooling function
vector<vector<float> > averagePooling(const vector<vector<float> >& input, int poolSize) {
    int rows = input.size();
    int cols = input[0].size();
    int outputRows = rows / poolSize;
    int outputCols = cols / poolSize;

    vector<vector<float> > output(outputRows, vector<float>(outputCols, 0.0f));

    for (int i = 0; i < outputRows; ++i) {
        for (int j = 0; j < outputCols; ++j) {
            float sum = 0.0f;
            for (int m = 0; m < poolSize; ++m) {
                for (int n = 0; n < poolSize; ++n) {
                    sum += input[i * poolSize + m][j * poolSize + n];
                }
            }
            output[i][j] = sum / (poolSize * poolSize);
        }
    }

    return output;
}
float relu(float x) {
    return max(0.0f, x);
}

// Tanh activation function
float tanh_activation(float x) {
    return tanh(x);
}

// Apply ReLU activation function element-wise on a matrix
vector<vector<float> > applyReLU(const vector<vector<float> >& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    vector<vector<float> > result(rows, vector<float>(cols, 0.0f));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = relu(matrix[i][j]);
        }
    }

    return result;
}

// Apply tanh activation function element-wise on a matrix
vector<vector<float> > applyTanh(const vector<vector<float> >& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    vector<vector<float> > result(rows, vector<float>(cols, 0.0f));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = tanh_activation(matrix[i][j]);
        }
    }

    return result;
}
vector<vector<double> > softmax(const vector<vector<float> >& input) {
    vector<vector<double> > probabilities(input.size(),vector<double>(input[0].size(),0));
    double sum = 0.0f;

    // Compute exponentials and sum
    for(int i=0;i<input.size();i++){
        for (int j=0;j<input[i].size();j++) {
            float val = input[i][j];
            //cout<<"exp is "<<exp(val)<<" val is "<<val<<endl;
            sum += exp(val);
        }
    }
    //cout<<"sum is "<<sum<<"\n";
    // Compute probabilities
    for(int j=0;j<input.size();j++){
        for (int i = 0; i < input[j].size(); ++i) {
            probabilities[j][i] = exp(input[j][i]) / sum;
        }
    }

    return probabilities;
}

// Sigmoid function
vector<vector<float> > sigmoid(const vector<vector<float> >& input) {
    vector<vector<float> > probabilities(input.size(),vector<float>(input[0].size(),0));

    // Compute probabilities
    for(int j=0;j<input.size();j++){
        for (size_t i = 0; i < input[j].size(); ++i) {
            probabilities[j][i] = 1.0f / (1.0f + exp(-input[j][i]));
        }
    }

    return probabilities;
}
void printMatrix(const vector<vector<float> >& input) {
    for(int j=0;j<input.size();j++){
        for (size_t i = 0; i < input[j].size(); ++i) {
            cout << input[j][i] << " ";
        }
        cout << endl;
    }
    
}


int main(int argv, char* argc[]){
    int task = atos(argc[1]);
    if(task==1){
        int n=atos(argc[2]);
        int m=atos(argc[3]);
        int p=atos(argc[4]);
        vector<vector<float> >input(n,vector<float>(n,0.0));
        vector<vector<float> >kernel(m,vector<float>(m,0.0));
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                input[i][j]=atos(argc[5+i*n+j]);
            }
        }
        for(int i=0;i<m;i++){
            for(int j=0;j<m;j++){
                kernel[i][j]=atos(argc[5+n*n+i*m+j]);
            }
        }
        if(p!=0){
            vector<vector<float> > output = convolutionWithPadding(input,kernel,p);
            printMatrix(output);
        }
        else{
            vector<vector<float> > output = convolutionWithoutPadding(input,kernel);
            printMatrix(output);
        }
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
            vector<vector<float> > output= applyReLU(input);
            printMatrix(output);
        }
        else if(sub_task==1){
            vector<vector<float> > output= applyTanh(input);
            printMatrix(output);
        }
    }
    else if(task==3){
        int sub_task = atos(argc[2]);
        int pool_size = atos(argc[3]);
        int n = atos(argc[4]);
        vector<vector<float> >input(n,vector<float>(n,0.0));
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                input[i][j]=atos(argc[5+i*n+j]);
            }
        }
        if(sub_task==0){
            vector<vector<float> > output = maxPooling(input,pool_size);
            printMatrix(output);
        }
        else if(sub_task==1){
            vector<vector<float> > output = averagePooling(input,pool_size);
            printMatrix(output);
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
            vector<vector<float> > output = sigmoid(input);
            printMatrix(output);
        }
        else if(sub_task==1){
            vector<vector<double> > output = softmax(input);
            for(int j=0;j<output.size();j++){
        for (size_t i = 0; i < output[j].size(); ++i) {
                    cout << output[j][i] << " ";
                }
                cout << endl;
    }
        }
    }
    else{
        cout<<"Invalid task"<<endl;
    }
}
