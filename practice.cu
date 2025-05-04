#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <cmath>

#define N 10000000 // Size for vector addition
#define M 30       // Size for matrix multiplication

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    } \
}

using namespace std;

// CUDA kernel for vector addition
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel for matrix multiplication
__global__ void matrixMul(float *A, float *B, float *C, int m) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < m && col < m) {
        float sum = 0;
        for (int k = 0; k < m; ++k) {
            sum += A[row * m + k] * B[k * m + col];
        }
        C[row * m + col] = sum;
    }
}

// Helper function to display a vector
void displayVector(float* vec, int size, int numElements = 10) {
    cout << "[ ";
    for (int i = 0; i < min(numElements, size); i++) {
        cout << vec[i] << " ";
    }
    cout << "... ]" << endl;
}

// Helper function to display a matrix
void displayMatrix(float* mat, int rows, int cols, int numRows = 5, int numCols = 5) {
    for (int i = 0; i < min(numRows, rows); i++) {
        cout << "[ ";
        for (int j = 0; j < min(numCols, cols); j++) {
            cout << mat[i * cols + j] << " ";
        }
        cout << "]" << endl;
    }
}

int main() {
    // Vector memory
    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;
    h_a = new float[N];
    h_b = new float[N];
    h_c = new float[N];
    vector<float> cpu_c_vector(N);

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    CHECK_CUDA_ERROR(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, N * sizeof(float)));

    // Vector addition (CPU)
    auto start_cpu = chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cpu_c_vector[i] = h_a[i] + h_b[i];
    }
    auto end_cpu = chrono::high_resolution_clock::now();
    double cpu_time = chrono::duration<double>(end_cpu - start_cpu).count();

    // Vector addition (GPU)
    cudaEvent_t start, stop;
    float gpu_time;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    vectorAdd<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);
    CHECK_CUDA_ERROR(cudaGetLastError());            // Add this line
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());       // Add this line


    cudaEventRecord(stop);

    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    cout << "\nVector Addition (CPU and GPU) Results:" << endl;
    cout << "CPU Vector (first 10): ";
    displayVector(cpu_c_vector.data(), N);

    cout << "GPU Vector (first 10): ";
    displayVector(h_c, N);

    cout << "CPU Time: " << cpu_time << " sec" << endl;
    cout << "GPU Time: " << gpu_time / 1000.0 << " sec" << endl;
    cout << "Speedup: " << cpu_time / (gpu_time / 1000.0) << endl;

    bool vector_match = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c[i] - cpu_c_vector[i]) > 1e-5) {
            vector_match = false;
            break;
        }
    }
    cout << "Vector Result Match: " << (vector_match ? "Yes" : "No") << endl;

    // Matrix memory
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    h_A = new float[M * M];
    h_B = new float[M * M];
    h_C = new float[M * M];
    vector<float> cpu_C_vector(M * M);

    for (int i = 0; i < M * M; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * M * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, M * M * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * M * sizeof(float)));

    // Matrix multiplication (CPU)
    start_cpu = chrono::high_resolution_clock::now();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            float sum = 0;
            for (int k = 0; k < M; k++) {
                sum += h_A[i * M + k] * h_B[k * M + j];
            }
            cpu_C_vector[i * M + j] = sum;
        }
    }
    end_cpu = chrono::high_resolution_clock::now();
    cpu_time = chrono::duration<double>(end_cpu - start_cpu).count();

    // Matrix multiplication (GPU)
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * M * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, M * M * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + 15) / 16, (M + 15) / 16);

    cudaEventRecord(start);
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M);
    CHECK_CUDA_ERROR(cudaGetLastError());            // Add this line
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());       // Add this line
    



    cudaEventRecord(stop);

    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, M * M * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    cout << "\nMatrix Multiplication (CPU and GPU) Results:" << endl;
    cout << "CPU Matrix (first 5x5):" << endl;
    displayMatrix(cpu_C_vector.data(), M, M);
    cout << "GPU Matrix (first 5x5):" << endl;
    displayMatrix(h_C, M, M);

    cout << "CPU Time: " << cpu_time << " sec" << endl;
    cout << "GPU Time: " << gpu_time / 1000.0 << " sec" << endl;
    cout << "Speedup: " << cpu_time / (gpu_time / 1000.0) << endl;

    bool matrix_match = true;
    for (int i = 0; i < M * M; i++) {
        if (fabs(h_C[i] - cpu_C_vector[i]) > 1e-5) {
            matrix_match = false;
            break;
        }
    }
    cout << "Matrix Result Match: " << (matrix_match ? "Yes" : "No") << endl;

    // Cleanup
    delete[] h_a; delete[] h_b; delete[] h_c;
    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
