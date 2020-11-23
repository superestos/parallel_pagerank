#include "include/pagerank.h"

#define H2D (cudaMemcpyHostToDevice)
#define D2H (cudaMemcpyDeviceToHost)

__global__ void compute(const int nodes, const int edges, float* value, float* new_value, const int* rowdeg, const int* colptr, const int* row, const int* col) 
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < nodes) {
        new_value[tid] = 1 - alpha;

        for (int e = colptr[tid]; e < colptr[tid + 1]; e++) {
            new_value[tid] += alpha * value[row[e]] / (float)rowdeg[row[e]];
        }
    }

}

__global__ void copy_value(const int nodes, float* value, float* new_value)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < nodes) {
        value[tid] = new_value[tid];
    }
}

void pagerank(const int nodes, const int edges, float* value, const int* rowdeg, const int* colptr, const int* row, const int* col)
{
    float *d_value, *d_new_value;
    int *d_rowdeg, *d_colptr, *d_row, *d_col;

    cudaMalloc(&d_value, sizeof(float) * nodes);
    cudaMalloc(&d_new_value, sizeof(float) * nodes);
    cudaMalloc(&d_rowdeg, sizeof(int) * nodes);
    cudaMalloc(&d_colptr, sizeof(int) * (nodes + 1));
    cudaMalloc(&d_row, sizeof(int) * edges);
    cudaMalloc(&d_col, sizeof(int) * edges);

    cudaMemcpy(d_value, value, sizeof(float) * nodes, H2D);
    cudaMemcpy(d_rowdeg, rowdeg, sizeof(int) * nodes, H2D);
    cudaMemcpy(d_colptr, colptr, sizeof(int) * (nodes + 1), H2D);
    cudaMemcpy(d_row, row, sizeof(int) * edges, H2D);
    cudaMemcpy(d_col, col, sizeof(int) * edges, H2D);

    for (int i = 0; i < iteration; i++) {
        compute<<<nodes/512+1, 512>>>(nodes, edges, d_value, d_new_value, d_rowdeg, d_colptr, d_row, d_col);
        copy_value<<<nodes/512+1, 512>>>(nodes, d_value, d_new_value);
    }

    cudaMemcpy(value, d_value, sizeof(float) * nodes, D2H);

    cudaFree(d_value);
    cudaFree(d_new_value);
    cudaFree(d_rowdeg);
    cudaFree(d_colptr);
    cudaFree(d_row);
    cudaFree(d_col);
}