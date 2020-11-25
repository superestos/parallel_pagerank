#include "include/pagerank.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#define H2D (cudaMemcpyHostToDevice)
#define D2H (cudaMemcpyDeviceToHost)

int MONTE_CARLO = 0;

__global__ void compute(const int num_active_nodes, int* active_nodes, float* value, float* new_value, const int* rowdeg, const int* colptr, const int* row, const int* col)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < num_active_nodes) {
        int n = active_nodes[tid];
        new_value[n] = 1 - alpha;

        for (int e = colptr[n]; e < colptr[n + 1]; e++) {
            new_value[n] += alpha * value[row[e]] / (float)rowdeg[row[e]];
        }
    }
}

__global__ void find_active(const int num_active_nodes, int* active_nodes, float* value, float* new_value, int* is_next_nodes)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < num_active_nodes) {
        int n = active_nodes[tid];
        is_next_nodes[tid] = abs(value[n] - new_value[n]) > epsilon? 1: 0;
    }
}

__global__ void copy_value(const int num_active_nodes, int* active_nodes, float* value, float* new_value)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < num_active_nodes) {
        int n = active_nodes[tid];
        value[n] = new_value[n];
    }
}

__global__ void coalesce_next_active(const int num_active_nodes, int* active_nodes, int* next_nodes, int* is_next_nodes)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_active_nodes && is_next_nodes[tid] < is_next_nodes[tid + 1]) {
        next_nodes[is_next_nodes[tid]] = active_nodes[tid];
    }
}

__global__ void copy_active(const int num_active_nodes, int* active_nodes, int* next_nodes)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < num_active_nodes) {
        active_nodes[tid] = next_nodes[tid];
    }
}

void pagerank(const int nodes, const int edges, float* value, const int* rowdeg, const int* colptr, const int* row, const int* col)
{
    float *d_value, *d_new_value;
    int *d_active_nodes, *d_is_next_nodes, *d_next_nodes, *d_rowdeg, *d_colptr, *d_row, *d_col;

    int num_active_nodes = nodes;
    int* active_nodes = new int[nodes];
    for (int n = 0; n < nodes; n++) {
        active_nodes[n] = n;
    }

    const int threads_per_block = 128;

    cudaMalloc(&d_value, sizeof(float) * nodes);
    cudaMalloc(&d_new_value, sizeof(float) * nodes);

    cudaMalloc(&d_active_nodes, sizeof(int) * nodes);
    cudaMalloc(&d_is_next_nodes, sizeof(int) * (nodes + 1));
    cudaMalloc(&d_next_nodes, sizeof(int) * nodes);

    cudaMalloc(&d_rowdeg, sizeof(int) * nodes);
    cudaMalloc(&d_colptr, sizeof(int) * (nodes + 1));
    cudaMalloc(&d_row, sizeof(int) * edges);
    cudaMalloc(&d_col, sizeof(int) * edges);

    cudaMemcpy(d_value, value, sizeof(float) * nodes, H2D);
    cudaMemcpy(d_active_nodes, active_nodes, sizeof(int) * nodes, H2D);

    cudaMemcpy(d_rowdeg, rowdeg, sizeof(int) * nodes, H2D);
    cudaMemcpy(d_colptr, colptr, sizeof(int) * (nodes + 1), H2D);
    cudaMemcpy(d_row, row, sizeof(int) * edges, H2D);
    cudaMemcpy(d_col, col, sizeof(int) * edges, H2D);

    while (true) {
        compute<<<num_active_nodes/threads_per_block+1,threads_per_block>>>(num_active_nodes, d_active_nodes, d_value, d_new_value, d_rowdeg, d_colptr, d_row, d_col);
        find_active<<<num_active_nodes/threads_per_block+1,threads_per_block>>>(num_active_nodes, d_active_nodes, d_value, d_new_value, d_is_next_nodes);
        copy_value<<<num_active_nodes/threads_per_block+1,threads_per_block>>>(num_active_nodes, d_active_nodes, d_value, d_new_value);

        thrust::exclusive_scan(thrust::device, d_is_next_nodes, d_is_next_nodes + num_active_nodes + 1, d_is_next_nodes);
        coalesce_next_active<<<num_active_nodes/threads_per_block+1, threads_per_block>>>(num_active_nodes, d_active_nodes, d_next_nodes, d_is_next_nodes);

        cudaMemcpy(&num_active_nodes, &d_is_next_nodes[num_active_nodes], sizeof(int), D2H);

        if (num_active_nodes == 0)
            break;
        
        copy_active<<<num_active_nodes/threads_per_block+1,threads_per_block>>>(num_active_nodes, d_active_nodes, d_next_nodes);
    }

    cudaMemcpy(value, d_value, sizeof(float) * nodes, D2H);

    cudaFree(d_value);
    cudaFree(d_new_value);

    cudaFree(d_active_nodes);
    cudaFree(d_is_next_nodes);
    cudaFree(d_next_nodes);

    cudaFree(d_rowdeg);
    cudaFree(d_colptr);
    cudaFree(d_row);
    cudaFree(d_col);
}
