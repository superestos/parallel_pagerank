#include <curand_kernel.h>

#include "../include/pagerank.h"

#define H2D (cudaMemcpyHostToDevice)
#define D2H (cudaMemcpyDeviceToHost)

int MONTE_CARLO = 1;

__global__ void setup(const int nodes, float* value, curandStateMRG32k3a *state)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < nodes) {
        value[tid] = 0;
        curand_init(0, tid, 0, &state[tid]);
    }
}

__global__ void random_walk(const int nodes, float* value, const int* rowptr, const int* col, curandStateMRG32k3a *state)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < nodes) {
        int cur = tid;
        for (int i = 0; i < length; i++) {
            if (curand_uniform(&state[tid]) < alpha)
                cur = col[ rowptr[cur] + (int)(curand_uniform(&state[tid]) * (rowptr[cur + 1] - rowptr[cur])) ];
            else
                cur = tid;

            atomicAdd(&value[cur], 1);
        }
    }
}

__global__ void normalize(const int nodes, float* value)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < nodes) {
        value[tid] /= length;
    }
}


void pagerank(const int nodes, const int edges, float* value, const int* rowdeg, const int* rowptr, const int* row, const int* col) {
    float *d_value;
    int *d_rowptr, *d_col;
    curandStateMRG32k3a *state;

    cudaMalloc(&state, sizeof(curandStateMRG32k3a) * nodes);

    cudaMalloc(&d_value, sizeof(float) * nodes);

    cudaMalloc(&d_rowptr, sizeof(int) * (nodes + 1));
    cudaMalloc(&d_col, sizeof(int) * edges);

    cudaMemcpy(d_rowptr, rowptr, sizeof(int) * (nodes + 1), H2D);
    cudaMemcpy(d_col, col, sizeof(int) * edges, H2D);

    const int threads_per_block = 1024;

    setup<<<nodes/threads_per_block+1, threads_per_block>>>(nodes, d_value, state);
    random_walk<<<nodes/threads_per_block+1, threads_per_block>>>(nodes, d_value, d_rowptr, d_col, state);
    normalize<<<nodes/threads_per_block+1, threads_per_block>>>(nodes, d_value);

    cudaMemcpy(value, d_value, sizeof(float) * nodes, D2H);

    cudaFree(state);
    cudaFree(d_value);
    cudaFree(d_rowptr);
    cudaFree(d_col);
}
