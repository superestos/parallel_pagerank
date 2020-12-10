#include <curand_kernel.h>

#include "../include/pagerank.h"

#include <thrust/sort.h>

#define H2D (cudaMemcpyHostToDevice)
#define D2H (cudaMemcpyDeviceToHost)

#define WARP_SIZE 32

int MONTE_CARLO = 1;

__global__ void setup_walker(const int nodes, float* value, int* current, int* source)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < nodes) {
        value[tid] = 0;
        current[tid] = tid;
        source[tid] = tid;
    }
}

__global__ void setup_rand(curandStateMRG32k3a *state, unsigned long seed)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &state[tid]);
}

__global__ void random_walk(const int walks, float* value, const int* rowptr, const int* col, curandStateMRG32k3a *state, int* current, int* source)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int rid = (threadIdx.x % WARP_SIZE) + blockIdx.x * WARP_SIZE;

    if (tid < walks) {
        int cur = current[tid];
        int deg = rowptr[cur + 1] - rowptr[cur];

        if (curand_uniform(&state[rid]) < alpha)
            cur = deg == 0? cur: col[ rowptr[cur] + (int)(curand_uniform(&state[rid]) * deg) ];
        else
            cur = source[tid];

        atomicAdd(&value[cur], 1);
        current[tid] = cur;
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
    int *d_rowptr, *d_col, *current, *source;
    curandStateMRG32k3a *state;

    const int threads_per_block = 512;

    const int blocks = 16;
    int* partition = new int[blocks + 1];
    for (int b = 0; b < blocks; b++) {
        partition[b] = b * (nodes / blocks);
    }
    partition[blocks] = nodes;

    cudaMalloc(&state, sizeof(curandStateMRG32k3a) * ((partition[blocks] - partition[blocks - 1])/threads_per_block + 1) * WARP_SIZE);

    cudaMalloc(&d_value, sizeof(float) * nodes);

    cudaMalloc(&d_rowptr, sizeof(int) * (nodes + 1));
    cudaMalloc(&d_col, sizeof(int) * edges);

    cudaMalloc(&current, sizeof(int) * nodes);
    cudaMalloc(&source, sizeof(int) * nodes);

    cudaMemcpy(d_rowptr, rowptr, sizeof(int) * (nodes + 1), H2D);
    cudaMemcpy(d_col, col, sizeof(int) * edges, H2D);

    setup_walker<<<nodes/threads_per_block+1, threads_per_block>>>(nodes, d_value, current, source);

    setup_rand<<<(partition[blocks] - partition[blocks - 1])/threads_per_block + 1, WARP_SIZE>>>(state, time(NULL));

    for (int i = 0; i < length; i++) {
        thrust::sort_by_key(thrust::device, current, current + nodes, source);
        for (int b = 0; b < blocks; b++) {
            random_walk<<<(partition[b + 1] - partition[b])/threads_per_block+1, threads_per_block>>> \
                (nodes, d_value, d_rowptr, d_col, state, current + partition[b], source + partition[b]);
        }
    }
    
    normalize<<<nodes/threads_per_block+1, threads_per_block>>>(nodes, d_value);

    cudaMemcpy(value, d_value, sizeof(float) * nodes, D2H);

    cudaFree(state);
    cudaFree(d_value);
    cudaFree(d_rowptr);
    cudaFree(d_col);
    cudaFree(current);
    cudaFree(source);
}

