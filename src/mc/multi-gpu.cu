/*
void pagerank(const int nodes, const int edges, float* value, const int* rowdeg, const int* rowptr, const int* row, const int* col) {

    int ngpus;
    cudaGetDeviceCount(&ngpus);

    float **d_value = new float*[ngpus];
    cudaStream_t* streams = new cudaStream_t[ngpus];

    const int threads_per_block = 1024;

    omp_set_num_threads(ngpus);
    #pragma omp parallel
    {
        int i = omp_get_thread_num();
        cudaSetDevice(i);
        
        cudaStreamCreate(&streams[i]);
        
        int *d_rowptr, *d_col;
        curandStateMRG32k3a *state;

        cudaMalloc(&state, sizeof(curandStateMRG32k3a) * nodes);

        cudaMalloc(&d_value[i], sizeof(float) * nodes);

        cudaMalloc(&d_rowptr, sizeof(int) * (nodes + 1));
        cudaMalloc(&d_col, sizeof(int) * edges);

        setup<<<nodes/threads_per_block+1, threads_per_block, 0, streams[i]>>>(nodes, d_value[i], state);
        random_walk<<<nodes/threads_per_block+1, threads_per_block, 0, streams[i]>>>(nodes, d_value[i], d_rowptr, d_col, state);
        normalize<<<nodes/threads_per_block+1, threads_per_block, 0, streams[i]>>>(nodes, d_value[i]);

        cudaMemcpyAsync(d_rowptr, rowptr, sizeof(int) * (nodes + 1), H2D, streams[i]);
        cudaMemcpyAsync(d_col, col, sizeof(int) * edges, H2D, streams[i]);

        cudaDeviceSynchronize();

        cudaFree(d_rowptr);
        cudaFree(d_col);
        cudaFree(state);
    }


    /*
    cudaStream_t *streams = new cudaStream_t[ngpus];

    float **d_value = new float*[ngpus];
    int **d_rowptr = new int*[ngpus];
    int **d_col = new int*[ngpus];
    curandStateMRG32k3a **state = new curandStateMRG32k3a *[ngpus];
    
}

*/

#include <curand_kernel.h>

#include "../include/pagerank.h"

#define H2D (cudaMemcpyHostToDevice)
#define D2H (cudaMemcpyDeviceToHost)

#define WARP_SIZE 32

int MONTE_CARLO = 1;

__global__ void setup(const int nodes, float* value, curandStateMRG32k3a *state)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < nodes) {
        value[tid] = 0;
    }
    if (threadIdx.x < WARP_SIZE) {
        int rid = threadIdx.x + blockIdx.x * WARP_SIZE;
        curand_init(0, rid, 0, &state[rid]);
    }
}

__global__ void random_walk(const int nodes, float* value, const int* rowptr, const int* col, curandStateMRG32k3a *state)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int rid = (threadIdx.x % WARP_SIZE) + blockIdx.x * WARP_SIZE;

    if (tid < nodes) {
        int cur = tid;
        for (int i = 0; i < length; i++) {
            int deg = rowptr[cur + 1] - rowptr[cur];
            if (curand_uniform(&state[rid]) < alpha)
                cur = deg == 0? cur: col[ rowptr[cur] + (int)(curand_uniform(&state[rid]) * deg) ];
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
    int *d_rowptr, *d_col;
    curandStateMRG32k3a *state;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&state, sizeof(curandStateMRG32k3a) * (nodes / threads_per_block + 1) * WARP_SIZE);

    cudaMalloc(&d_value, sizeof(float) * nodes);

    cudaMalloc(&d_rowptr, sizeof(int) * (nodes + 1));
    cudaMalloc(&d_col, sizeof(int) * edges);

    cudaMemcpyAsync(d_rowptr, rowptr, sizeof(int) * (nodes + 1), H2D, stream);
    cudaMemcpyAsync(d_col, col, sizeof(int) * edges, H2D, stream);

    setup<<<nodes/threads_per_block+1, threads_per_block, 0, stream>>>(nodes, d_value, state);
    random_walk<<<nodes/threads_per_block+1, threads_per_block, 0, stream>>>(nodes, d_value, d_rowptr, d_col, state);
    normalize<<<nodes/threads_per_block+1, threads_per_block, 0, stream>>>(nodes, d_value);

    cudaMemcpyAsync(value, d_value, sizeof(float) * nodes, D2H, stream);

    cudaFree(state);
    cudaFree(d_value);
    cudaFree(d_rowptr);
    cudaFree(d_col);
}
