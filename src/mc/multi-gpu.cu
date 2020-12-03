/*

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

    

    for (int i = 0; i < ngpus; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        
        cudaMalloc(&state[i], sizeof(curandStateMRG32k3a) * nodes);
        
        cudaMalloc(&d_value[i], sizeof(float) * nodes);

        cudaMalloc(&d_rowptr[i], sizeof(int) * (nodes + 1));
        cudaMalloc(&d_col[i], sizeof(int) * edges);

        cudaMemcpyAsync(d_rowptr[i], rowptr, sizeof(int) * (nodes + 1), H2D, streams[i]);
        cudaMemcpyAsync(d_col[i], col, sizeof(int) * edges, H2D, streams[i]);

        setup<<<nodes/threads_per_block+1, threads_per_block, 0, streams[i]>>>(nodes, d_value[i], state[i]);
        random_walk<<<nodes/threads_per_block+1, threads_per_block, 0, streams[i]>>>(nodes, d_value[i], d_rowptr[i], d_col[i], state[i]);
        normalize<<<nodes/threads_per_block+1, threads_per_block, 0, streams[i]>>>(nodes, d_value[i]);
    }

    for (int i = 0; i < ngpus; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }

    cudaSetDevice(0);
    

    cudaMemcpyAsync(value, d_value, sizeof(float) * nodes, D2H, streams[0]);

    for (int i = 0; i < ngpus; i++) {
        cudaFree(d_value[i]);
    }
    
}

*/
