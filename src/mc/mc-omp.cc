#include <random>

#include "../include/pagerank.h"

int MONTE_CARLO = 1;

void pagerank(const int nodes, const int edges, float* value, const int* rowdeg, const int* rowptr, const int* row, const int* col) 
{
    int threads = omp_get_max_threads();
    
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    auto new_value = new int[nodes * threads];

    #pragma omp parallel for
    for (int n = 0; n < nodes; n++) {
        value[n] = 0.0;
        for (int i = 0; i < threads; i++) {
            new_value[n * threads + i] = 0;
        }
    }

    #pragma omp parallel shared(threads)
    {
        std::random_device rand_dev;
        std::mt19937 generator(rand_dev());
        
        const int tid = omp_get_thread_num();

        for (int n = tid; n < nodes; n += threads) {
            for (int j = 0; j < walkers; j++) {
                int cur = n;
                for (int i = 0; i < length; i++) {
                    if (distribution(generator) < alpha) 
                        cur = rowdeg[cur] == 0? cur: col[rowptr[cur] + (int)(distribution(generator) * rowdeg[cur])];
                    else 
                        cur = n;

                    new_value[cur * threads + tid] += 1;
                }
            }
        }
    }
        
    #pragma omp parallel for
    for (int n = 0; n < nodes; n++) {
        for (int i = 0; i < threads; i++) {
            value[n] += new_value[n * threads + i];
        }
        value[n] /= (length * walkers);
    }

    delete [] new_value;
}