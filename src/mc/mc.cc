#include <random>

#include "../include/pagerank.h"

int MONTE_CARLO = 1;

void pagerank(const int nodes, const int edges, float* value, const int* rowdeg, const int* rowptr, const int* row, const int* col) 
{
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_real_distribution<float> distribution(0.0,1.0);

    for (int n = 0; n < nodes; n++) {
        value[n] = 0.0;
    }

    for (int n = 0; n < nodes; n++) {
        for (int j = 0; j < walkers; j++) {
            int cur = n;
            for (int i = 0; i < length; i++) {
                if (distribution(generator) < alpha) 
                    cur = rowdeg[cur] == 0? cur: col[rowptr[cur] + (int)(distribution(generator) * rowdeg[cur])];
                else 
                    cur = n;

                value[cur] += 1;
            }
        }
    }

    for (int n = 0; n < nodes; n++) {
        value[n] /= (length * walkers);
    }
}