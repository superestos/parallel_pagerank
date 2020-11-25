#include "../include/pagerank.h"

int MONTE_CARLO = 0;

/*
void pagerank(const int nodes, const int edges, float* value, const int* rowdeg, const int* colptr, const int* row, const int* col)
{
    auto new_value = new float[nodes];
    auto active_nodes = new int[nodes];
    auto next_nodes = new int[nodes];
    auto num_active_nodes = nodes;
    for (int n = 0; n < nodes; n++) {
        active_nodes[n] = n;
    }

    while(true) {
        int e;
        #pragma omp parallel for
        for (int n = 0; n < num_active_nodes; n++) {
            new_value[active_nodes[n]] = 1 - alpha;
            for (int e = colptr[active_nodes[n]]; e < colptr[active_nodes[n] + 1]; e++) {
                new_value[active_nodes[n]] += alpha * value[row[e]] / (float)rowdeg[row[e]];
            }
        }

        int num_next_nodes = 0;
        for (int n = 0; n < num_active_nodes; n++) {
            if (abs(value[active_nodes[n]] - new_value[active_nodes[n]]) > epsilon) {
                next_nodes[num_next_nodes++] = n;
            }
        }

        #pragma omp parallel for
        for (int n = 0; n < num_active_nodes; n++) {
            value[active_nodes[n]] = new_value[active_nodes[n]];
        }

        for (int n = 0; n < num_next_nodes; n++) {
            active_nodes[n] = active_nodes[next_nodes[n]];
        }

        num_active_nodes = num_next_nodes;

        if (num_active_nodes == 0)
            break;
    }

    delete [] new_value;
    delete [] active_nodes;
    delete [] next_nodes;
}
*/


void pagerank(const int nodes, const int edges, float* value, const int* rowdeg, const int* colptr, const int* row, const int* col)
{
    auto new_value = new float[nodes];
    auto active_nodes = new int[nodes];
    auto is_next_nodes = new int[nodes];
    auto next_nodes = new int[nodes];
    auto num_active_nodes = nodes;
    for (int n = 0; n < nodes; n++) {
        active_nodes[n] = n;
    }

    int threads = omp_get_max_threads();
    auto node_partition = new int[threads + 1];
    auto next_partition = new int[threads + 1];

    while(true) {
        #pragma omp parallel for
        for (int n = 0; n < num_active_nodes; n++) {
            new_value[active_nodes[n]] = 1 - alpha;
            for (int e = colptr[active_nodes[n]]; e < colptr[active_nodes[n] + 1]; e++) {
                new_value[active_nodes[n]] += alpha * value[row[e]] / (float)rowdeg[row[e]];
            }
        }

        node_partition[0] = 0;
        next_partition[0] = 0;
        for (int i = 1; i < threads; i++) {
            node_partition[i] = num_active_nodes / threads * i;
            next_partition[i] = 0;
        }
        node_partition[threads] = num_active_nodes;
        next_partition[threads] = 0;

        #pragma omp parallel shared(threads)
        {
            const int tid = omp_get_thread_num();
            for (int n = node_partition[tid]; n < node_partition[tid + 1]; n++) {
                if (abs(value[active_nodes[n]] - new_value[active_nodes[n]]) > epsilon) {
                    is_next_nodes[n] = 1;
                    next_partition[tid + 1] += 1;
                }
                else {
                    is_next_nodes[n] = 0;
                }
            }
        }        

        #pragma omp parallel for
        for (int n = 0; n < num_active_nodes; n++) {
            value[active_nodes[n]] = new_value[active_nodes[n]];
        }

        for (int i = 1; i <= threads; i++) {
            next_partition[i] += next_partition[i - 1];
        }

        #pragma omp parallel shared(threads)
        {
            const int tid = omp_get_thread_num();
            int index = 0;
            for (int n = node_partition[tid]; n < node_partition[tid + 1]; n++) {
                if (is_next_nodes[n]) {
                    next_nodes[next_partition[tid] + index] = active_nodes[n];
                    index++;
                }
            }
        }

        num_active_nodes = next_partition[threads];

        #pragma omp parallel for
        for (int n = 0; n < num_active_nodes; n++) {
            active_nodes[n] = next_nodes[n];
        }

        if (num_active_nodes == 0)
            break;
    }

    delete [] new_value;
    delete [] active_nodes;
    delete [] is_next_nodes;
    delete [] next_nodes;
}
