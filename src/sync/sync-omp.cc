#include "../include/pagerank.h"

int MONTE_CARLO = 0;

void pagerank(const int nodes, const int edges, float* value, const int* rowdeg, const int* colptr, const int* row, const int* col)
{
    auto new_value = new float[nodes];

    for (int i = 0; i < iteration; i++) {
        #pragma omp parallel for
        for (int n = 0; n < nodes; n++) {
            new_value[n] = 1 - alpha;
        }

        float message;
        #pragma omp parallel for private(message)
        for (int e = 0; e < edges; e++) {
            message = alpha * value[row[e]] / (float)rowdeg[row[e]];

            #pragma omp atomic
            new_value[col[e]] += message;
        }
        
        #pragma omp parallel for
        for (int n = 0; n < nodes; n++) {
            value[n] = new_value[n];
        }
    }

    delete [] new_value;
}


/*
void pagerank(const int nodes, const int edges, float* value, const int* rowdeg, const int* colptr, const int* row, const int* col)
{
    int threads = omp_get_max_threads();
    //auto node_partition = new int[nodes + 1];
    //node_partition[0] = 0;

    auto new_value = new float[nodes];

    while(true) {

        #pragma omp parallel for
        for (int n = 0; n < nodes; n++) {
            new_value[n] = 1 - alpha;
        }

        int n;
        #pragma omp parallel private(n) shared(threads) num_threads(threads)
        {
            const int tid = omp_get_thread_num();
            for (n = tid; n < nodes; n += threads) {
                for(int e = colptr[n]; e < colptr[n + 1]; e++)
                    new_value[n] += alpha * value[row[e]] / (float)rowdeg[row[e]];
            }
        }

        volatile bool flag = true;
        #pragma omp parallel for shared(flag)
        for (int n = 0; n < nodes; n++) {
            if (!flag)
                continue;

            if (abs(value[n] - new_value[n]) > epsilon) {
                flag = false;
            }
        }

        #pragma omp parallel for
        for (int n = 0; n < nodes; n++) {
            value[n] = new_value[n];
        }

        if (flag)
            break;
    }

    delete [] new_value;
}



void pagerank(const int nodes, const int edges, float* value, const int* rowdeg, const int* colptr, const int* row, const int* col)
{
    int threads = omp_get_max_threads();
    auto edge_partition = new int[threads + 1];
    edge_partition[0] = 0;
    for (int i = 1; i < threads; i++) {
        edge_partition[i] = colptr[col[(i * edges) / threads]];
    }
    edge_partition[threads] = edges;

    auto new_value = new float[nodes];

    while(true) {

        #pragma omp parallel for
        for (int n = 0; n < nodes; n++) {
            new_value[n] = 1 - alpha;
        }

        #pragma omp parallel shared(threads) num_threads(threads)
        {
            const int tid = omp_get_thread_num();
            const int max_e = edge_partition[tid + 1];
            for(int e = edge_partition[tid]; e < max_e; e++) {
                new_value[col[e]] += alpha * value[row[e]] / (float)rowdeg[row[e]];
            }
        }

        volatile bool flag = true;
        #pragma omp parallel for shared(flag)
        for (int n = 0; n < nodes; n++) {
            if (!flag)
                continue;

            if (abs(value[n] - new_value[n]) > epsilon) {
                flag = false;
            }
        }

        #pragma omp parallel for
        for (int n = 0; n < nodes; n++) {
            value[n] = new_value[n];
        }

        if (flag)
            break;
    }

    delete [] new_value;
}
*/