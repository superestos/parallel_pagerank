#include "include/pagerank.h"

int MONTE_CARLO = 0;

void pagerank(const int nodes, const int edges, float* value, const int* rowdeg, const int* colptr, const int* row, const int* col)
{
    auto new_value = new float[nodes];

    for (int i = 0; i < iteration; i++) {
        for (int n = 0; n < nodes; n++) {
            new_value[n] = 1 - alpha;
        }

        for (int e = 0; e < edges; e++) {
            new_value[col[e]] += alpha * value[row[e]] / (float)rowdeg[row[e]];
        }

        for (int n = 0; n < nodes; n++) {
            value[n] = new_value[n];
        }
    }

    delete [] new_value;
}