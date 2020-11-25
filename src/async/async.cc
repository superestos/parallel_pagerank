#include "../include/pagerank.h"

int MONTE_CARLO = 0;

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