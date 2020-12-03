// Directed graph (each unordered pair of nodes is saved once): web-Stanford.txt 
// Stanford web graph row 2002
// Nodes: 281903 Edges: 2312497

#include "include/pagerank.h"

u_int64_t GetTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec * (u_int64_t)1000000 + tv.tv_usec;
}

extern void pagerank(const int nodes, const int edges, float* value, const int* rowdeg, const int* ptr, const int* row, const int* col);
extern int MONTE_CARLO;

void csc2csr(const int nodes, const int edges, const int* rowdeg, int* rowptr, int* row, int* col) 
{
    int* new_row = new int[edges];
    int* new_col = new int[edges];
    int* index = new int[nodes];

    rowptr[0] = 0;
    for (int i = 0; i < nodes; i++) {
        rowptr[i + 1] = rowptr[i] + rowdeg[i];
        index[i] = 0;
    }

    for (int i = 0; i < edges; i++) {
        new_row[rowptr[row[i]] + index[row[i]]] = row[i];
        new_col[rowptr[row[i]] + index[row[i]]] = col[i];
        index[row[i]]++;
    }

    for (int i = 0; i < edges; i++) {
        row[i] = new_row[i];
        col[i] = new_col[i];
    }

    delete [] new_row;
    delete [] new_col;
    delete [] index;

    /*
    for (int n = 0; n < nodes; n++) {
        for (int i = rowptr[new_row[n]]; i < rowptr[new_row[n]]; i++) {
            int min = i;
            for (int j = i + 1; j < rowptr[new_row[n]]; j++) {
                if (new_col[min] > new_col[j]) {
                    min = j;
                }
            }
            std::swap(new_col[i], new_col[min]);
        }
    }
    */
}

void init_data(int nodes, int edges, char *filename, float* value, int* rowdeg, int* colptr, int* row , int* col)
{
    FILE* file = fopen(filename, "r");
    assert(file != NULL);

    for (int i = 0; i < nodes; i++) {
        rowdeg[i] = 0;
        value[i] = alpha;
    }

    fread(row, sizeof(int), edges, file);
    fread(col, sizeof(int), edges, file);

    int j = 0;
    colptr[0] = 0;

    for (int i = 0; i < edges; i++) {
        rowdeg[row[i]]++;
        while (j < col[i]) {
            colptr[++j] = i;
        }
    }

    colptr[nodes] = edges;
}

int main(int argc, char* argv[])
{   
    if (argc != 7 || strcasecmp("-f", argv[1]) || strcasecmp("-n", argv[3]) || strcasecmp("-e", argv[5])) {
        fprintf(stderr, "Usage ./pagerank -f (file name) -n (number of nodes) -e (number of edges)\n");
        return 1;
    }

    char *filename = argv[2];
    int nodes = atoi(argv[4]);
    int edges = atoi(argv[6]);

    float* value = new float[nodes];
    int* rowdeg = new int[nodes];
    int* ptr = new int[nodes + 1];
    int* row = new int[edges];
    int* col = new int[edges];

    u_int64_t start_t, total_t;

    start_t = GetTimeStamp();
    init_data(nodes, edges, filename, value, rowdeg, ptr, row, col);
    total_t = GetTimeStamp() - start_t;
    printf("I/O         time usage: %d microseconds\n", total_t);

    if (MONTE_CARLO) {
        csc2csr(nodes, edges, rowdeg, ptr, row, col);
    }

    start_t = GetTimeStamp();
    //for(int i = 0; i < 10; i++) {
        pagerank(nodes, edges, value, rowdeg, ptr, row, col);
    //}
    total_t = GetTimeStamp() - start_t;
    printf("Computation time usage: %d microseconds\n", total_t);

    //int target = 198519;
    int target = 1;
    printf("PageRank value of node %d: %f\n", target, value[target]);

    

    delete [] value;
    delete [] rowdeg;
    delete [] row;
    delete [] col;

    return 0;
}