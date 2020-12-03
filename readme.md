## dataset

Stanford
http://snap.stanford.edu/data/web-Stanford.html

Google
http://snap.stanford.edu/data/web-Google.html

LiveJournal
http://snap.stanford.edu/data/soc-LiveJournal1.html

## compile

```bash
g++ -o pagerank -fopenmp src/main.cc src/sync/sync-omp.cc
```

```bash
nvcc -o pagerank src/main.cc src/sync/sync-cuda.cu
```

## run

```bash
./run.sh stanford
./run.sh google
./run.sh livejournal
```

## result

sync.cc

stanford

I/O         time usage: 62894 microseconds
Computation time usage: 9712150 microseconds
PageRank value of node 1: 32.950520

google

I/O         time usage: 128097 microseconds
Computation time usage: 79874643 microseconds
PageRank value of node 1: 1.277758

livejournal

I/O         time usage: 1168667 microseconds
Computation time usage: 757234389 microseconds
PageRank value of node 1: 19.971058

sync-omp.ccz

stanford

I/O         time usage: 52582 microseconds
Computation time usage: 1713952 microseconds
PageRank value of node 1: 32.950520

google

I/O         time usage: 100624 microseconds
Computation time usage: 11559313 microseconds
PageRank value of node 1: 1.277758

livejournal

I/O         time usage: 1083613 microseconds
Computation time usage: 138585274 microseconds
PageRank value of node 1: 19.971058



async.cc

stanford

I/O         time usage: 51340 microseconds
Computation time usage: 1095136 microseconds
PageRank value of node 1: 32.982006

google

I/O         time usage: 104508 microseconds
Computation time usage: 5655090 microseconds
PageRank value of node 1: 1.279837

livejournal

I/O         time usage: 1078292 microseconds
Computation time usage: 47351883 microseconds
PageRank value of node 1: 19.967129



async-omp.cc

stanford

I/O         time usage: 45756 microseconds
Computation time usage: 252289 microseconds
PageRank value of node 1: 32.982006

google

I/O         time usage: 101111 microseconds
Computation time usage: 928979 microseconds
PageRank value of node 1: 1.279837

livejournal

I/O         time usage: 1098548 microseconds
Computation time usage: 14678046 microseconds
PageRank value of node 1: 19.967129

mc.cc

stanford

I/O         time usage: 52446 microseconds
Computation time usage: 62788629 microseconds
PageRank value of node 1: 31.950001

google

I/O         time usage: 106990 microseconds
Computation time usage: 268881820 microseconds
PageRank value of node 1: 1.890000

mc-omp.cc

stanford

I/O         time usage: 49758 microseconds
Computation time usage: 6777395 microseconds
PageRank value of node 1: 31.875000

google

I/O         time usage: 99884 microseconds
Computation time usage: 27057848 microseconds
PageRank value of node 1: 2.115000

livejournal

I/O         time usage: 1058972 microseconds
Computation time usage: 231170631 microseconds
PageRank value of node 1: 31.870001
