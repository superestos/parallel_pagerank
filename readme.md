## dataset

Stanford
http://snap.stanford.edu/data/web-Stanford.html

281,903	2,312,497 17.6MB

Google
http://snap.stanford.edu/data/web-Google.html

875,713	5,105,039 38.9MB

LiveJournal
http://snap.stanford.edu/data/soc-LiveJournal1.html

4,847,571	68,993,773 526.4MB

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

sync-cuda.cu

stanford

I/O         time usage: 47073 microseconds
Computation time usage: 2464376 microseconds
PageRank value of node 1: 32.950520

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   77.27%  452.00ms       500  904.00us  841.36us  994.61us  aggregate(int, float*, float*, int const *)
                   13.62%  79.649ms       500  159.30us  145.96us  174.56us  compute(int, float*, float*, int const *, int const *, int const *)
                    8.96%  52.416ms        50  1.0483ms  104.77us  3.2086ms  [CUDA memcpy HtoD]
                    0.15%  904.60us        10  90.459us  89.090us  97.826us  [CUDA memcpy DtoH]


google

I/O         time usage: 133497 microseconds
Computation time usage: 3517403 microseconds
PageRank value of node 1: 1.277758

 GPU activities:   79.24%  1.31511s       500  2.6302ms  2.6160ms  2.6557ms  compute(int, float*, float*, int const *, int const *, int const *)
                   10.87%  180.37ms       500  360.75us  348.04us  410.63us  aggregate(int, float*, float*, int const *)
                    9.51%  157.87ms        50  3.1575ms  748.59us  10.006ms  [CUDA memcpy HtoD]
                    0.38%  6.3203ms        10  632.03us  389.93us  1.7154ms  [CUDA memcpy DtoH]

livejournal

I/O         time usage: 820499 microseconds
Computation time usage: 16428674 microseconds
PageRank value of node 1: 19.971058

 GPU activities:   76.83%  11.0871s       500  22.174ms  22.080ms  22.276ms  compute(int, float*, float*, int const *, int const *, int const *)
                   12.27%  1.77022s        50  35.404ms  5.0423ms  114.15ms  [CUDA memcpy HtoD]
                   10.73%  1.54866s       500  3.0973ms  3.0801ms  3.1667ms  aggregate(int, float*, float*, int const *)
                    0.17%  24.094ms        10  2.4094ms  1.6197ms  7.8667ms  [CUDA memcpy DtoH]


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

async-cuda.cu

stanford

I/O         time usage: 47391 microseconds
Computation time usage: 3502787 microseconds
PageRank value of node 1: 32.982006

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   97.18%  1.39338s       222  6.2765ms  2.4640us  17.678ms  compute(int, int*, float*, float*, int const *, int const *, int const *, int const *)
                    2.56%  36.722ms        60  612.04us  97.186us  2.1992ms  [CUDA memcpy HtoD]
                    0.08%  1.0930ms       232  4.7110us     768ns  94.050us  [CUDA memcpy DtoH]
                    0.06%  900.88us       222  4.0580us  3.0720us  14.304us  void thrust::cuda_cub::core::_kernel_agent<thrust::cuda_cub::__scan::ScanAgent<int*, int*, thrust::plus<int>, int, int, thrust::detail::integral_constant<bool, bool=0>>, int*, int*, thrust::plus<int>, int, thrust::cuda_cub::cub::ScanTileState<int, bool=1>, thrust::cuda_cub::__scan::AddInitToExclusiveScan<int, thrust::plus<int>>>(int*, int, thrust::plus<int>, int, int, bool)
                    0.04%  543.37us       222  2.4470us     992ns  11.265us  find_active(int, int*, float*, float*, int*)
                    0.03%  428.27us       222  1.9290us     928ns  10.657us  copy_value(int, int*, float*, float*)
                    0.02%  338.24us       222  1.5230us     832ns  8.9600us  coalesce_next_active(int, int*, int*, int*)
                    0.02%  221.12us       212  1.0430us     800ns  5.3760us  copy_active(int, int*, int*)
                    0.01%  173.13us       222     779ns     704ns  1.3440us  void thrust::cuda_cub::core::_kernel_agent<thrust::cuda_cub::__scan::InitAgent<thrust::cuda_cub::cub::ScanTileState<int, bool=1>, int>, thrust::cuda_cub::cub::ScanTileState<int, bool=1>, int>(bool=1, thrust::cuda_cub::cub::ScanTileState<int, bool=1>)

google

I/O         time usage: 98503 microseconds
Computation time usage: 2619654 microseconds
PageRank value of node 1: 1.279837

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   75.63%  409.22ms       199  2.0564ms  1.9520us  6.2714ms  compute(int, int*, float*, float*, int const *, int const *, int const *, int const *)
                   22.55%  122.01ms        60  2.0335ms  425.10us  5.4425ms  [CUDA memcpy HtoD]
                    0.93%  5.0506ms       209  24.165us     768ns  602.86us  [CUDA memcpy DtoH]
                    0.25%  1.3317ms       199  6.6910us     992ns  39.809us  find_active(int, int*, float*, float*, int*)
                    0.23%  1.2187ms       199  6.1240us  3.1040us  35.648us  void thrust::cuda_cub::core::_kernel_agent<thrust::cuda_cub::__scan::ScanAgent<int*, int*, thrust::plus<int>, int, int, thrust::detail::integral_constant<bool, bool=0>>, int*, int*, thrust::plus<int>, int, thrust::cuda_cub::cub::ScanTileState<int, bool=1>, thrust::cuda_cub::__scan::AddInitToExclusiveScan<int, thrust::plus<int>>>(int*, int, thrust::plus<int>, int, int, bool)
                    0.20%  1.0698ms       199  5.3760us     960ns  33.569us  copy_value(int, int*, float*, float*)
                    0.13%  681.24us       199  3.4230us     832ns  33.569us  coalesce_next_active(int, int*, int*, int*)
                    0.06%  313.67us       189  1.6590us     832ns  19.521us  copy_active(int, int*, int*)
                    0.04%  209.83us       199  1.0540us     768ns  1.8560us  void thrust::cuda_cub::core::_kernel_agent<thrust::cuda_cub::__scan::InitAgent<thrust::cuda_cub::cub::ScanTileState<int, bool=1>, int>, thrust::cuda_cub::cub::ScanTileState<int, bool=1>, int>(bool=1, thrust::cuda_cub::cub::ScanTileState<int, bool=1>)


livejournal

I/O         time usage: 831540 microseconds
Computation time usage: 5525880 microseconds
PageRank value of node 1: 19.967129

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   49.86%  1.83446s        60  30.574ms  5.2344ms  101.60ms  [CUDA memcpy HtoD]
                   49.00%  1.80279s       237  7.6067ms  1.8560us  52.767ms  compute(int, int*, float*, float*, int const *, int const *, int const *, int const *)
                    0.68%  24.876ms       247  100.71us     768ns  8.5067ms  [CUDA memcpy DtoH]
                    0.15%  5.4367ms       237  22.939us     992ns  222.02us  find_active(int, int*, float*, float*, int*)
                    0.14%  5.0466ms       237  21.293us     928ns  192.42us  copy_value(int, int*, float*, float*)
                    0.08%  2.9164ms       237  12.305us  3.1040us  127.59us  void thrust::cuda_cub::core::_kernel_agent<thrust::cuda_cub::__scan::ScanAgent<int*, int*, thrust::plus<int>, int, int, thrust::detail::integral_constant<bool, bool=0>>, int*, int*, thrust::plus<int>, int, thrust::cuda_cub::cub::ScanTileState<int, bool=1>, thrust::cuda_cub::__scan::AddInitToExclusiveScan<int, thrust::plus<int>>>(int*, int, thrust::plus<int>, int, int, bool)
                    0.07%  2.4956ms       237  10.529us     832ns  166.31us  coalesce_next_active(int, int*, int*, int*)
                    0.03%  998.42us       227  4.3980us     832ns  107.27us  copy_active(int, int*, int*)
                    0.01%  243.37us       237  1.0260us     768ns  1.9210us  void thrust::cuda_cub::core::_kernel_agent<thrust::cuda_cub::__scan::InitAgent<thrust::cuda_cub::cub::ScanTileState<int, bool=1>, int>, thrust::cuda_cub::cub::ScanTileState<int, bool=1>, int>(bool=1, thrust::cuda_cub::cub::ScanTileState<int, bool=1>)


mc.cc

stanford

I/O         time usage: 42365 microseconds
Computation time usage: 32260834 microseconds
PageRank value of node 1: 32.639999

google

I/O         time usage: 131703 microseconds
Computation time usage: 103374297 microseconds
PageRank value of node 1: 1.360000

livejournal

I/O         time usage: 1189023 microseconds
Computation time usage: 890902554 microseconds
PageRank value of node 1: 19.700001

mc-omp.cc

stanford

I/O         time usage: 23746 microseconds
Computation time usage: 3614330 microseconds
PageRank value of node 1: 31.870001

google

I/O         time usage: 101329 microseconds
Computation time usage: 10914644 microseconds
PageRank value of node 1: 1.480000

livejournal

I/O         time usage: 1084153 microseconds
Computation time usage: 96116458 microseconds
PageRank value of node 1: 19.940001

mc-cuda.cu

stanford

I/O         time usage: 47149 microseconds
Computation time usage: 2285887 microseconds
PageRank value of node 1: 31.190001

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   49.90%  163.10ms        10  16.310ms  15.238ms  17.008ms  setup(int, float*, curandStateMRG32k3a*)
                   42.18%  137.85ms        10  13.785ms  13.348ms  14.122ms  random_walk(int, float*, int const *, int const *, curandStateMRG32k3a*)
                    7.63%  24.937ms        20  1.2468ms  106.76us  2.4056ms  [CUDA memcpy HtoD]
                    0.27%  897.08us        10  89.707us  87.874us  95.266us  [CUDA memcpy DtoH]
                    0.02%  64.930us        10  6.4930us  6.0490us  6.8490us  normalize(int, float*)

google

I/O         time usage: 94369 microseconds
Computation time usage: 3212417 microseconds
PageRank value of node 1: 1.470000

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   58.49%  736.95ms        10  73.695ms  73.158ms  74.523ms  random_walk(int, float*, int const *, int const *, curandStateMRG32k3a*)
                   36.23%  456.51ms        10  45.651ms  42.946ms  54.947ms  setup(int, float*, curandStateMRG32k3a*)
                    4.90%  61.788ms        20  3.0894ms  734.16us  5.7610ms  [CUDA memcpy HtoD]
                    0.35%  4.4339ms        10  443.39us  289.32us  1.4274ms  [CUDA memcpy DtoH]
                    0.02%  190.88us        10  19.088us  18.592us  20.064us  normalize(int, float*)

livejournal

I/O         time usage: 837066 microseconds
Computation time usage: 11396985 microseconds
PageRank value of node 1: 20.250000

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   64.52%  6.17237s        10  617.24ms  615.42ms  618.23ms  random_walk(int, float*, int const *, int const *, curandStateMRG32k3a*)
                   26.90%  2.57391s        10  257.39ms  252.98ms  295.09ms  setup(int, float*, curandStateMRG32k3a*)
                    8.00%  764.99ms        20  38.250ms  3.7658ms  107.45ms  [CUDA memcpy HtoD]
                    0.57%  54.671ms        10  5.4671ms  1.5866ms  8.6177ms  [CUDA memcpy DtoH]
                    0.01%  1.0709ms        10  107.09us  106.69us  107.49us  normalize(int, float*)

multi-gpu.cu

stanford

I/O         time usage: 47571 microseconds
Computation time usage: 2659527 microseconds
PageRank value of node 1: 32.796669

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   51.58%  468.21ms        30  15.607ms  13.890ms  17.032ms  setup(int, float*, curandStateMRG32k3a*)
                   44.91%  407.67ms        30  13.589ms  13.011ms  14.260ms  random_walk(int, float*, int const *, int const *, curandStateMRG32k3a*)
                    3.05%  27.729ms        60  462.15us  96.579us  886.67us  [CUDA memcpy HtoD]
                    0.44%  3.9764ms        30  132.55us  86.114us  199.37us  [CUDA memcpy DtoH]
                    0.02%  182.05us        30  6.0680us  5.3440us  6.5600us  normalize(int, float*)

google

I/O         time usage: 92903 microseconds
Computation time usage: 3617789 microseconds
PageRank value of node 1: 1.303333

 GPU activities:   60.68%  2.21131s        30  73.710ms  73.104ms  74.628ms  random_walk(int, float*, int const *, int const *, curandStateMRG32k3a*)
                   37.30%  1.35912s        30  45.304ms  42.385ms  54.942ms  setup(int, float*, curandStateMRG32k3a*)
                    1.73%  63.008ms        60  1.0501ms  294.67us  1.8812ms  [CUDA memcpy HtoD]
                    0.28%  10.158ms        30  338.60us  265.06us  559.76us  [CUDA memcpy DtoH]
                    0.02%  568.75us        30  18.958us  18.272us  20.481us  normalize(int, float*)

livejournal

I/O         time usage: 857300 microseconds
Computation time usage: 11675446 microseconds
PageRank value of node 1: 19.793333

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   67.87%  18.5075s        30  616.92ms  615.03ms  617.97ms  random_walk(int, float*, int const *, int const *, curandStateMRG32k3a*)
                   28.35%  7.73163s        30  257.72ms  251.46ms  307.93ms  setup(int, float*, curandStateMRG32k3a*)
                    3.59%  977.65ms        60  16.294ms  1.6294ms  34.341ms  [CUDA memcpy HtoD]
                    0.18%  49.990ms        30  1.6663ms  1.4701ms  2.3650ms  [CUDA memcpy DtoH]
                    0.01%  3.2131ms        30  107.10us  106.37us  107.62us  normalize(int, float*)
