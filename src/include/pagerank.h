#ifndef PAGERANK_H
#define PAGERANK_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

const float alpha = 0.85;
const float epsilon = 0.0015; // for asynchronize update
const int iteration = 50; // for synchronize update
const int length = 100; // for random walk update

u_int64_t GetTimeStamp();

#endif