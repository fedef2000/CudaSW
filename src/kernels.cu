#include "kernels.cuh"
#include <stdio.h>

__global__ void test_kernel() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

void launch_test_kernel() {
    test_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
