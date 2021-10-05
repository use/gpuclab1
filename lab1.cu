#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef unsigned long long bignum;

// CUDA kernel. Each thread takes care of one element of c
__device__ int isPrime(bignum x)
{

    bignum i;
    bignum lim = (bignum)sqrt((float)x) + 1;

    if (x % 2 == 0)
    {
        return 0;
    }

    for (i = 3; i < lim; i += 2)
    {
        if (x % i == 0)
            return 0;
    }

    return 1;
}

__global__ void checkPrimes(int *results, int arr_size)
{
    // Get our global thread ID
    bignum index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < arr_size)
    {
        bignum number = 2 * index + 1;
        results[index] = isPrime(number);
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Usage: prime upbound\n");
        exit(-1);
    }

    bignum n = (bignum)atoi(argv[1]);

    // Host input vectors
    int *h_results;

    // Device input vectors
    int *d_results;

    size_t arr_size = (int)ceil((float) ((n - 1.0) / 2.0));
    printf("arr_size: %ld\n", arr_size);

    // Size, in bytes, of each vector
    size_t results_num_bytes = arr_size * sizeof(int);

    // Allocate memory for each vector on host
    h_results = (int *)malloc(results_num_bytes);

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_results, results_num_bytes);

    bignum i;
    // Initialize vectors on host
    for (i = 0; i < arr_size; i++)
    {
        h_results[i] = 0;
    }

    // Copy host vectors to device
    cudaMemcpy(d_results, h_results, results_num_bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 1024;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float) ((n + 1.0) / 2.0 / blockSize));
    // gridSize = (int)ceil((float)n / blockSize);
    printf("gridSize: %d\n", gridSize);
    printf("gridSize * blockSize: %d\n", gridSize*blockSize);

    // Execute the kernel
    checkPrimes<<<gridSize, blockSize>>>(d_results, arr_size);

    // Copy array back to host
    cudaMemcpy(h_results, d_results, results_num_bytes, cudaMemcpyDeviceToHost);

    // Sum up vector c and print result divided by n, this should equal 1 without error
    bignum sum = 0;
    for (i = 0; i < arr_size; i++)
    {
        sum += h_results[i];
    }
    printf("final result: %lld\n", sum);

    // Release device memory
    cudaFree(d_results);

    // Release host memory
    free(h_results);

    return 0;
}
