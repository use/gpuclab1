#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef unsigned long long bignum;

// CUDA kernel. Each thread takes care of one element of c
__device__ int isPrime(bignum x){

    bignum i;
    bignum lim = (bignum) sqrt((float) x) + 1;


    if (x%2==0) {
	    return 0;
    }

    for(i=3; i < lim; i+=2){
        if ( x % i == 0)
            return 0;
    }

   return 1;
}

__global__ void checkPrimes(int *results, bignum *numbers, int n)
{
    // Get our global thread ID
    int index = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (index < n)
        results[index] = isPrime(numbers[index]);
}


int main( int argc, char* argv[] )
{
    if(argc < 2)
    {
        printf("Usage: prime upbound\n");
        exit(-1);
    }

    bignum n = (bignum) atoi(argv[1]);
 
    // Host input vectors
    bignum *h_numbers;
    int *h_results;
 
    // Device input vectors
    bignum *d_numbers;
    // Device output vector
    int *d_results;
 
    // Size, in bytes, of each vector
    size_t bytes_numbers = n*sizeof(bignum);
    size_t bytes_results = n*sizeof(int);
 
    // Allocate memory for each vector on host
    h_numbers = (bignum*)malloc(bytes_numbers);
    h_results = (int*)malloc(bytes_results);
 
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_numbers, bytes_numbers);
    cudaMalloc(&d_results, bytes_results);
 
    bignum i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        h_numbers[i] = i + 1;
        h_results[i] = 0;
    }
 
    // Copy host vectors to device
    cudaMemcpy( d_numbers, h_numbers, bytes_numbers, cudaMemcpyHostToDevice);
    cudaMemcpy( d_results, h_results, bytes_results, cudaMemcpyHostToDevice);
 
    int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 1024;
 
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n/blockSize);
 
    // Execute the kernel
    checkPrimes<<<gridSize, blockSize>>>(d_results, d_numbers, n);
 
    // Copy array back to host
    cudaMemcpy( h_results, d_results, bytes_results, cudaMemcpyDeviceToHost );
 
    // Sum up vector c and print result divided by n, this should equal 1 without error
    int sum = 0;
    for(i=0; i<n; i++)
        sum += h_results[i];
    printf("final result: %d\n", sum);
 
    // Release device memory
    cudaFree(d_numbers);
    cudaFree(d_results);
 
    // Release host memory
    free(h_numbers);
    free(h_results);
 
    return 0;
}
