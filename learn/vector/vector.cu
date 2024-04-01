#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#define N 10000000

inline unsigned int ceil(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

__global__ void vectorAddKernel(float *out, float *a, float *b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

void vectorAdd(float *a, float *b, float *out) {
    int size = N * sizeof(float);
    float *a_d, *b_d, *out_d;

    // allocate memory on the GPU
    cudaMalloc((void**)&a_d, size);
    cudaMalloc((void**)&b_d, size);
    cudaMalloc((void**)&out_d, size);

    // Copy data to the GPU
    cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, size, cudaMemcpyHostToDevice);

    // Launch the kernel
    vectorAddKernel<<<ceil(N/256.0),256>>>(out, a, b, N);

    // Copy data back to the CPU
    cudaMemcpy(out, out_d, size, cudaMemcpyDeviceToHost);

    // Free memory on the GPU
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(out_d);
}

int main(){
    float *a, *b, *out; 

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

    // Main function
	clock_t start = clock();
    vectorAdd(a, b, out);
    clock_t end = clock();
	double time_taken = ((double)end - start) / CLOCKS_PER_SEC;
    printf("Time taken to add vectors: %f seconds\n", time_taken);

	// Free allocated memory
    free(a);
    free(b);
    free(out);

	return 0;
}