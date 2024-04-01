#include <stdio.h> // printf in GPU

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cuda_hello<<<1,1>>>(); 
    cudaDeviceSynchronize(); // wait for the GPU to finish so that the message is printed
    return 0;
}