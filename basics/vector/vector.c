#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#define N 10000000

void vectorAdd(float *out, float *a, float *b) {
    for(int i = 0; i < N; i++){
        out[i] = a[i] + b[i];
    }
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
    vectorAdd(out, a, b);
	clock_t end = clock();
	double time_taken = ((double)end - start) / CLOCKS_PER_SEC;
    printf("Time taken to add vectors: %f seconds\n", time_taken);

	// Free allocated memory
    free(a);
    free(b);
    free(out);

	return 0;
}