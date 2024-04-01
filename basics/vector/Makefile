build-cpu:
	gcc vector.c -o vector_cpu

build:
	nvcc vector.cu -o vector

profile:
	nvprof ./vector