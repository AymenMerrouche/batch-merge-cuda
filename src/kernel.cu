
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <ctime> 
#include <cstdlib>
#include <math.h>

#include "sortedArray.h"

// Function that catches the error
void testCUDA(cudaError_t error, const char *file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

__device__ void diagonalKernel(int idx, int *M, int* A, int* B, const int sizeA, const int sizeB, const int sizeM) {
	
	int K[2];
	int P[2];
	int Q[2];
	int offset;

	// Initializing low and high diagonal points
	if (idx > sizeA) {
		K[0] = idx - sizeA;
		K[1] = sizeA;
		P[0] = sizeA;
		P[1] = idx - sizeA;
	}
	else {
		K[0] = 0;
		K[1] = idx;
		P[0] = idx;
		P[1] = 0;
	}

	while (true) {
		offset = abs(K[1] - P[1]) / 2;
		Q[0] = K[0] + offset;
		Q[1] = K[1] - offset;
		if ((Q[1] >= 0) && (Q[0] <= sizeB) && 
			((Q[1] == sizeA) || (Q[0] == 0) || (A[Q[1]] > B[Q[0]-1]))) {
			if ((Q[0] == sizeB) || (Q[1] == 0) || (A[Q[1]-1] <= B[Q[0]])) {
				if ((Q[1] < sizeA) && ((Q[0] == sizeB) || (A[Q[1]] <= B[Q[0]]))) {
					M[idx] = A[Q[1]];
				}
				else {
					M[idx] = B[Q[0]];
				}
				break;
			}
			else {
				K[0] = Q[0] + 1;
				K[1] = Q[1] - 1;
			}
		}
		else {
			P[0] = Q[0] - 1;
			P[1] = Q[1] + 1;
		}
	}
}

__global__ void mergeKernel(int *M, int* A, int* B, const int sizeA, const int sizeB, const int sizeM) {
	int idx = threadIdx.x;

	while (idx < sizeM) {
		diagonalKernel(idx, M, A, B, sizeA, sizeB, sizeM);
		idx += blockDim.x;
	}
}

double parallelMerge(int *M, int* A, int* B, const int sizeA, const int sizeB) {
	int *gpuA(0), *gpuB(0), *gpuM(0);
	int sizeM = sizeA + sizeB;
	int ntpb; // Number of threads per block
	float ExecutionTime;
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;			// GPU timer instructions

	// Allocate GPU buffers for input arrays A & B, and merged array M
	testCUDA(cudaMalloc(&gpuA, sizeA * sizeof(int)));
	testCUDA(cudaMalloc(&gpuB, sizeB * sizeof(int)));
	testCUDA(cudaMalloc(&gpuM, sizeM * sizeof(int)));

	// Copy our arrays A & B from host memory to GPU buffers.
	testCUDA(cudaMemcpy(gpuA, A, sizeA * sizeof(int), cudaMemcpyHostToDevice));
	testCUDA(cudaMemcpy(gpuB, B, sizeB * sizeof(int), cudaMemcpyHostToDevice));

	// One thread per element of the final array M. If M is larger than 
	// 1024, use 1024 threads
	if (sizeM > 1024) {
		ntpb = 1024;
	}
	else {
		ntpb = sizeM;
	}

	// GPU Timer events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// One block = batch size of 1 for now
	mergeKernel <<<1, ntpb>>> (gpuM, gpuA, gpuB, sizeA, sizeB, sizeM);

	// GPU Timer Instructions
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ExecutionTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy M from GPU buffer to host memory.
	testCUDA(cudaMemcpy(M, gpuM, sizeM * sizeof(int), cudaMemcpyDeviceToHost));

	// Free GPU memory !
	cudaFree(gpuA);
	cudaFree(gpuB);
	cudaFree(gpuM);

	return (double) ExecutionTime;
}

// Sequential Merge
void sequentialMerge(int *M, int* A, int* B, const int sizeA, const int sizeB) {
	int j = 0;
	int i = 0;
	int sizeM = sizeA + sizeB;
	//printf("%d,", sizeM);

	while (i + j < sizeM) {
		if (i >= sizeA) {
			M[i + j] = B[j];
			j++;
		}
		else if ((j >= sizeB) || (A[i] < B[j])) {
			M[i + j] = A[i];
			i++;
		}
		else {
			M[i + j] = B[j];
			j++;
		}
	}
}

double batchSequentialMerge(int *M, int* A, int* B, const int sizeA, const int sizeB, const int batchSize) {
	float time;
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < batchSize; i++) {
		sequentialMerge(M+i*(sizeA+sizeB), A + i * sizeA, B + i * sizeB, sizeA, sizeB);
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	return duration.count();
}

// Helper function to print results
void printMergeResults(int *M, int* A, int* B, const int sizeA, const int sizeB, const int batchSize) {
	for (int i = 0; i < batchSize; i++) {
		printf("A:");
		printArray(A, sizeA);
		printf("B:");
		printArray(B, sizeB);
		printf("M:");
		printArray(M, sizeA + sizeB);
		printf("\n");
	}
}

int main()
{	
	 //Create batch of sorted A and B arrays
	// Batch = 1 for now
	const int VERBOSE = 1;
	const int batchSize = 1;
	const int arraySizeA = 5;
	const int arraySizeB = 4;

	double cpuExecutionTime, gpuExecutionTime;

	int A[batchSize][arraySizeA];
	int B[batchSize][arraySizeB];

	// Two arrays to welcome the result from sequential and parallel functions
	int M[batchSize][arraySizeA + arraySizeB];
	int MG[batchSize][arraySizeA + arraySizeB];
	

	printf("Creating arrays. \n");
	std::srand((unsigned)time(0));
	for (int i = 0; i < batchSize; i++) {
		createSortedArray(&(A[i][0]), arraySizeA);
		createSortedArray(&(B[i][0]), arraySizeB);
	}

	// Merge arrays:
	// The Sequential way on the CPU:
	cpuExecutionTime = batchSequentialMerge(&(M[0][0]), &(A[0][0]), &(B[0][0]), arraySizeA, arraySizeB, batchSize);
	if (VERBOSE == 1) {
		printMergeResults(&(M[0][0]), &(A[0][0]), &(B[0][0]), arraySizeA, arraySizeB, batchSize);
	}
	printf("CPU Execution time %f ms\n", cpuExecutionTime);
	
	// The parallelized way on CUDA
	gpuExecutionTime = parallelMerge(&(MG[0][0]), &(A[0][0]), &(B[0][0]), arraySizeA, arraySizeB);
	if (VERBOSE == 1) {
		printMergeResults(&(MG[0][0]), &(A[0][0]), &(B[0][0]), arraySizeA, arraySizeB, batchSize);
	}
	printf("GPU Execution time %f ms\n", gpuExecutionTime);
   
	testCUDA(cudaDeviceReset());
    return 0;
}

