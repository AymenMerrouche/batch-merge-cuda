
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <ctime> 
#include <cstdlib>
#include <math.h>

#include "sortedArray.h"

# define nb_arrays 5
# define size_of_array 100

// Function that catches the error
void testCUDA(cudaError_t error, const char *file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

// Helper function to print results
void printSortResults(int *M, int* T, const int arraySizeT, const int nb_arrays_to_srt) {
	int i, j;

	for (i = 0; i < nb_arrays_to_srt; i++)
	{
		printf("Tab :");
		printArray(&(T[i*arraySizeT]), arraySizeT);
		printf("Sorted Tab :");
		printArray(&(M[i*arraySizeT]), arraySizeT);
		printf("\n");
	}
}

__device__ void diagonalKernel(int effective_idx, int idx, int *M, int* A, int* B, const int sizeA, const int sizeB, const int sizeM) {
	
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
					M[effective_idx] = A[Q[1]];
				}
				else {
					M[effective_idx] = B[Q[0]];
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
__global__ void batchMergeSortKernel(int *M, int* T, const int sizeT, const int sizeM, const int nb_steps) {
	extern __shared__ int local_M[];

	int idx, relative_idx, begin_index_of_A, begin_index_of_B, sizeB;
	int *effective_A, *effective_B;
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	int bidx = blockIdx.x;
	int *temp = &T[bidx*sizeT];

	for (int i = 0; i < nb_steps; i++) {
		idx = threadIdx.x;
		int effective_tab_size = powf(2, i);
		
		while (idx < sizeM) {
			begin_index_of_A = (idx / (2 * effective_tab_size)) * (2 * effective_tab_size);
			begin_index_of_B = begin_index_of_A + effective_tab_size;

			// Only if there is a paired array to merge with 
			if (begin_index_of_B < sizeM) {
				effective_A = &temp[begin_index_of_A];
				effective_B = &temp[begin_index_of_B];
				relative_idx = idx % (2 * effective_tab_size);
				sizeB = effective_tab_size;
				if (begin_index_of_B + effective_tab_size >= sizeM) {
					sizeB = sizeM - begin_index_of_B;
				}
				diagonalKernel(begin_index_of_A + relative_idx, relative_idx, local_M,
					effective_A, effective_B, effective_tab_size,
					sizeB, effective_tab_size + sizeB);
			}
			else { // Else, stay in place
				for (int j = begin_index_of_A; j < sizeM; j++) {
					local_M[j] = temp[j];
				}
			}
			idx += blockDim.x;
		}
		__syncthreads();
		
		// Copy to temp
		if ((threadIdx.x == 0)) {
			for (int j = 0; j < sizeM; j++) {
				local_M[sizeM + j] = local_M[j];
			}
		}
		__syncthreads();
		temp = local_M + sizeM;
	}
	// Copy sorted local_M in the global M
	if ((threadIdx.x == 0)) {
		for (int j = 0; j < sizeM; j++) {
			M[bidx*sizeM + j] = local_M[j];
		}
	}
}

double batchParallelMergeSort(int *M, int *T, const int sizeT, const int nb_tabs) {
	int *gpuT(0), *gpuM(0);
	int sizeM = sizeT;
	float ExecutionTime;
	int ntpb;
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;			// GPU timer instructions
	int nb_steps = (int)ceil(log2(sizeM));			// Number of steps to perform
	printf("%d steps for %d elements", nb_steps, sizeT);

	// Allocate GPU buffers for input array T , and sorted array M
	testCUDA(cudaMalloc(&gpuT, sizeT * nb_tabs * sizeof(int)));
	testCUDA(cudaMalloc(&gpuM, sizeM * nb_tabs * sizeof(int)));

	// Copy our array T from host memory to GPU buffers.
	testCUDA(cudaMemcpy(gpuT, T, sizeT * nb_tabs * sizeof(int), cudaMemcpyHostToDevice));


	// GPU Timer events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// One thread per element in the array. If T is larger than 
	// 1024, use 1024 threads
	if (sizeT > 1024) {
		ntpb = 1024;
	}
	else {
		ntpb = sizeT;
	}

	batchMergeSortKernel <<<nb_tabs, ntpb, 2 * sizeM * sizeof(int)>> > (gpuM, gpuT, sizeT, sizeM, nb_steps);
   
	// GPU Timer Instructions
	testCUDA(cudaEventRecord(stop, 0));
	testCUDA(cudaEventSynchronize(stop));
	testCUDA(cudaEventElapsedTime(&ExecutionTime, start, stop));
	testCUDA(cudaEventDestroy(start));
	testCUDA(cudaEventDestroy(stop));
	// Copy M from GPU buffer to host memory.
	testCUDA(cudaMemcpy(M, gpuM, sizeM * nb_tabs * sizeof(int), cudaMemcpyDeviceToHost));
	// Free GPU memory !
	cudaFree(gpuT);
	cudaFree(gpuM);

	return (double) ExecutionTime;
}

int main()
{	
	// Batch = 1 for now
	const int VERBOSE = 1;

	// For the Sorting of an array
    const int arraySizeT = size_of_array;
	const int nb_arrays_to_srt = nb_arrays;
    // time containers
    double gpuExecutionTimeSort, cpuExecutionTimeSort;
	// array containes (they all are flattened)
    // Array for the merge path sort
    int T[nb_arrays_to_srt*arraySizeT];
    // Array to welcome the result
    int MS[nb_arrays_to_srt*arraySizeT];
	// create the array
	printf("Creating %d arrays of size %d...\n", nb_arrays_to_srt, arraySizeT);
	createRandVector(&(T[0]), nb_arrays_to_srt*arraySizeT);
	for (int i = 0; i < nb_arrays_to_srt; i++)
	{
		printf("Tab :");
		printArray(&(T[i*arraySizeT]), arraySizeT);
		printf("\n");
	}

	// Sort Array:
    // The parallelized way on CUDA
	gpuExecutionTimeSort = batchParallelMergeSort(&(MS[0]), &(T[0]), arraySizeT, nb_arrays_to_srt);
	if (VERBOSE == 1) {
		printf("\n Sorting Results \n \n :");
		printSortResults(&(MS[0]), &(T[0]), arraySizeT, nb_arrays_to_srt);
	}
	printf("GPU Execution Sort time %f ms\n", gpuExecutionTimeSort);
   
	testCUDA(cudaDeviceReset());
    return 0;
}

