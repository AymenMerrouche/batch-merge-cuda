
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <ctime> 
#include <cstdlib>
#include <math.h>

#include "sortedArray.h"
# define NTPB 16
# define NB 5
# define nb_arrays 5
# define size_of_array 16

// Function that catches the error
void testCUDA(cudaError_t error, const char *file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

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



__global__ void mergeSortKernel(int *M, int* T, const int sizeT, const int sizeM, const int i) {
	int idx = threadIdx.x;
    // size of each sorted tab within a couple to sort
    int effective_tab_size = pow(2, i); 
    // total number of tabs to sort
    int nb_tabs = sizeT/effective_tab_size;
    // shared memory to contain result
    __shared__ int local_M[NTPB];
	int begin_index_of_A = (idx / (2*effective_tab_size)) * (2*effective_tab_size);
	int begin_index_of_B = (idx / (2*effective_tab_size)) * (2*effective_tab_size) + effective_tab_size;
    int *effective_A = &T[begin_index_of_A];
    int *effective_B = &T[begin_index_of_B];
	
	while (idx < sizeM) {
		diagonalKernel(begin_index_of_A + idx %(2*effective_tab_size), idx %(2*effective_tab_size), local_M, effective_A, effective_B, effective_tab_size, effective_tab_size, 2*effective_tab_size);
		idx += blockDim.x;
	}
    // copy sorted effecive_M in the global M
	M[begin_index_of_A + idx %(2*effective_tab_size)] = local_M[begin_index_of_A + idx %(2*effective_tab_size)]; 
}

double parallelMergeSort(int *M, int *T, const int sizeT) {
	int *gpuT(0), *gpuM(0);
	int sizeM = sizeT;
	float ExecutionTime;
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;			// GPU timer instructions

	// Allocate GPU buffers for input array T , and sorted array M
	testCUDA(cudaMalloc(&gpuT, sizeT * sizeof(int)));
	testCUDA(cudaMalloc(&gpuM, sizeM * sizeof(int)));

	// Copy our array T from host memory to GPU buffers.
	testCUDA(cudaMemcpy(gpuT, T, sizeT * sizeof(int), cudaMemcpyHostToDevice));


	// GPU Timer events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// One block = batch size of 1 for now
    // get number of sorting steps to perform
    int nb_steps = log2(sizeM);
	int i=0;
	int *intemediate = &gpuT[0];
    for (i=0; i<nb_steps; i++){
        mergeSortKernel <<<1, NTPB>>> (gpuM, intemediate, sizeT, sizeM, i);
		intemediate = &gpuM[0];
    }
	
	// GPU Timer Instructions
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ExecutionTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	// Copy M from GPU buffer to host memory.
	
	testCUDA(cudaMemcpy(M, gpuM, sizeM * sizeof(int), cudaMemcpyDeviceToHost));
	// Free GPU memory !
	cudaFree(gpuT);
	cudaFree(gpuM);

	return (double) ExecutionTime;
}

__global__ void batchMergeSortKernel(int *M, int* T, const int sizeT, const int sizeM, const int i) {
	int idx = threadIdx.x;
	int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	int bidx = blockIdx.x;
    // size of each sorted tab within a couple to sort
    int effective_tab_size = pow(2, i); 
    // total number of tabs to sort
    int nb_tabs = sizeT/effective_tab_size;
    // shared memory to contain result
    __shared__ int local_M[NTPB];
	int begin_index_of_A = (idx / (2*effective_tab_size)) * (2*effective_tab_size);
	int begin_index_of_B = (idx / (2*effective_tab_size)) * (2*effective_tab_size) + effective_tab_size;
    int *effective_A = &T[bidx*sizeT + begin_index_of_A];
    int *effective_B = &T[bidx*sizeT + begin_index_of_B];
	
	diagonalKernel(begin_index_of_A + idx %(2*effective_tab_size), idx %(2*effective_tab_size), local_M, effective_A, effective_B, effective_tab_size, effective_tab_size, 2*effective_tab_size);

    // copy sorted effecive_M in the global M
	M[bidx*sizeT + begin_index_of_A + idx %(2*effective_tab_size)] = local_M[begin_index_of_A + idx %(2*effective_tab_size)]; 
}

double batchParallelMergeSort(int *M, int *T, const int sizeT, const int nb_tabs) {
	int *gpuT(0), *gpuM(0);
	int sizeM = sizeT;
	float ExecutionTime;
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;			// GPU timer instructions

	// Allocate GPU buffers for input array T , and sorted array M
	testCUDA(cudaMalloc(&gpuT, sizeT * nb_tabs * sizeof(int)));
	testCUDA(cudaMalloc(&gpuM, sizeM * nb_tabs * sizeof(int)));

	// Copy our array T from host memory to GPU buffers.
	testCUDA(cudaMemcpy(gpuT, T, sizeT * nb_tabs * sizeof(int), cudaMemcpyHostToDevice));


	// GPU Timer events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// One block = batch size of 1 for now
    // get number of sorting steps to perform
    int nb_steps = log2(sizeM);
	int i=0;
	int *intemediate = &gpuT[0];
    for (i=0; i<nb_steps; i++){
        batchMergeSortKernel <<<NB, NTPB>>> (gpuM, intemediate, sizeT, sizeM, i);
		intemediate = &gpuM[0];
    }
	
	// GPU Timer Instructions
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ExecutionTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	// Copy M from GPU buffer to host memory.
	
	testCUDA(cudaMemcpy(M, gpuM, sizeM * nb_tabs * sizeof(int), cudaMemcpyDeviceToHost));
	// Free GPU memory !
	cudaFree(gpuT);
	cudaFree(gpuM);

	return (double) ExecutionTime;
}

// Helper function to print results
void printSortResults(int *M, int* T, const int arraySizeT, const int nb_arrays_to_srt) {
	int i, j;

	for (i=0; i<nb_arrays_to_srt; i++)
	{
		printf("Tab :");
		printArray(&(T[i*arraySizeT]), arraySizeT);
		printf("Sorted Tab :");
		printArray(&(M[i*arraySizeT]), arraySizeT);
		printf("\n");
	}
}
// Helper function to create arrays
void createRandVector(int *pt, int size_T)
{
	for(int j = 0; j < size_T; j++)
	{
		pt[j] = (rand() % 200);
	}
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

