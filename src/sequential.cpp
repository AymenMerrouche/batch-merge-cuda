#include <chrono>
#include "sortedArray.h"

#include "sequential.h"

// Sequential Merge
void sequentialMerge(int *M, int* A, int* B, const int sizeA, const int sizeB) {
	int j = 0;
	int i = 0;
	int sizeM = sizeA + sizeB;

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


void sequentialMergeRecursion(int *M, int *T, int start, int stop, int size) {
	if ((stop - start) < 2) return;
	int mid = start + (stop - start) / 2;
	sequentialMergeRecursion(T, M, start, mid, size); // Alternate between T and M
	sequentialMergeRecursion(T, M, mid, stop, size);

	sequentialMerge(M + start, T + start, T + mid, mid - start, stop - mid);
}

void sequentialMergeSort(int *M, int *T, int *temp, const int sizeT){
	int start = 0;
	int stop = sizeT;

	// Copy initial values to the array to sort inplace
	for (int j = 0; j < sizeT; j++) {
		M[j] = T[j];
		temp[j] = T[j];
	}

	sequentialMergeRecursion(M, temp, start, stop, stop - start);
}

double batchSequentialMergeSort(int *M, int *T, int *temp, const int sizeT, const int nb_tabs){
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < nb_tabs; i++) {
		sequentialMergeSort(M + i * sizeT, T + i * sizeT, temp + i * sizeT, sizeT);
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	return duration.count();
}