#pragma once
#ifndef SEQUENTIAL_INCLUDED
#define SEQUENTIAL_INCLUDED

void sequentialMerge(int *M, int* A, int* B, const int sizeA, const int sizeB);
void sequentialMergeSort(int *M, int* T, const int sizeT);
double batchSequentialMergeSort(int *M, int *T, int *temp, const int sizeT, const int nb_tabs);
void sequentialMergeRecursion(int *M, int start, int stop);

#endif //SEQUENTIAL_INCLUDED#pragma once
