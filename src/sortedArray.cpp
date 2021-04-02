#include "sortedArray.h"

#include <ctime> 
#include <cstdlib>
#include <stdio.h>
#include <iostream>

void createSortedArray(int *pt, int arraySize)
{
	//printf("%d\n", UPPER_RANDOM);
	int el = (std::rand() % UPPER_RANDOM);
	for (int i = 0; i < arraySize;  i++) {
		//printf("%d\n", el);
		pt[i] = el;
		el += (std::rand() % UPPER_RANDOM);
	}
}

void printArray(int *pt, int arraySize) {
	printf("{");
	for (int j = 0; j < arraySize; j++) {
		printf("%d,", pt[j]);
	}
	printf("}\n");
}

void createRandVector(int *pt, int size_T)
{
	for (int j = 0; j < size_T; j++)
	{
		pt[j] = (rand() % 200);
	}
}