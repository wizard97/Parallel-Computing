// By Daniel Aaron Wisner, daw268
// compile with "gcc -std=c99 daw268_problem1.c -o p1", run with ./p1
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

// Number of times to repeat (averaged at end)
#define NUM_RUNS 10

// Define dimension of vector
#define K 15
// Number of datapoints
#define N (1<<20)

#define ELEMENTS K*N

// Populate with random data
void populate(float *const data, const uint32_t size);

// Calculate centroid using method 1
float *centroid_1(float *const res, const float *data,
		  const uint32_t rows, const uint32_t cols);

// Calc centroid using method 2
float *centroid_2(float *const res, const float *data,
		  const uint32_t rows, const uint32_t cols);

// Print the contents of a vector
void print_vector(float *const data, const uint32_t size);

int main(int *argc, char **argv)
{
    // Allocate the memory
    float *const data = malloc(sizeof(float)*(ELEMENTS+K));

    float *res = data + ELEMENTS;
  
    if (!data) {
	printf("Malloc failed\n");
	exit(1);
    }

    // Seed rand
    time_t t;
    srand((unsigned int) time(&t));
    // Fill with random data
    populate(data, ELEMENTS);


    //Calculate using 1st method
    clock_t diff1, diff2, c2, c1 = clock();
    for (uint16_t i = 0; i< NUM_RUNS;i++)
    {
	centroid_1(res, data, N, K);
    }
    diff1 = clock() - c1;
    // Calculate avg time
    float c1_s = ((float)diff1)/(CLOCKS_PER_SEC);
    printf("Centroid 1: %f secs\n", c1_s);

   
    // Calculate again using second method
    c2 = clock();
    for (uint16_t i=0; i < NUM_RUNS; i++)
    {
	centroid_2(res, data, N, K);
    }
    diff2 = clock() - c2;
    // Calculate using 2nd method
    float c2_s = ((float)diff2)/(CLOCKS_PER_SEC);
    printf("Centroid 2: %f secs\n", c2_s);
    
    //print_vector(res, K);

    free(data);
    return 0;
}



float *centroid_1(float *const res, const float *data,
		  const uint32_t rows, const uint32_t cols)
{
    //zero out res
    for (uint32_t i=0; i < cols; i++)
    {
	res[i] = 0;
    }
  
    // iterate through rows
    for (uint32_t i=0; i < rows; i++)
    {
        // Iterate through columns
	for (uint32_t j=0; j < cols; j++)
	{
	    res[j] += data[cols*i + j]; 

	}     
    }

    // Divide by 1/N
    for (uint32_t i=0; i < cols; i++)
    {
	res[i] /= rows;
    }


    return res;
}



float *centroid_2(float *const res, const float *data,
		  const uint32_t rows, const uint32_t cols)
{
  
    //zero out res
    for (uint32_t i=0; i < cols; i++)
    {
	res[i] = 0;
    }
  
    // iterate through cols
    for (uint32_t j=0; j < cols; j++)
    {
        // Iterate through rows
	for (uint32_t i=0; i < rows; i++)
	{
	    res[j] += data[cols*i + j]; 

	}     
    }

    // Divide by 1/N

    for (uint32_t i=0; i < cols; i++)
    {
	res[i] /= rows;
    }

    return res;
}


void populate(float *const data, const uint32_t size)
{
    for (uint32_t i=0; i < size; i++)
    {
	data[i] = ((float)rand())/((float)RAND_MAX);
    }

}


void print_vector(float *const data, const uint32_t size)
{
    printf("<");

    for (uint32_t i=0; i < size - 1; i++)
    {
	printf("%f, ", data[i]);
    }


    printf("%f>\n", data[size-1]);

}
