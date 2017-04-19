// By Daniel Aaron Wisner, daw268, compile with "gcc -std=c99 daw268_problem1.c -o p2", run with ./p2
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

// define the stride size in number of floats
#define STRIDE 1
// Max size of N
#define MAX_LENGTH ((uint64_t)1<<26)
// Min size of N
#define MIN_SIZE ((uint64_t)1<<10)
// Number of runs to avg
#define NUM_RUNS 10

int main(int *argc, char **argv)
{
  
    // Allocate the massive array
    float *data = malloc(sizeof(float)*MAX_LENGTH);

    if(!data)
    {
	printf("Malloc return NULL!\n");
	exit(1);
    }

    
    //open file to save csv
    char str[80];

    FILE *fp = NULL;
    
    for (uint64_t n = MIN_SIZE; n < MAX_LENGTH; n <<=1)
    {
        // Create csv file for each N size
	snprintf(str, sizeof(str),"data/n_%lu.csv", n);
	if (fp) fclose(fp);
	fp = fopen(str, "w");

	if (!fp)
	{
	    printf("Failed to create file");
	    exit(1);
	}

	fprintf(fp,"\"Stride (floats)\",\"elements touched\",\"tavg (ns) @ n=%lu\"\n", n);
	
    
	printf("Running %lu of %lu\n", n, MAX_LENGTH);

	// Run though each stride length
	for (uint64_t stride = STRIDE; stride <= n>>1; stride <<= 1)
	{

	    //Start clock
	    clock_t start = clock();
	    // Run stride*NUM_RUNS times
	    for (uint64_t r=0; r < stride*NUM_RUNS; r++)
	    {
		//touch the elements
		for (uint64_t i=0; i < n; i+=stride)
		{
		    data[i] = data[i]*3.14;
		}

	    }
	    clock_t rt = clock() - start;
	    // Calculate avg time
	    float avg_rt = (1000000000*(float)rt)/((float)n*NUM_RUNS*CLOCKS_PER_SEC);
	    printf("\tStride %lu of %lu (%f nS) (%lu elements)\n", stride, n/2, avg_rt, n/stride);
 
	    fprintf(fp, "%lu,%lu,%f\n", stride, n/stride, avg_rt);
	}
    }

    free(data);
    return 0;

}
