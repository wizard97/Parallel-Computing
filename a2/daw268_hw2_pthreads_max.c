// Compile with 
// gcc daw268_hw2_pthreads_max.c -o p1 -lpthread -O3 -std=gnu99 -Wall -lrt
// Run with ./p1
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

//params
#define N_START (1<<8)
#define N_END (1 << 15)
#define NUM_THREADS 3
#define NUM_RUNS 5


#define SRAND_VAL 8
#define BILLION 1000000000L

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))


// data struct to pass inwith pthread_create
typedef struct perm_w1_mt_arg
{
    uint64_t dim, col, r_start, r_end, k_max; 
    float *data;
} perm_w1_mt_arg_t;



// helpers
void swap_row(uint64_t dim, float data[][dim], uint64_t r1, uint64_t r2);
void print_matrix(uint64_t dim, float data[dim][dim]);
void fill_crap(uint64_t dim, float data[dim][dim]);


// implimentations
void perm_w1(const uint64_t dim, float data[dim][dim]);
void perm_w1_mt(const uint64_t dim, float data[dim][dim]); //multithreaded


uint64_t perm_w1_findmax(const uint64_t dim, const float data[dim][dim], const uint64_t idx); //st
uint64_t perm_w1_mt_findmax(const uint64_t dim, const float data[dim][dim], const uint64_t idx);
void *perm_w1_mt_findmax_th(void *threadarg); //mt helper

int main(int argc, char **argv)
{
    FILE *fp = NULL;
    char str[50];

    // Create csv file for each N size
    snprintf(str, sizeof(str),"data/p1_%dthds.csv", NUM_THREADS);
    fp = fopen(str, "w");
    fprintf(fp, "N,\"1 Thread (ns)\",\"%d Threads (ns)\",\"Speedup\"\n", NUM_THREADS);
    
    for (uint64_t n = N_START; n<=N_END; n<<=1)
    {
	
	float (*data)[n] = malloc(sizeof(float)*n*n);
	float (*data2)[n] = malloc(sizeof(float)*n*n);

	if (!data || !data2)
	{
	    printf("Malloc() failed\n");
	    exit(1);
	}

    
	srand(SRAND_VAL);
	fill_crap(n, data);
	//print_matrix(N, data);

	//copy
	memcpy(data2, data, sizeof(float)*n*n);

	struct timespec start, end;
	uint64_t t1=0, t2=0;

	for (uint16_t i=0; i < NUM_RUNS; i++)
	{
	    // copy from backup
	    memcpy(data2, data, sizeof(float)*n*n);

	    //Run the single threaded version
	    clock_gettime(CLOCK_MONOTONIC, &start);
	    perm_w1(n, data2);
	    clock_gettime(CLOCK_MONOTONIC, &end);
	    t1 += BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;

	    // copy from backup
	    memcpy(data2, data, sizeof(float)*n*n);

	    // Run the multithreaded version
	    clock_gettime(CLOCK_MONOTONIC, &start);
	    perm_w1_mt(n, data2);
	    clock_gettime(CLOCK_MONOTONIC, &end);
	    t2 += BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	
	}
    
	t1 /= NUM_RUNS;
	t2 /= NUM_RUNS;

	//normalize
	float c1_s = ((float)t1)/(n*n);
        float c2_s = ((float)t2)/(n*n);
        float su = (c1_s-c2_s)/c1_s;
    
	printf("N = %lu\nSingle threaded: %f ns\nMulti threaded(%d): %f ns\n", n, c1_s, NUM_THREADS, c2_s);
	printf("Speedup: %.2f%%\n\n", 100*su);
	fprintf(fp, "%lu,%f,%f,%f\n", n, c1_s, c2_s, su);
    }
    fclose(fp);
    pthread_exit(0);
}


void perm_w1(const uint64_t dim, float data[dim][dim])
{
    for (uint64_t i=0; i < dim; i++) {
      uint64_t k_max = perm_w1_findmax(dim, (const float (*)[])data, i);

        //swap_row if larger row found
        if (k_max != i)
            swap_row(dim, data, i, k_max);
    }
}

//multithreaded
void perm_w1_mt(const uint64_t dim, float data[dim][dim])
{
    for (uint64_t i=0; i < dim; i++) {
      uint64_t k_max = perm_w1_mt_findmax(dim, (const float (*)[])data, i);

        //swap_row if larger row found
        if (k_max != i)
            swap_row(dim, data, i, k_max);
    }
}



uint64_t perm_w1_mt_findmax(const uint64_t dim, const float data[dim][dim], const uint64_t idx)
{   
    pthread_t threads[NUM_THREADS];
    perm_w1_mt_arg_t targs[NUM_THREADS];
    int rc;

    uint64_t elems = dim - idx;
  
    // if less than number of threads call single threaded version
    if (elems < 2*NUM_THREADS)
	return perm_w1_findmax(dim, data, idx);

    
    // figure out chunk size
    uint64_t cs = elems/NUM_THREADS;
    uint64_t cs_m = elems%NUM_THREADS;
    
    // init thread args

    for (uint64_t t=0, start = idx, stop = idx + cs+cs_m; t < NUM_THREADS; t++, start += cs, stop+=cs) {
	targs[t].dim = dim;
	targs[t].col = idx;
	targs[t].r_start = start;
	targs[t].r_end = stop;
	targs[t].data = (float*)data;

	rc = pthread_create(&threads[t], NULL, &perm_w1_mt_findmax_th, &targs[t]);
	if (rc) {
	    printf("ERROR; return code from pthread_create() is %d\n", rc);
	    exit(rc);
	}
	    
    }

    void *status;
    uint64_t k_max = 0;
    
    for(long t=0; t< NUM_THREADS; t++) {
       rc = pthread_join(threads[t], &status);
       if (rc) {
          printf("ERROR; return code from pthread_join() is %d\n", rc);
          exit(-1);
          }

       if (!t) {
	   k_max = targs[t].k_max;
       } else if (data[targs[t].k_max][idx] > data[k_max][idx] ) {
	   k_max = targs[t].k_max;
       }
       //printf("Main: completed join with thread %ld having a status of %ld\n", t,(long)status);
     }
 

    return k_max;
}


uint64_t perm_w1_findmax(const uint64_t dim, const float data[dim][dim], const uint64_t idx)
{
    uint64_t k_max = idx;

    for (uint64_t k=idx+1; k < dim; k++) {
           //save max
	if (fabsf(data[k][idx]) > data[k_max][idx]) {
	    k_max = k;
	}
    }

    return k_max;
}



void *perm_w1_mt_findmax_th(void *threadarg)
{
    perm_w1_mt_arg_t *arg = (perm_w1_mt_arg_t*)threadarg;

    float (*data)[arg->dim] = (float (*)[arg->dim])arg->data;

    arg->k_max = arg->r_start;
    
    for (uint64_t k = arg->r_start + 1; k < arg->r_end; k++)
    {
	 //save max
	if (fabsf(data[k][arg->col]) > data[arg->k_max][arg->col]) {
                arg->k_max = k;
	}
	
    }

    pthread_exit(NULL); 
}


// Precondition: must be square array
void swap_row(uint64_t dim, float data[dim][dim], uint64_t r1, uint64_t r2)
{
    float tmp[dim];
    // save r1 temp
    memcpy(tmp, &data[r1][0], dim*sizeof(float));
    // copy r2 to r1
    memcpy(&data[r1][0], &data[r2][0], dim*sizeof(float));
    // copy tmp to r2
    memcpy(&data[r2][0], tmp, dim*sizeof(float));
}

void print_matrix(uint64_t dim, float data[dim][dim])
{
    for (uint64_t i=0; i<dim; i++) {

        for (uint64_t j=0; j < dim-1; j++) {
            printf("%.4f, ", data[i][j]);
        }
        printf("%.4f\n", data[i][dim-1]);
    }
}

void fill_crap(uint64_t dim, float data[dim][dim])
{
    for (uint64_t i=0; i<dim; i++) {

        for (uint64_t j=0; j < dim; j++) {
            data[i][j] = ((float)rand())/RAND_MAX;
        }

    }
}
