// Compile with 
// gcc daw268_hw2_openmp_sort.c -o p2 -fopenmp -O3 -std=gnu99 -Wall -lrt
// Run with ./p2
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

// Number of elements
#define N_START (1<<8)
#define N_END (1<<14)

#define NUM_RUNS 4
// Number of threads, usually num_cores - 1
#define NUM_THREADS 3
// When recursive msort switches to isort
#define MSORT_BASE_CASE_ISORT 24
//When parallell recursive merge switches to serial merge
#define MERGE_MT_BASE_SERIAL 1000
//Enable Merge multithreading (uncomment to disable)
#define MERGE_MT


#ifdef MERGE_MT
#define MMT_ENABLE 'Y'
#else
#define MMT_ENABLE 'N'
#endif

#define SRAND_VAL 8
#define BILLION 1000000000L

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(a,b) (((a)>(b))?(a):(b))

void fill_crap(uint64_t dim, float data[dim][dim]);
void print_matrix(uint64_t dim, float data[dim][dim]);
void print_addr_matrix(const uint64_t dim, float (**rows)[dim]);


void perm_2_mt(const uint32_t dim, float (**rows)[dim]);
    
void r_merge_sort_mt(const uint32_t dim, float (**rows)[dim], const uint32_t col, const uint32_t l, 
		     const uint32_t h);

void r_insert_sort(const uint32_t dim, float (**rows)[dim], const uint32_t col, const uint32_t l, const uint32_t h);


void r_merge(const uint32_t dim, float (**rows)[dim], float (**p1)[dim], float (**p2)[dim], const uint32_t col,
	     const uint32_t l,  uint32_t n1,  uint32_t n2);

void r_merge_mt(const uint32_t dim, float (**rows)[dim], float (**a)[dim], float (**b)[dim], const uint32_t col,
		const uint32_t l,  uint32_t la,  uint32_t lb);

uint32_t r_binary_search(const uint32_t dim, float (**rows)[dim], const uint32_t col,
			 uint32_t l, uint32_t h, const float val);

void perm_2(const uint32_t dim, float (**rows)[dim]);;

void r_merge_sort(const uint32_t dim, float (**rows)[dim], const uint32_t col, const uint32_t l, const uint32_t h);
	    
int main(int argc, char **argv)
{
    FILE *fp = NULL;
    char str[50];

    // Create csv file for each N size
    snprintf(str, sizeof(str),"data/p2_%dthds_MMT%c.csv", NUM_THREADS, MMT_ENABLE);
    fp = fopen(str, "w");
    fprintf(fp, "N,\"1 Thread (ns)\",\"%d Threads (ns)\",\"Speedup\"\n", NUM_THREADS);
    
    //omp_set_dynamic(1);
    omp_set_num_threads(NUM_THREADS);
    
    srand(SRAND_VAL);


    //print_addr_matrix(N, rows);
    printf("Starting\n\n\n");
    
    uint64_t t1 =0, t2=0;
    struct timespec start, end;
     
    for (uint32_t n=N_START; n<=N_END; n <<= 1)
    {
	
	//pointer to array of N floats
	//data[0] = float* to first row
	float (*data)[n] = malloc(sizeof(float)*n*n);
	// pointer to an array of N pointers that point to N floats
	// rows[0] = (float *)[N], address of first first row
	float (**rows)[n] = malloc(sizeof(float (*)[])*n);

	if(!data || !rows) {
	    printf("malloc() failed\n");
	    exit(1);
	}

	fill_crap(n, data);
	
	for (uint16_t r=0;r < NUM_RUNS; r++)
	{
	    //faster to work with pointers to rows than actual rows
	    for (uint32_t i=0; i< n; i++) {
		rows[i] = &data[i];
	    }

	    //single threaded
	    clock_gettime(CLOCK_MONOTONIC, &start);
	    perm_2(n, rows);	
	    clock_gettime(CLOCK_MONOTONIC, &end);
	    t1 += BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;

	    //faster to work with pointers to rows than actual rows
	    for (uint32_t i=0; i< n; i++) {
		rows[i] = &data[i];
	    }	

	    // multithreaded
	    clock_gettime(CLOCK_MONOTONIC, &start);
	    perm_2_mt(n, rows);	
	    clock_gettime(CLOCK_MONOTONIC, &end);
	    t2 += BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
    
	}

	    
	t1 /= NUM_RUNS;
	t2 /= NUM_RUNS;

	//normalize
	float c1_s = ((float)t1)/(n*n);
	float c2_s = ((float)t2)/(n*n);
	float su = (c1_s-c2_s)/c1_s;
    
	printf("N = %u\nSingle threaded: %f ns\nMulti threaded(%d): %f ns\n", n, c1_s, NUM_THREADS, c2_s);
	printf("Speedup: %.2f%%\n\n", 100*su);
	fprintf(fp, "%u,%f,%f,%f\n", n, c1_s, c2_s, su);

	free(data);
	free(rows);

    }


    fclose(fp);

    return 0;
}


void perm_2_mt(const uint32_t dim, float (**rows)[dim])
{
    for (uint32_t i=0; i< dim; i++) {

	r_merge_sort_mt(dim, rows, i, i, dim);
	//r_insert_sort(dim, rows, i, i, dim);
    }

}


void perm_2(const uint32_t dim, float (**rows)[dim])
{
    
    for (uint32_t i=0; i< dim; i++) {
	r_merge_sort(dim, rows, i, i, dim);
    }

}



void r_merge(const uint32_t dim, float (**rows)[dim], float (**p1)[dim], float (**p2)[dim], const uint32_t col,
	     const uint32_t l,  uint32_t n1,  uint32_t n2)
{


    uint32_t i=0, j=0, x=l;
    while (i < n1 && j < n2) {
	
	if ((*p1[i])[col] > (*p2[j])[col]) {
	    rows[x] = p1[i++];
	} else {
	    rows[x] = p2[j++];
	}
	x++;
    }

    
    // copy the remainder
    while (i < n1) {
	rows[x++] = p1[i++];
    }

    while (j < n2) {
	rows[x++] = p2[j++];
    }


}



void r_merge_mt(const uint32_t dim, float (**rows)[dim], float (**a)[dim], float (**b)[dim], const uint32_t col,
		const uint32_t l,  uint32_t la,  uint32_t lb)
{
 
    
    // a and la  must be largest
    if (lb > la) {
	uint32_t swp = la;
	la = lb;
	lb = swp;
	
        float (**pswp)[dim] = a;
	a = b;
	b = pswp;
    }


    if (la <= MERGE_MT_BASE_SERIAL) {
	r_merge(dim, rows, a, b, col, l, la, lb);
	return;
    }

    
    //r is incluseive in p1
    uint32_t r = (la)/2;

    //binary search a[r] in p2

    // 0 - lb
    uint32_t s = r_binary_search(dim, b, col, 0, lb, (*a[r])[col]);
    uint32_t offset = (r) + s; //offset

    rows[l + offset] = a[r];

#pragma omp parallel sections
    {

#pragma omp section
	{
	    r_merge_mt(dim, rows, a, b, col, l, r, s);
	}

#pragma omp section
	{
	    r_merge_mt(dim, rows, a + r  + 1, b+s, col, l+(offset+1) ,la-(r+1), lb - s);
	    
	}
	
    }
    

}


uint32_t r_binary_search(const uint32_t dim, float (**rows)[dim], const uint32_t col,
			 uint32_t l, uint32_t h, const float val)
{
    /* For debugging a linear search implimentation
       uint32_t n = h-l;
       uint32_t m = l + n/2;


       uint32_t tmp = h;
    
       for (uint32_t i=l; i < h; i++) {
       if (val > (*rows[i])[col])
       {
       tmp = i;
       break;
       }
	    
       }
       return tmp;
    
    */
	
    h--;

    h = MAX(l, h+1);
    
    while (l < h)
    {
	uint32_t m  = (l + h) / 2;
	if (val >= (*rows[m])[col])
	    h=m;
	else
	    l = m+1;
    }
    return h;
    

    
}


void r_merge_sort_mt(const uint32_t dim, float (**rows)[dim], const uint32_t col, const uint32_t l, const uint32_t h)
{
    uint32_t n = h-l;

    //base case
    
    if (n <= MSORT_BASE_CASE_ISORT) {
	r_insert_sort(dim, rows, col, l, h);
	return;
    }

    uint32_t s = n/2;
    uint32_t m = l + s;
    
#pragma omp parallel sections
    {
#pragma omp section
	r_merge_sort_mt(dim, rows, col, l, m);

#pragma omp section
	r_merge_sort_mt(dim, rows, col, m, h);
    }
    
       

    
    //r_merge(dim, rows, col, l, m, h);
    float (**p1)[dim] = malloc(n*sizeof(float (*)[]));
    float (**p2)[dim] = p1 + s;

    
    if (!p1)
	exit(1);

    memcpy(p1, &rows[l], sizeof(float (*)[])*n);

    
#ifdef MERGE_MT
    r_merge_mt(dim, rows, p1, p2, col, l, s, n-s);
#else
    r_merge(dim, rows, p1, p2, col, l, s, n-s);
#endif

    free(p1);
    
}




void r_merge_sort(const uint32_t dim, float (**rows)[dim], const uint32_t col, const uint32_t l, const uint32_t h)
{
    uint32_t n = h-l;

    //base case
    
    if (n <= MSORT_BASE_CASE_ISORT) {
	r_insert_sort(dim, rows, col, l, h);
	return;
    }

    uint32_t s = n/2;
    uint32_t m = l + s;
    

    r_merge_sort_mt(dim, rows, col, l, m);


    r_merge_sort_mt(dim, rows, col, m, h);
    
    
    
    //r_merge(dim, rows, col, l, m, h);
    float (**p1)[dim] = malloc(n*sizeof(float (*)[]));
    float (**p2)[dim] = p1 + s;

    
    if (!p1)
	exit(1);

    memcpy(p1, &rows[l], sizeof(float (*)[])*n);

    r_merge(dim, rows, p1, p2, col, l, s, n-s);

    free(p1);   
}



void r_insert_sort(const uint32_t dim, float (**rows)[dim], const uint32_t col, const uint32_t l, const uint32_t h)
{
    
    for (uint32_t i = l+1; i < h; i++) {
      
	uint32_t j = i;
	while (j > l && (*rows[j])[col] > (*rows[j-1])[col]) {
	    //Swapping
	    float (*tmp)[dim] = rows[j];
	    rows[j] = rows[j-1];
	    rows[j-1] = tmp;
	
	    j--;
	}

    } 
}





void fill_crap(uint64_t dim, float data[dim][dim])
{   
    for (uint64_t i=0; i<dim; i++) {

	for (uint64_t j=0; j < dim; j++) {
	    data[i][j] = ((float)rand())/((float)RAND_MAX);
	}

    }

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


void print_addr_matrix(const uint64_t dim, float (**rows)[dim])
{
    for (uint64_t i=0; i<dim; i++) {

        for (uint64_t j=0; j < dim-1; j++) {
            printf("%.4f, ", (*rows[i])[j]);
        }
        printf("%.4f\n", (*rows[i])[dim-1]);
    }
}
