/**** COMPILING ***/
//mpicc -std=gnu99 -O3 -Wall -fopenmp main.c -lm -o a5

/*** RUNNING ******/
// -np can be any even number!
// Node-Node
//mpirun -mca plm_rsh_no_tree_spawn 1 --map-by node -bind-to core -host en-openmpi00,en-openmpi02 -np 2 ./p1

// Socket-Socket
//mpirun -mca plm_rsh_no_tree_spawn 1 --map-by socket -bind-to core -np 2 ./p1

// Core-Core
//mpirun -mca plm_rsh_no_tree_spawn 1 --map-by socket -bind-to core -np 2 ./p1

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <omp.h>
#include <time.h>
#include <unistd.h>

#define BILLION 1000000000


/**** MACROS TO PLAY WITH ******/
#define NUM_RUNS (1<<10)
#define N 10000
#define R ((double)1.0)
/******************************/


#define DEBUG(S, ...) printf("DEBUG (rank %d of %d): " S "\n", myrank, nprocs, ##__VA_ARGS__)
// Only the rank 0 process debugs
#define DEBUGR0(S, ...) if(ROOT()) DEBUG(S, ##__VA_ARGS__)

// The root node has rank 0
#define ROOT() !myrank


double riemann_rectangle(double xstart, double xend, double ystart, double yend,
        uint32_t steps_x, uint32_t steps_y, double (*f)(double, double, void*), void *args);

double hemisphere(double x, double y, void *args);

uint64_t get_dt(struct timespec *start, struct timespec *end);

static int myrank, nprocs;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);


    int name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &name_len);
    DEBUG("Node '%s' (rank %d) online and reporting for duty!", processor_name, myrank);


    struct timespec start, end; //timestamps
    clock_gettime(CLOCK_MONOTONIC, &start);

    // divide and conquer!
    double r = R;
    double dx = (2*r)/nprocs;
    // integrate from X: [-R + rank*dx, -R + (rank+1)*dx], Y: [-R, R]
    double p_res, res;
    p_res = riemann_rectangle(-r + myrank*dx, -r + (myrank+1)*dx, -r, r, N/nprocs, N, &hemisphere, &r);

    // might as well send answer to everysone
    MPI_Allreduce(&p_res, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    clock_gettime(CLOCK_MONOTONIC, &end);

    FILE *fd;
    if (ROOT()) {
    	char name[20];
    	snprintf(name, sizeof(name), "%d_riemann.csv", nprocs);
    	fd = fopen(name, "w");
    	fprintf(fd, "Length,ns/byte\n");

    }

    uint64_t rt = rt = get_dt(&start, &end);

    DEBUGR0("Results: %f in %lu us\n", res, rt/1000);


    //Done!
    if (ROOT())
	   fclose(fd);

    MPI_Finalize();
}



double riemann_rectangle(double xstart, double xend, double ystart, double yend,
        uint32_t steps_x, uint32_t steps_y, double (*f)(double, double, void*), void *args)
{
    // no divide by zero
    if (!steps_x || !steps_y)
        return 0;


    const double dx = (xend - xstart)/(double)steps_x;
    const double dy = (yend - ystart)/(double)steps_y;

    double res = 0;

    # pragma omp parallel for schedule(guided) reduction(+: res)
    for (uint32_t i=0; i < steps_x; i++)
    {
        for (uint32_t j=0; j < steps_y; j++)
        {
            // always recalculate from index i and j to avoid percision errors
            double x = xstart + i*dx;
            double y = ystart + j*dx;

            res += f(x,y,args) + f(x+dx,y,args) + f(x,y+dy,args) + f(x+dx,y+dy,args);
        }
    }

    // do final division once at end to minimize errors
    res *= (dx*dy)/4;

    return res;
}


double hemisphere(double x, double y, void *args)
{
    const double r = *((double*)args);

    if (x*x + y*y < r*r)
        return sqrt(r*r - x*x - y*y);


    return 0;
}




uint64_t get_dt(struct timespec *start, struct timespec *end)
{
    return BILLION*(end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec);
}
