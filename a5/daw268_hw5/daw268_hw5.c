/**** COMPILING ***/
//mpicc -std=gnu99 -O3 -Wall -fopenmp daw268_hw5.c -lm -o daw268_hw5

/*** RUNNING ******/
/*
mpirun -mca plm_rsh_no_tree_spawn 1 -np <N> -use-hwthread-cpus -hostfile hostfile --map-by node:PE=<X> daw268_hw5 <X> <r> <k_start> <k_end>
Where:
N = number of nodes (processes)
X = number of omp threads per process
r = is the radius of the hemisphere
k_start = Starting number of steps (2^k_start) while integrating
k_end = Ending number of steps (2^k_end) while integrating

Result:
Output saved in .csv file in data folder
<N>nodes_<X>threads_riemann.csv

Or run the test.sh bash script which will test a range of k's, nodes, and OMP threads
*/
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

// colors
#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"
#define RESET KNRM

#define BILLION 1000000000
#define DEFAULT_NUM_OMP_THREADS 4
#define DEFAULT_R ((double)1.0)
#define DEFAULT_K_START 8
#define DEFAULT_K_END 18

/**** MACROS TO PLAY WITH ******/
#define NUM_RUNS 3
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

static int myrank, nprocs, nthreads;

int main(int argc, char **argv)
{
    nthreads = DEFAULT_NUM_OMP_THREADS;
    if (argc >= 2)
        nthreads = atof(argv[1]);

    double r = DEFAULT_R;
    if (argc >= 3)
        r = atof(argv[2]);

    uint32_t k_start = DEFAULT_K_START;
    uint32_t k_end = DEFAULT_K_END;

    if (argc >= 5) {
        k_start = atoi(argv[3]);
        k_end = atoi(argv[4]);
    }


    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    omp_set_num_threads(nthreads);
    omp_set_dynamic(0); //disable dynamic


    DEBUGR0(KGRN "OMP Thds/Proc=%d, r=%f, k=%u-%u\n"KNRM, nthreads, r, k_start, k_end);

    FILE *fd;
    if (ROOT()) {
        char name[50];
        snprintf(name, sizeof(name), "data/%dnodes_%dthreads_riemann.csv", nprocs, nthreads);
        fd = fopen(name, "w");
        fprintf(fd, "k,Error,Runtime ns\n");
        //fprintf(fd, "%u,%u,%lu\n", nprocs, n, rt);

    }

    int name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &name_len);
    DEBUG("Node '%s' (rank %d) online and reporting for duty!", processor_name, myrank);


    struct timespec start, end; //timestamps
    uint64_t rt = 0;


    // divide and conquer!
    double dx = (2*r)/nprocs;
    // integrate from X: [-R + rank*dx, -R + (rank+1)*dx], Y: [-R, R]
    double p_res, res;


    for (uint32_t k = k_start; k <= k_end; k++)
    {
        uint32_t n = 1<<k;
        for (int i=0; i < NUM_RUNS; i++)
        {
            DEBUGR0(KBLU "Starting run k=%u (%d of %d)\n" RESET, k, i+1, NUM_RUNS);
            clock_gettime(CLOCK_MONOTONIC, &start);

            p_res = riemann_rectangle(-r + myrank*dx, -r + (myrank+1)*dx, -r, r, n/nprocs, n, &hemisphere, &r);

            // Reduce answer by summing all partial results, send to root
            MPI_Reduce(&p_res, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            clock_gettime(CLOCK_MONOTONIC, &end);

            rt += get_dt(&start, &end);
        }

        rt /= NUM_RUNS;

        double exs = (2*M_PI*(r*r*r))/3;
        DEBUGR0(KGRN "Results k=%u: %f " KRED "(error: %.2E)" RESET " in %lu ms (avg)\n", k, res, fabs(res-exs), rt/1000000);

        if (ROOT())
            fprintf(fd, "%u,%.2E,%lu\n", k, fabs(res-exs), rt);
    }

    //Done!
    if (ROOT())
	   fclose(fd);

    DEBUGR0(KGRN "Completed test!"KNRM);
    DEBUGR0(KGRN"OMP Thds/Proc=%d, r=%f, k=%u-%u\n"KNRM, nthreads, r, k_start, k_end);

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
