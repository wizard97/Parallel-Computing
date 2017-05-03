/**** COMPILING ***/
//mpicc -O3 -Wall -std=gnu99 daw268_hw3_part_1.c -o p1

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
#include <omp.h>
#include <time.h>
#include <unistd.h>


/**** MACROS TO PLAY WITH ******/
#define NUM_RUNS (1<<10)
#define N 1024
/******************************/


#define DEBUG(S, ...) printf("DEBUG (rank %d of %d): " S "\n", myrank, nprocs, ##__VA_ARGS__)
// Only the rank 0 process debugs
#define DEBUGR0(S, ...) if(ROOT()) DEBUG(S, ##__VA_ARGS__)

// The root node has rank 0
#define ROOT() !myrank


double riemann_rectangle(double xstart, double xend, double ystart, double yend,
        uint32_t steps_x, uint32_t steps_y, double (*f)(double, double, void*), void *args);

double hemisphere(double x, double y, void *args);

static int myrank, nprocs;

int main(int argc, char **argv)
{
    MPI_Comm cart;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);


    int name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &name_len);
    DEBUG("Node '%s' (rank %d) online and reporting for duty!", processor_name, myrank);


    FILE *fd;
    if (ROOT()) {
    	char name[20];
    	snprintf(name, sizeof(name), "%d_riemann.csv", nprocs);
    	fd = fopen(name, "w");
    	fprintf(fd, "Length,ns/byte\n");

    }


    double start=MPI_Wtime(); /*start timer*/


	//get data
    float r = 1;
	double res = riemann_rectangle(-1, 1, -1, 1, 100, 100, &hemisphere, &r);
	//MPI_Reduce(&avg_runtime, &res, 1,MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);

    printf("Res: %f\n", res);

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

    for (uint32_t i=0; i < steps_x; i++)
    {
        for (uint32_t j=0; j < steps_y; j++)
        {
            double x = xstart + i*dx;
            double y = ystart + j*dx;

            double f_avg = (f(x,y,args) + f(x+dx,y,args) + f(x,y+dy,args) + f(x+dx,y+dy,args))/4;
            res += f_avg*dx*dy;
        }
    }

    return res;
}


double hemisphere(double x, double y, void *args)
{
    const double r = *(double*)args;

    if (x*x + y*y <= r*r)
        return sqrt(r*r - x*x - y*y);

    return 0;
}
