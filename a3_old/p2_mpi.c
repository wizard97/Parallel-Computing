// Same socket 16 procs: mpirun -mca plm_rsh_no_tree_spawn 1 --map-by ppr:16:socket -bind-to core -np 16 ./p1
// Between sockets 16 procs: mpirun -mca plm_rsh_no_tree_spawn 1 --map-by socket -bind-to core -np 16 ./p1
// Between nodes 16 procs: mpirun -mca plm_rsh_no_tree_spawn 1 --map-by node -bind-to core -np 16 -host en-openmpi00,en-openmpi02 ./p1
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <mpi.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <string.h>

#define MAX_NUM_SWEEPS 50
#define NUM_RUNS (1<<18)
//max msg length in bytes
#define MAX_MSG_LEN (1<<20)
#define SEED_RAND 97

#define N 5
#define M 10
#define NUM_PROCS 2
#define MAX_SWEEPS 10

#define MAX_COLS (NORMAL_COLS + NON_DIV_COLS) //max cols in a packet
#define NORMAL_COLS (N/(2*NUM_PROCS))
#define NON_DIV_COLS (N%(2*NUM_PROCS))


#define DEBUG(S, ...) printf("DEBUG (rank %d of %d): " S "\n", cord, nproc, ##__VA_ARGS__)
// Only the rank 0 process debugs
#define DEBUGR0(S, ...) if(ROOT()) DEBUG(S, ##__VA_ARGS__)

// The root node has cord 0
#define ROOT() !cord
#define SIGN(X) (-1 + (2*(X >= 0)))


int           blocklengths[] = {M*MAX_COLS, N*MAX_COLS, 1, 1};
MPI_Datatype types[] = {MPI_DOUBLE, MPI_DOUBLE, MPI_UNSIGNED, MPI_UNSIGNED};
MPI_Datatype mpi_colblock_type;


typedef struct ColBlock
{
    double A_s[M][MAX_COLS];
    double V_s[N][MAX_COLS];
    unsigned int ncols;
    unsigned int block_id;
} colblock_t;


static int myrank, nproc;
static MPI_Comm comm_cart;
static int cord;

void print_matrix(uint64_t m, uint64_t n, double data[m][n], bool matlab);
void init_MPI(int *argc, char ***argv);
void jacobi(const uint32_t m, const uint32_t n, double A[m][n], double V[n][n]);
void fill_crap(uint64_t m, uint64_t n, double data[m][n]);
void copy_cols(const uint32_t m, const uint32_t n, uint32_t startcol, uint32_t numcols, double A[m][n], double dest[m][numcols]);
void jacobi_blocks(colblock_t *b1, colblock_t *b2);



int main(int argc, char **argv)
{
    DEBUG("Calling init");
    
    srand(SEED_RAND);
    
    init_MPI(&argc, &argv);

    int name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &name_len);
    DEBUG("Hello, my names is %s\n", processor_name);

    double (*A)[N] = (double (*)[N]) malloc(sizeof(double[M][N])); // MxN
    double (*V)[N] = (double (*)[N]) malloc(sizeof(double[N][N])); // NxN
    double (*S) = (double (*)) malloc(sizeof(double[N])); // 1d array of singular values

    if (!A | !V | !S) {
	DEBUG("Malloc failed!!");
    }

    // eye(n)
    for (uint32_t i=0; i < N; i++) {	
        for (uint32_t j=0; j < N; j++) 
	    V[i][j] = 0;	
	V[i][i] = 1;
    }

    fill_crap(M, N, A);   

    
    colblock_t *myblocks = malloc(3*sizeof(colblock_t));
	DEBUG("A:");
	print_matrix(M, N, A, false);
    
    if (ROOT()) {
        copy_cols(M, N, 0, MAX_COLS, A, myblocks[0].A_s);
	copy_cols(N, N, 0, MAX_COLS, V, myblocks[0].V_s);

	myblocks[0].ncols = MAX_COLS;
	myblocks[0].block_id = 0;
    } else {
	copy_cols(M, N, (2*NORMAL_COLS)*cord + NON_DIV_COLS, NORMAL_COLS, A, myblocks[0].A_s);
	copy_cols(N, N, (2*NORMAL_COLS)*cord + NON_DIV_COLS, NORMAL_COLS, V, myblocks[0].V_s);
	
	myblocks[0].ncols = NORMAL_COLS;
	myblocks[0].block_id = 2*cord;
	DEBUG("Setting cord: %d", 2*cord);
    }

    copy_cols(M, N, (2*NORMAL_COLS)*cord + NORMAL_COLS + NON_DIV_COLS, NORMAL_COLS, A, myblocks[1].A_s);
    copy_cols(N, N, (2*NORMAL_COLS)*cord + NORMAL_COLS + NON_DIV_COLS, NORMAL_COLS, V, myblocks[1].V_s);
	
    myblocks[1].ncols = NORMAL_COLS;
    myblocks[1].block_id = 2*cord + 1;


    for (uint32_t i=0; i < MAX_SWEEPS; i++) {
	jacobi_blocks(myblocks, myblocks+1);
	
	int rank_dest, source;
	MPI_Cart_shift(comm_cart, 0, 1, &source, &rank_dest);
	int ts = rand()%2;
	    
	MPI_Sendrecv(myblocks+ts, 1, mpi_colblock_type, rank_dest, 0xFF,
                myblocks+2, 1, mpi_colblock_type, cord-1, 0xFF,
		     comm_cart, MPI_STATUS_IGNORE);
	memcpy(myblocks+ts, myblocks+2, sizeof(colblock_t));
     }
	

    DEBUG("Gathering data\n");


    
    if (ROOT()) {
	colblock_t *res  = malloc(2*nproc*sizeof(colblock_t));
	MPI_Gather(myblocks, 2, mpi_colblock_type, res, 2, mpi_colblock_type, 0, comm_cart);
	//merge the blocks
	for (uint32_t i=0; i<2*nproc;i++) {
	    // we need to recast it
	    double (*Vres)[res[i].ncols] = (double (*)[res[i].ncols])res[i].V_s;
	    double (*Ares)[res[i].ncols] = (double (*)[res[i].ncols])res[i].A_s;

	    // merge A
	    for (uint32_t j=0; j < M; j++) {
		for (uint32_t k=0; k < res[i].ncols; k++) {
 
		    if (!res[i].block_id) {
			A[j][k] = Ares[j][k];
		    } else {
		        
			A[j][NORMAL_COLS*res[i].block_id + NON_DIV_COLS + k] = Ares[j][k];
		    }

		}

	    }
	    
	    //merge V
	    for (uint32_t j=0; j < N; j++) {
		for (uint32_t k=0; k < res[i].ncols; k++) {
		    if (!res[i].block_id) {
			V[j][k] = Vres[j][k];
		    } else {
			V[j][NORMAL_COLS*res[i].block_id + NON_DIV_COLS + k] = Vres[j][k];
		    }
		  
		}

	    }
	    
	}
	DEBUG("Restored A:");
	print_matrix(M, N, A, true);
    } else {
	MPI_Gather(myblocks, 2, mpi_colblock_type, NULL, 2*nproc, mpi_colblock_type,
               0, comm_cart);	
    }
    

	
    MPI_Finalize();
    
}



void copy_cols(const uint32_t m, const uint32_t n,  uint32_t startcol, uint32_t numcols, double A[m][n], double dest[m][numcols])
{
    for (uint32_t i=0; i < m ; i++) { // rows
	for (uint32_t j=0; j < numcols; j++) {
	    dest[i][j] = A[i][startcol+j];
	}
    }
}


void jacobi_blocks(colblock_t *b1, colblock_t *b2)
{
    if(b1->block_id > b2->block_id) {
	colblock_t *tmp = b2;
	b2 = b1;
	b1 = tmp;
    }
    	    // we need to recast it
    double (*A1)[b1->ncols] = (double (*)[b1->ncols])b1->A_s;
    double (*A2)[b2->ncols] = (double (*)[b2->ncols])b1->A_s;
    
    double (*V1)[b1->ncols] = (double (*)[b1->ncols])b1->V_s;
    double (*V2)[b2->ncols] = (double (*)[b2->ncols])b2->V_s;
	    
    double off =  0;
    double on = 0;

    for (uint32_t j=0; j < b1->ncols; j++) {
	for (uint32_t k=j+1; k < b1->ncols; k++) {

	    double a_jj = 0;
	    double a_jk = 0;
	    double a_kk = 0;

	    // calc dot products
	    for (uint32_t i=0; i < M; i++) {
		a_jj += A1[i][j]*A1[i][j]; // col a_j dot a_j
		a_jk += A1[i][j]*A1[i][k]; // col a_j dot a_k
		a_kk += A1[i][k]*A1[i][k]; // col a_k dot a_k
	    }
	    off += 2*(a_jk*a_jk); // add the off diagnol
	    //on += (k == b1->ncols-1) ? (a_jj*a_jj) : 0; //add the on diagnol
	    double tau = (a_kk-a_jj)/(2*a_jk);
	    double t = 1/(tau + SIGN(tau)*sqrt(1+tau*tau));
	    double c = 1/sqrt(1+t*t); //cos
	    double s = c*t; //sin
		
	    //update A
	    for (uint32_t i=0; i < M; i++) {
		//save ith row of A, cols j and k
		double a_jk[2] = {A1[i][j], A1[i][k] };
		A1[i][j] = a_jk[0]*c - a_jk[1]*s;
		A1[i][k] = a_jk[0]*s + a_jk[1]*c;   
	    }
	    //update V
	    for (uint32_t i=0; i < N; i++) {
		//save ith row of V, cols j and k
		double v_jk[2] = {V1[i][j], V1[i][k] };
		V1[i][j] = v_jk[0]*c - v_jk[1]*s;
		V1[i][k] = v_jk[0]*s + v_jk[1]*c;   
	    }
		
	}

	// now do same thing on other block
	for (uint32_t k=0; k < b2->ncols; k++) {

	    double a_jj = 0;
	    double a_jk = 0;
	    double a_kk = 0;

	    // calc dot products
	    for (uint32_t i=0; i < M; i++) {
		a_jj += A1[i][j]*A1[i][j]; // col a_j dot a_j
		a_jk += A1[i][j]*A2[i][k]; // col a_j dot a_k
		a_kk += A2[i][k]*A2[i][k]; // col a_k dot a_k
	    }
	    off += 2*(a_jk*a_jk); // add the off diagnol
	    on += (k == b2->ncols-1) ? (a_jj*a_jj) : 0; //add the on diagnol
	    double tau = (a_kk-a_jj)/(2*a_jk);
	    double t = 1/(tau + SIGN(tau)*sqrt(1+tau*tau));
	    double c = 1/sqrt(1+t*t); //cos
	    double s = c*t; //sin
		
	    //update A
	    for (uint32_t i=0; i < M; i++) {
		//save ith row of A, cols j and k
		double a_jk[2] = {A1[i][j], A2[i][k] };
		A1[i][j] = a_jk[0]*c - a_jk[1]*s;
		A2[i][k] = a_jk[0]*s + a_jk[1]*c;   
	    }
	    //update V
	    for (uint32_t i=0; i < N; i++) {
		//save ith row of V, cols j and k
		double v_jk[2] = {V1[i][j], V2[i][k] };
		V1[i][j] = v_jk[0]*c - v_jk[1]*s;
		V2[i][k] = v_jk[0]*s + v_jk[1]*c;   
	    }
		
	}
    }
    
    // now perform the sweep just on block b2
    for (uint32_t j=0; j < b2->ncols; j++) {
	    for (uint32_t k=j+1; k < b2->ncols; k++) {

		double a_jj = 0;
		double a_jk = 0;
		double a_kk = 0;

		// calc dot products
		for (uint32_t i=0; i < M; i++) {
		    a_jj += A2[i][j]*A2[i][j]; // col a_j dot a_j
		    a_jk += A2[i][j]*A2[i][k]; // col a_j dot a_k
		    a_kk += A2[i][k]*A2[i][k]; // col a_k dot a_k
		}
		off += 2*(a_jk*a_jk); // add the off diagnol
		on += (k == b2->ncols-1) ? (a_jj*a_jj) : 0; //add the on diagnol
		double tau = (a_kk-a_jj)/(2*a_jk);
		double t = 1/(tau + SIGN(tau)*sqrt(1+tau*tau));
	        double c = 1/sqrt(1+t*t); //cos
		double s = c*t; //sin
		
		//update A
		for (uint32_t i=0; i < M; i++) {
		    //save ith row of A, cols j and k
		    double a_jk[2] = {A2[i][j], A2[i][k] };
		    A2[i][j] = a_jk[0]*c - a_jk[1]*s;
		    A2[i][k] = a_jk[0]*s + a_jk[1]*c;   
		}
		//update V
	        for (uint32_t i=0; i < N; i++) {
		    //save ith row of V, cols j and k
		    double v_jk[2] = {V2[i][j], V2[i][k] };
		    V2[i][j] = v_jk[0]*c - v_jk[1]*s;
		    V2[i][k] = v_jk[0]*s + v_jk[1]*c;   
		}
		
	    }
	}

}



void jacobi(const uint32_t m, const uint32_t n, double A[m][n], double V[n][n])
{
    double off =  INFINITY; //positive infinity
    double on = 0;
    // Terminating conditon sqrt(off) < DBL_EPSILON*N*sqrt(on)

    uint32_t num;
    for (num=0; num < MAX_NUM_SWEEPS && (sqrt(off) > DBL_EPSILON*n*sqrt(on)); num++) {
	off = 0;
	on = 0;
	for (uint32_t j=0; j < n; j++) {
	    for (uint32_t k=j+1; k < n; k++) {

		double a_jj = 0;
		double a_jk = 0;
		double a_kk = 0;

		// calc dot products
		for (uint32_t i=0; i < m; i++) {
		    a_jj += A[i][j]*A[i][j]; // col a_j dot a_j
		    a_jk += A[i][j]*A[i][k]; // col a_j dot a_k
		    a_kk += A[i][k]*A[i][k]; // col a_k dot a_k
		}
		off += 2*(a_jk*a_jk); // add the off diagnol
		on += (k == n-1) ? (a_jj*a_jj) : 0; //add the on diagnol
		double tau = (a_kk-a_jj)/(2*a_jk);
		double t = 1/(tau + SIGN(tau)*sqrt(1+tau*tau));
	        double c = 1/sqrt(1+t*t); //cos
		double s = c*t; //sin
		
		//update A
		for (uint32_t i=0; i < m; i++) {
		    //save ith row of A, cols j and k
		    double a_jk[2] = {A[i][j], A[i][k] };
		    A[i][j] = a_jk[0]*c - a_jk[1]*s;
		    A[i][k] = a_jk[0]*s + a_jk[1]*c;   
		}
		//update V
	        for (uint32_t i=0; i < n; i++) {
		    //save ith row of V, cols j and k
		    double v_jk[2] = {V[i][j], V[i][k] };
		    V[i][j] = v_jk[0]*c - v_jk[1]*s;
		    V[i][k] = v_jk[0]*s + v_jk[1]*c;   
		}
		
	    }
	}
    } //endsweep
}



void fill_crap(uint64_t m, uint64_t n, double data[m][n])
{
    for (uint64_t i=0; i<m; i++) {

	for (uint64_t j=0; j < n; j++) {
	    data[i][j] = ((double)rand())/((double)RAND_MAX);
	}

    }

}



void init_MPI(int *argc, char ***argv)
{
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    int dim = NUM_PROCS;
    int periodic = 0x01;
    MPI_Cart_create(MPI_COMM_WORLD, 1, &dim, &periodic, true, &comm_cart);

    if (comm_cart == MPI_COMM_NULL) {
	DEBUG("Got NULL for MPI_Cart_create, exiting");
	MPI_Finalize();
	exit(1);
    }

    MPI_Comm_rank(comm_cart, &myrank);
    MPI_Comm_size(comm_cart, &nproc);
    MPI_Cart_coords(comm_cart, myrank, 1, &cord);
    DEBUG("Topology set to cart.");
    
    //init the new datatype
    MPI_Aint     offsets[4];
    colblock_t *tmp = malloc(sizeof(colblock_t));
    MPI_Address(tmp, &offsets[0]);
    MPI_Address(&tmp->V_s, &offsets[1]);
    MPI_Address(&tmp->ncols, &offsets[2]);
    MPI_Address(&tmp->block_id, &offsets[3]);
    free(tmp);
    
    offsets[1] -=  offsets[0];
    offsets[2] -=  offsets[0];
    offsets[3] -= offsets[0];
    offsets[0] = 0;


    MPI_Type_create_struct(4, blocklengths, offsets, types, &mpi_colblock_type);
    MPI_Type_commit(&mpi_colblock_type);
    
}


void print_matrix(uint64_t m, uint64_t n, double data[m][n], bool matlab)
{
    if (matlab)
	printf("[ ");
    for (uint64_t i=0; i< m; i++) {

        for (uint64_t j=0; j < n-1; j++) {
            printf("%.4f, ", data[i][j]);
        }
        printf(matlab ? "%.4f; " : "%.4f\n", data[i][n-1]);
    }

    if (matlab)
	printf("]\n");
}
