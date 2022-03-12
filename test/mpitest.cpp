#include<mpi.h>
#include<stdio.h>


int main(int argc, char* argv[])
{
    int ierr;
    int iproc;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
    printf("IP = %d\n", iproc);
    MPI_Finalize();
    return 0;
}