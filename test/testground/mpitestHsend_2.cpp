#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <cassert>

// mpicxx.openmpi -o mpitestHsend_2.exe mpitestHsend_2.cpp -Wall
// mpirun.openmpi -np 2 ./mpitestHsend_2.exe

int bufsize = 36864 * 100;
int Nlocal = 1024 * 1024;
int dCount = 1024;
int NREPEAT = 100;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    assert(size == 2);
    

    double *data = new double[Nlocal];
    double *dataRecv = new double[Nlocal];
    for (int i = 0; i < Nlocal; i++)
        data[i] = i + rank * Nlocal;
    for (int i = 0; i < Nlocal; i++)
        dataRecv[i] = 0;

    int dSize = sizeof(double);

    
    uint8_t *buf = new uint8_t[bufsize];
    MPI_Buffer_attach(buf, bufsize);

    if (rank == 0)
    {
        MPI_Datatype typePullRecv, typePushRecv;
        {
            int b[1];
            MPI_Aint d[1];
            int dAlt[1];
            b[0] = dCount * dSize;
            d[0] = 0;
            dAlt[0] = 0;
            // MPI_Type_create_hindexed(1, b, d, MPI_UINT8_T, &typePullRecv);
            MPI_Type_indexed(1, b, dAlt, MPI_UINT8_T, &typePullRecv);
        }
        {
            int *b = new int[dCount];
            MPI_Aint *d = new MPI_Aint[dCount];
            int *dAlt = new int[dCount];
            for (int i = 0; i < dCount; i++)
                b[i] = dSize, d[i] = dSize * i * 2, dAlt[i] = i * 2 * dSize;
            // MPI_Type_create_hindexed(dCount, b, d, MPI_UINT8_T, &typePushRecv);
            MPI_Type_indexed(dCount, b, dAlt, MPI_UINT8_T, &typePushRecv);
            delete[] d;
            delete[] b;
            delete[] dAlt;
        }

        MPI_Type_commit(&typePullRecv);
        MPI_Type_commit(&typePushRecv);
        MPI_Request req0, req1, req2;
        MPI_Status stat;

        MPI_Bsend_init(data, 1, typePushRecv, 0, 123, MPI_COMM_WORLD, &req1);
        MPI_Bsend_init(data, 1, typePushRecv, 1, 123, MPI_COMM_WORLD, &req2);
        MPI_Recv_init(dataRecv, 1, typePullRecv, 0, 123, MPI_COMM_WORLD, &req0);

        for (int i = 0; i < NREPEAT; i++)
        {
            MPI_Start(&req1);
            MPI_Start(&req2);
            MPI_Start(&req0);
            MPI_Wait(&req1, &stat);
            MPI_Wait(&req2, &stat);
            MPI_Wait(&req0, &stat);
            std::cout << "Done " << i << std::endl;
        }

        MPI_Request_free(&req0);
        MPI_Request_free(&req1);
        MPI_Request_free(&req2);
        MPI_Type_free(&typePullRecv);
        MPI_Type_free(&typePushRecv);
    }
    else if (rank == 1)
    {
        MPI_Datatype typePullRecv, typePushRecv;
        {
            int b[1];
            MPI_Aint d[1];
            b[0] = dCount * dSize;
            d[0] = 0;
            MPI_Type_create_hindexed(1, b, d, MPI_UINT8_T, &typePullRecv);
        }
        {
            int *b = new int[dCount];
            MPI_Aint *d = new MPI_Aint[dCount];
            for (int i = 0; i < dCount; i++)
                b[i] = dSize, d[i] = dSize * i;
            MPI_Type_create_hindexed(dCount, b, d, MPI_UINT8_T, &typePushRecv);
            delete[] d;
            delete[] b;
        }

        MPI_Type_commit(&typePullRecv);
        MPI_Type_commit(&typePushRecv);
        MPI_Request req0;
        MPI_Status stat;

        MPI_Recv_init(dataRecv, 1, typePullRecv, 0, 123, MPI_COMM_WORLD, &req0);

        for (int i = 0; i < NREPEAT; i++)
        {
            MPI_Start(&req0);
            MPI_Wait(&req0, &stat);
        }

        MPI_Request_free(&req0);
        MPI_Type_free(&typePullRecv);
        MPI_Type_free(&typePushRecv);
    }

    std::cout << "rank " << rank << "Done " << std::endl;

    std::cout << std::setprecision(14);
    if (rank == 0)
    {
        // for (int i = 0; i < dCount * 2; i++)
        //     std::cout << rank << " data " << i << ": " << data[i] << std::endl;
        for (int i = 0; i < dCount * 2; i++)
            std::cout << rank << " dataRecv " << i << ": " << dataRecv[i] << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1)
    {
        // for (int i = 0; i < dCount * 2; i++)
        //     std::cout << rank << " data " << i << ": " << data[i] << std::endl;
        for (int i = 0; i < dCount * 2; i++)
            std::cout << rank << " dataRecv " << i << ": " << dataRecv[i] << std::endl;
    }

    delete[] data;
    delete[] dataRecv;
    uint8_t *obuf;
    int obufsize;
    MPI_Buffer_detach(&obuf, &obufsize);
    delete[] buf;
    MPI_Finalize();

    return 0;
}