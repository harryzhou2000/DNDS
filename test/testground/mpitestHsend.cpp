#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <cassert>

// mpicxx.openmpi -o mpitestHsend.exe mpitestHsend.cpp -Wall

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    DNDS_assert(size == 2);
    int Nlocal = 1024 * 1024;
    double *data = new double[Nlocal];
    double *dataRecv = new double[Nlocal];
    for (int i = 0; i < Nlocal; i++)
        data[i] = rank;
    for (int i = 0; i < Nlocal; i++)
        dataRecv[i] = 0;

    int dSize = sizeof(double);

    int dCount = 1024;

    uint8_t *buf = new uint8_t[1024 * 1024];
    MPI_Buffer_attach(buf, 1024 * 1024 * 16);

    if (rank == 0)
    {
        MPI_Datatype typeHere;
        int b[1];
        MPI_Aint d[1];
        b[0] = dCount * dSize;
        d[0] = 0;
        MPI_Type_create_hindexed(1, b, d, MPI_UINT8_T, &typeHere);
        MPI_Type_commit(&typeHere);
        MPI_Request req, req1, req2, req3;

        MPI_Recv_init(data, 1, typeHere, 1, 123, MPI_COMM_WORLD, &req);
        MPI_Start(&req);
        MPI_Status stat;
        MPI_Wait(&req, &stat);

        MPI_Bsend_init(data, 1, typeHere, 0, 123, MPI_COMM_WORLD, &req1);
        MPI_Bsend_init(data, 1, typeHere, 1, 123, MPI_COMM_WORLD, &req3);
        MPI_Recv_init(dataRecv, 1, typeHere, 0, 123, MPI_COMM_WORLD, &req2);
        MPI_Start(&req1);
        MPI_Start(&req2);
        MPI_Start(&req3);
        MPI_Wait(&req1, &stat);
        MPI_Wait(&req2, &stat);
        MPI_Wait(&req3, &stat);

        MPI_Request_free(&req);
        MPI_Request_free(&req1);
        MPI_Request_free(&req2);
        MPI_Request_free(&req3);
        MPI_Type_free(&typeHere);
    }
    else if (rank == 1)
    {
        MPI_Datatype typeHere;
        int *b = new int[dCount];
        MPI_Aint *d = new MPI_Aint[dCount];
        for (int i = 0; i < dCount; i++)
            b[i] = dSize, d[i] = dSize * i * 2;
        MPI_Type_create_hindexed(dCount, b, d, MPI_UINT8_T, &typeHere);
        MPI_Type_commit(&typeHere);
        MPI_Request req, req3;
        MPI_Bsend_init(data, 1, typeHere, 0, 123, MPI_COMM_WORLD, &req);
        MPI_Start(&req);
        MPI_Status stat;
        MPI_Wait(&req, &stat);

        MPI_Recv_init(dataRecv, 1, typeHere, 0, 123, MPI_COMM_WORLD, &req3);
        MPI_Start(&req3);
        MPI_Wait(&req3, &stat);

        MPI_Request_free(&req);
        MPI_Request_free(&req3);
        MPI_Type_free(&typeHere);

        delete[] d;
        delete[] b;
    }

    std::cout << "rank " << rank << "Done " << std::endl;

    std::cout << std::setprecision(14);
    if (rank == 0)
    {
        for (int i = 0; i < dCount * 2; i++)
            std::cout << rank << " data " << i << ": " << data[i] << std::endl;
        for (int i = 0; i < dCount * 2; i++)
            std::cout << rank << " dataRecv " << i << ": " << dataRecv[i] << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1)
    {
        for (int i = 0; i < dCount * 2; i++)
            std::cout << rank << " data " << i << ": " << data[i] << std::endl;
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