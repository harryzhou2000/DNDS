#pragma once

#include <mpi.h>

namespace DNDS
{
    struct MPIInfo
    {
        MPI_Comm comm = MPI_COMM_NULL;
        int rank = -1;
        int size = -1;

        void setWorld()
        {
            comm = MPI_COMM_WORLD;
            int ierr;
            ierr = MPI_Comm_rank(comm, &rank);
            ierr = MPI_Comm_size(comm, &size);
        }
    };

}