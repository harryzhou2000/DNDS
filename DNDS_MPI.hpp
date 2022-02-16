#pragma once

#include <mpi.h>
#include <vector>


namespace DNDS
{

    typedef int MPI_int;
    typedef MPI_Aint MPI_index;

    typedef std::vector<MPI_int> tMPI_sizeVec;
    typedef std::vector<MPI_index> tMPI_indexVec;

    typedef std::vector<MPI_Status> tMPI_statVec;
    typedef std::vector<MPI_Request> tMPI_reqVec;
    

    
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