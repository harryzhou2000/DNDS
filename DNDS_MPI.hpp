#pragma once

#include <mpi.h>
#include <vector>
#include "DNDS_Defines.h"

namespace DNDS
{

    typedef int MPI_int;
    typedef MPI_Aint MPI_index;

    typedef std::vector<MPI_int> tMPI_sizeVec;
    typedef tMPI_sizeVec tMPI_intVec;
    typedef std::vector<MPI_index> tMPI_indexVec;
    typedef tMPI_indexVec tMPI_AintVec;

    typedef std::vector<MPI_Status> tMPI_statVec;
    typedef std::vector<MPI_Request> tMPI_reqVec;

    /**
     * \brief maps index or other DNDS types to MPI_Datatype ids
     */
    template <class Tbasic>
    constexpr MPI_Datatype __DNDSToMPIType()
    {
        static_assert(sizeof(Tbasic) == 8 || sizeof(Tbasic) == 4, "DNDS::Tbasic is not right size");
        return sizeof(Tbasic) == 8 ? MPI_INT64_T : (sizeof(Tbasic) == 4 ? MPI_INT32_T : MPI_DATATYPE_NULL);
    }

    const MPI_Datatype DNDS_MPI_INDEX = __DNDSToMPIType<index>();

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

        bool operator==(const MPIInfo &r) const
        {
            return (comm == r.comm) && (rank == r.rank) && (size == r.size);
        }
    };

    inline void InsertCheck(const MPIInfo &mpi, const std::string &info = "")
    {
#ifndef NDEBUG
        MPI_Barrier(mpi.comm);
        std::cout << "=== CHECK \"" << info << "\"  RANK " << mpi.rank << " ==="
                  << " @  FName: " << __FUNCTION__
                  << " @  Place: " << __FILE__ << ":" << __LINE__ << std::endl;
        MPI_Barrier(mpi.comm);
#endif
    }

    template <class F>
    inline void MPISerialDo(const MPIInfo &mpi, F f)
    {
        for (MPI_int i = 0; i < mpi.size; i++)
        {
            MPI_Barrier(mpi.comm);
            if (mpi.rank == i)
                f();
        }
    }

    typedef std::vector<std::pair<MPI_int, MPI_Datatype>> tMPI_typePairVec;
    /**
     * \brief wrapper of tMPI_typePairVec
     */
    struct MPITypePairHolder : public tMPI_typePairVec
    {
        using tMPI_typePairVec::tMPI_typePairVec;
        ~MPITypePairHolder()
        {
            for (auto &i : (*this))
                if (i.first >= 0 && i.second != 0 && i.second != MPI_DATATYPE_NULL)
                    MPI_Type_free(&i.second); //, std::cout << "Free Type" << std::endl;
        }
    };

    typedef std::shared_ptr<MPITypePairHolder> tpMPITypePairHolder;
    /**
     * \brief wrapper of tMPI_reqVec
     */
    struct MPIReqHolder : public tMPI_reqVec
    {
        using tMPI_reqVec::tMPI_reqVec;
        ~MPIReqHolder()
        {
            for (auto &i : (*this))
                if (i != MPI_REQUEST_NULL)
                    MPI_Request_free(&i); //, std::cout << "Free Req" << std::endl;
        }
        void clear()
        {
            for (auto &i : (*this))
                if (i != MPI_REQUEST_NULL)
                    MPI_Request_free(&i); //, std::cout << "Free Req" << std::endl;
            tMPI_reqVec::clear();
        }
    };

}

namespace DNDS
{
    namespace Debug
    {

#include <iostream>
#if defined(linux) || defined(_UNIX)
#include <sys/ptrace.h>
#include <unistd.h>
#endif
#if defined(_WIN32) || defined(__WINDOWS_)
#include <Windows.h>
#include <process.h>
#endif
        inline bool IsDebugged()
        {

#if defined(linux) || defined(_UNIX)
            return false;
#endif
#if defined(_WIN32) || defined(__WINDOWS_)
            return IsDebuggerPresent();
#endif
        }

        inline void MPIDebugHold(const MPIInfo &mpi)
        {
#if defined(linux) || defined(_UNIX)
            MPISerialDo(mpi, [&]
                        { log() << "Rank " << mpi.rank << " PID: " << getpid() << std::endl; });
#endif
#if defined(_WIN32) || defined(__WINDOWS_)
            MPISerialDo(mpi, [&]
                        { log() << "Rank " << mpi.rank << " PID: " << _getpid() << std::endl; });
#endif
            int holdFlag = 1;
            while (holdFlag)
            {
                for (MPI_int ir = 0; ir < mpi.size; ir++)
                {
                    int newDebugFlag;
                    if (mpi.rank == ir)
                    {
                        newDebugFlag = int(IsDebugged());
                        MPI_Bcast(&newDebugFlag, 1, MPI_INT, ir, mpi.comm);
                    }
                    else
                        MPI_Bcast(&newDebugFlag, 1, MPI_INT, ir, mpi.comm);

                    // std::cout << "DBG " << newDebugFlag;
                    if (newDebugFlag)
                        holdFlag = 0;
                }
            }
        }
    }
}