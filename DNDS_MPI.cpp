#include "DNDS_MPI.hpp"

namespace DNDS
{
    namespace Debug
    {

#include <iostream>
#if defined(linux) || defined(_UNIX)
#include <sys/ptrace.h>
#include <unistd.h>
#include <sys/stat.h>
#endif
#if defined(_WIN32) || defined(__WINDOWS_)
#include <Windows.h>
#include <process.h>
#endif
        inline bool IsDebugged()
        {

#if defined(linux) || defined(_UNIX)
            std::ifstream fin("/proc/self/status"); // able to detect gdb
            std::string buf;
            int tpid = 0;
            while (!fin.eof())
            {
                fin >> buf;
                if (buf == "TracerPid:")
                {
                    fin >> tpid;
                    exit;
                }
            }
            fin.close();
            return tpid != 0;
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