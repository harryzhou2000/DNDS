#include "../DNDS_FV_EikonalCRSolver.hpp"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    using namespace DNDS;
    MPIInfo mpi;
    mpi.setWorld();
    {
        EikonalCRSolver solver(mpi);
        solver.ReadMeshAndInitialize("data/mesh/Uniform/UniformA3.msh");
        solver.RunExplicitSSPRK4();
    }
    MPI_Finalize();
    return 0;
}