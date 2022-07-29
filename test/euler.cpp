#include "../DNDS_FV_EulerSolver.hpp"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    using namespace DNDS;
    MPIInfo mpi;
    mpi.setWorld();

    {
        EulerSolver solver(mpi);
        solver.ConfigureFromJson("data/eikonal_config.json");
        solver.ReadMeshAndInitialize();
        solver.RunExplicitSSPRK4();
    }
    MPI_Finalize();
    return 0;
}