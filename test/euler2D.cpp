#include "../DNDS_FV_EulerSolver.hpp"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    using namespace DNDS;
    MPIInfo mpi;
    mpi.setWorld();

    {
        EulerSolver<NS_2D> solver(mpi);
        solver.ConfigureFromJson("data/euler2D_config.json");
        solver.ReadMeshAndInitialize();
        // solver.RunExplicitSSPRK4();
        solver.RunImplicitEuler();
    }
    MPI_Finalize();
    return 0;
}