#include "../DNDS_FV_EulerSolver.hpp"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    using namespace DNDS;
    MPIInfo mpi;
    mpi.setWorld();

    {
        EulerSolver<NS> solver(mpi);
        solver.ConfigureFromJson("data/staticRecEuler_config.json");
        solver.ReadMeshAndInitialize();
        // solver.RunExplicitSSPRK4();
        // solver.RunImplicitEuler();
        solver.RunStaticReconstruction();
    }
    MPI_Finalize();
    return 0;
}