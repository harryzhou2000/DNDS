#include "../DNDS_FV_EulerSolver.hpp"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    using namespace DNDS;
    MPIInfo mpi;
    mpi.setWorld();

    {
        EulerSolver solver(mpi, NS);
        solver.ConfigureFromJson("data/euler_config.json");
        solver.ReadMeshAndInitialize();
        // solver.RunExplicitSSPRK4();
        solver.RunImplicitEuler();
    }
    MPI_Finalize();
    return 0;
}