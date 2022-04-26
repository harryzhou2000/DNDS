#include "../DNDS_FV_EikonalCRSolver.hpp"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    using namespace DNDS;
    MPIInfo mpi;
    mpi.setWorld();
    {
        EikonalCRSolver solver(mpi);
        solver.ConfigureFromJson("data/eikonal_config.json");
        // solver.ReadMeshAndInitialize("data/mesh/Uniform/UniformD100.msh");
        solver.ReadMeshAndInitialize();
        // solver.ReadMeshAndInitialize("data/mesh/NACA0012_WIDE_H3_Closed.msh");
        solver.RunExplicitSSPRK4();
    }
    MPI_Finalize();
    return 0;
}