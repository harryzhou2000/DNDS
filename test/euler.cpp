#include "../DNDS_FV_EulerSolver.hpp"

int main(int argc, char *argv[])
{
    std::cout << "Start" << std::endl;
    MPI_Init(&argc, &argv);
    using namespace DNDS;
    MPIInfo mpi;
    mpi.setWorld();
    std::cout << "Start" << std::endl;

    {
        EulerSolver<NS> solver(mpi);
        
        solver.ConfigureFromJson("data/euler_config.json");
        solver.ReadMeshAndInitialize();
        // solver.RunExplicitSSPRK4();
        solver.RunImplicitEuler();
    }
    MPI_Finalize();
    return 0;
}