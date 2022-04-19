#include "../DNDS_Mesh.hpp"

void testA();

int main(int argc, char *argv[])
{
    int ierr;
    ierr = MPI_Init(&argc, &argv);
    testA();
    ierr = MPI_Finalize();
    return 0;
}

void testA()
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();

    DNDS::SerialGmshReader2d gmshReader2D;
    if (mpi.rank == 0)
    {
        gmshReader2D.FileRead("data/mesh/CylinderBM_1.msh");
        gmshReader2D.InterpolateTopology();
        gmshReader2D.WriteMeshDebugTecASCII("data/out/debugmesh.plt");
    }
    DNDS::CompactFacedMeshSerialRW mesh(gmshReader2D, mpi);
    mesh.LogStatusSerialPart();

    std::cout << "\n";
}