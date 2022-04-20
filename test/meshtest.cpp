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
    DNDS::CompactFacedMeshSerialRW mesh(std::move(gmshReader2D), mpi);
    // mesh.LogStatusSerialPart();
    mesh.MetisSerialPartitionKWay(0);

    // mesh.LogStatusDistPart();

    mesh.ClearSerial();
    mesh.BuildSerialOut(0);
    mesh.PrintSerialPartPltASCIIDBG("data/out/debugmeshSO.plt", 0);
    mesh.BuildGhosts();

    using namespace DNDS;
    ArrayCascade<Real1> rk(Real1::Context(mesh.cell2faceDist->size()), mpi);
    forEachInArray(rk, [&](Real1 &e, DNDS::index i)
                   { e[0] = mpi.rank; });
    ArrayCascade<Real1> rkGhost(&rk);
    rkGhost.BorrowGGIndexing(*mesh.cell2faceDistGhost);
    rkGhost.createMPITypes();
    forEachInArray(rkGhost, [&](Real1 &e, DNDS::index i)
                   { e[0] = mpi.rank; }); 
    rkGhost.pushOnce();//so that difference between rk and rank shows the boundary cells of domains
    ArrayCascade<Real1> rkSerial(&rk);
    rkSerial.BorrowGGIndexing(*mesh.cell2node);
    rkSerial.createMPITypes();
    rkSerial.pullOnce();
    mesh.PrintSerialPartPltASCIIDataArray(
        "data/out/debugmeshSODist.plt", 0,
        1,
        [&](int i)
        { return std::string("RKCell"); },
        [&](int i, DNDS::index iv) -> real
        {
            return rkSerial[iv][0];
        });

    // std::cout << "\n";
}