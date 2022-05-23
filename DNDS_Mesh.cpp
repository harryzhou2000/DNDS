#include "DNDS_Mesh.hpp"

namespace DNDS
{
    void CompactFacedMeshSerialRWBuild(MPIInfo mpi, const std::string &gmshFile, const std::string &distDebugFile,
                                       std::shared_ptr<CompactFacedMeshSerialRW> &mesh, real RotZ)
    {
        DNDS::SerialGmshReader2d gmshReader2D;
        if (mpi.rank == 0)
        {
            gmshReader2D.FileRead(gmshFile);
            if (RotZ != 0.0)
                gmshReader2D.RotateZ(RotZ);
            gmshReader2D.InterpolateTopology();
            // gmshReader2D.WriteMeshDebugTecASCII("data/out/debugmesh.plt");
        }
        mesh = std::make_shared<CompactFacedMeshSerialRW>(gmshReader2D, mpi);
        std::move(gmshReader2D);
        // mesh.LogStatusSerialPart();
        (mesh)->MetisSerialPartitionKWay(0);
        // mesh.LogStatusDistPart();
        (mesh)->ClearSerial();
        (mesh)->BuildSerialOut(0);
        (mesh)->PrintSerialPartPltASCIIDBG(distDebugFile, 0);
        // InsertCheck(mpi, "PB1");
        (mesh)->BuildGhosts();
        // InsertCheck(mpi, "PBEND");
    }
}