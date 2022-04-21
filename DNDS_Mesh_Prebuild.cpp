#include "DNDS_Mesh_Prebuild.h"

namespace DNDS
{
    void CompactFacedMeshSerialRWBuild(MPIInfo mpi, const std::string &gmshFile, const std::string &distDebugFile,
                                       CompactFacedMeshSerialRW **mesh)
    {
        DNDS::SerialGmshReader2d gmshReader2D;
        if (mpi.rank == 0)
        {
            gmshReader2D.FileRead(gmshFile);
            // gmshReader2D.FileRead("data/mesh/NACA0012_WIDE_H3_O2.msh");
            gmshReader2D.InterpolateTopology();
            // gmshReader2D.WriteMeshDebugTecASCII("data/out/debugmesh.plt");
        }
        *mesh = new CompactFacedMeshSerialRW(gmshReader2D, mpi);
        std::move(gmshReader2D);
        // mesh.LogStatusSerialPart();
        (*mesh)->MetisSerialPartitionKWay(0);
        // mesh.LogStatusDistPart();
        (*mesh)->ClearSerial();
        (*mesh)->BuildSerialOut(0);
        (*mesh)->PrintSerialPartPltASCIIDBG(distDebugFile, 0);
        // InsertCheck(mpi, "PB1");
        (*mesh)->BuildGhosts();
        // InsertCheck(mpi, "PBEND");
    }
}