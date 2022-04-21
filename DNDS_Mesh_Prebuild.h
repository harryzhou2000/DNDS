#pragma once
#include "DNDS_Mesh.hpp"

namespace DNDS
{
    void CompactFacedMeshSerialRWBuild(MPIInfo mpi, const std::string &gmshFile, const std::string &distDebugFile,
                                         CompactFacedMeshSerialRW **mesh);
}