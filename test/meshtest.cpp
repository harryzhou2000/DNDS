#include "../DNDS_Mesh.hpp"

int main()
{
    DNDS::SerialGmshReader2d gmshReader2D;
    gmshReader2D.FileRead("data/mesh/CylinderBM_1.msh");
    gmshReader2D.InterpolateTopology();
    gmshReader2D.WriteMeshDebugTecASCII("data/out/debugmesh.plt");
    return 0;
}