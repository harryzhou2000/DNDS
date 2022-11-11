#include "../DNDS_Mesh.hpp"
#include "../DNDS_FV_VR.hpp"
#include "../DNDS_FV_CR.hpp"

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
    using namespace DNDS;
    DNDS::MPIInfo mpi;
    mpi.setWorld();

    // DNDS::SerialGmshReader2d gmshReader2D;
    // if (mpi.rank == 0)
    // {
    //     gmshReader2D.FileRead("data/mesh/in.msh");
    //     // gmshReader2D.FileRead("data/mesh/NACA0012_WIDE_H3_O2.msh");
    //     gmshReader2D.InterpolateTopology();
    //     gmshReader2D.WriteMeshDebugTecASCII("data/out/debugmesh.plt");
    // }
    // DNDS::CompactFacedMeshSerialRW mesh((gmshReader2D), mpi);
    // std::move(gmshReader2D);
    // // mesh.LogStatusSerialPart();
    // mesh.MetisSerialPartitionKWay(0);

    // // mesh.LogStatusDistPart();

    // mesh.ClearSerial();
    // mesh.BuildSerialOut(0);
    // mesh.PrintSerialPartPltASCIIDBG("data/out/debugmeshSO.plt", 0);
    // mesh.BuildGhosts();
    std::shared_ptr<CompactFacedMeshSerialRW> mesh;
    CompactFacedMeshSerialRWBuild(mpi, "data/mesh/in.msh", "data/out/debugmeshSO.plt", mesh);

    Array<Real1> rk(Real1::Context(mesh->cell2faceDist->size()), mpi);
    forEachInArray(rk, [&](Real1 &e, DNDS::index i)
                   { e[0] = mpi.rank; });
    Array<Real1> rkGhost(&rk);
    rkGhost.BorrowGGIndexing(*mesh->cell2faceDistGhost);
    rkGhost.createMPITypes();
    forEachInArray(rkGhost, [&](Real1 &e, DNDS::index i)
                   { e[0] = mpi.rank; });
    rkGhost.pushOnce(); // so that difference between rk and rank shows the boundary cells of domains
    Array<Real1> rkSerial(&rk);
    rkSerial.BorrowGGIndexing(*mesh->cell2node);
    rkSerial.createMPITypes();
    rkSerial.pullOnce();
    mesh->PrintSerialPartPltASCIIDataArray(
        "data/out/debugmeshSODist.plt", 0,
        1,
        [&](int i)
        { return std::string("RKCell"); },
        [&](int i, DNDS::index iv) -> real
        {
            return rkSerial[iv][0];
        });

    ImplicitFiniteVolume2D fv(mesh.get());
    VRFiniteVolume2D vfv(mesh.get(), &fv);

    vfv.initIntScheme();
    vfv.initMoment();
    vfv.initBaseDiffCache();
    vfv.initReconstructionMatVec();

    CRFiniteVolume2D cfv(vfv); //! mind the order!
    cfv.initReconstructionMatVec();

    ArrayLocal<VecStaticBatch<1u>> u;
    ArrayRecV uRec, uRecNew, uRecCR;
    fv.BuildMean(u);
    vfv.BuildRec(uRec, 1);
    // InsertCheck(mpi, "B1", __FUNCTION__, __FILE__, __LINE__);
    vfv.BuildRec(uRecNew, 1);
    // InsertCheck(mpi, "C0", __FUNCTION__, __FILE__, __LINE__);
    cfv.BuildRec(uRecCR, 1);
    // InsertCheck(mpi, "C1", __FUNCTION__, __FILE__, __LINE__);

    forEachInArrayPair(
        *u.pair,
        [&](decltype(u.dist)::element_type::tComponent &e, DNDS::index iCell)
        {
            auto c2n = mesh->cell2nodeLocal[iCell];
            Eigen::MatrixXd coords;
            mesh->LoadCoords(c2n, coords);
            Elem::ElementManager eCell(mesh->cellAtrLocal[iCell][0].type, vfv.cellRecAtrLocal[iCell][0].intScheme);
            double um = 0;
            eCell.Integration(
                um,
                [&](double &inc, int ng, Elem::tPoint &p, Elem::tDiFj &DiNj)
                {
                    Elem::tPoint pPhysics = coords * DiNj(0, Eigen::all).transpose();
                    // inc = pPhysics[1] * (1 - pPhysics[1]) * pPhysics[0] * (1 - pPhysics[0]) * 4
                    inc = pPhysics[1] * (1 - pPhysics[1]) * 10;
                    inc = pPhysics[1] * pPhysics[1] * 10;
                    inc *= Elem::DiNj2Jacobi(DiNj, coords)({0, 1}, {0, 1}).determinant();
                });
            um /= fv.volumeLocal[iCell];

            e.p()(0) = um;
            real ytest = vfv.cellCenters[iCell](1);
            // e.p()(0) = ytest * (1 - ytest) * 4;
            // e.p()(1) = ytest * 4;
            // std::cout << "UM" << vfv.cellCenters[iCell] << std::endl;
        });
    forEachInArray(
        *uRec.dist,
        [&](decltype(uRec.dist)::element_type::tComponent &e, DNDS::index iCell)
        {
            e.p().setZero();
        });

    u.InitPersistentPullClean();
    uRec.InitPersistentPullClean();

    for (int i = 0; i < 100; i++)
    {
        u.StartPersistentPullClean();
        uRec.StartPersistentPullClean();
        u.WaitPersistentPullClean();
        uRec.WaitPersistentPullClean();

        vfv.ReconstructionJacobiStep(u, uRec, uRecNew);
    }

    cfv.Reconstruction(u, uRec, uRecCR);
    for (int i = 0; i < u.dist->size(); i++)
    {
        std::cout << "REC: \n"
                  << uRecCR[i].transpose() << std::endl
                  << uRec[i].transpose() << std::endl;
    }

    // std::cout << "\n";
}