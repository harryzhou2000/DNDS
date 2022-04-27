#include "../DNDS_FV_VR.hpp"
#include "../DNDS_FV_CR.hpp"

int main(int argn, char *argv[])
{
    MPI_Init(&argn, &argv);
    using namespace DNDS;
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    // Debug::MPIDebugHold(mpi);

    auto fDiffs = [](Elem::tPoint p) -> Eigen::VectorXd
    {
        Eigen::VectorXd d(6);
        d(0) = (1 - std::cos(p(0) * 2 * pi)) * (1 - std::cos(p(1) * 2 * pi));
        d(1) = std::sin(p(0) * 2 * pi) * (1 - std::cos(p(1) * 2 * pi)) * 2 * pi;
        d(2) = std::sin(p(1) * 2 * pi) * (1 - std::cos(p(0) * 2 * pi)) * 2 * pi;
        d(3) = std::cos(p(0) * 2 * pi) * (1 - std::cos(p(1) * 2 * pi)) * 2 * pi * 2 * pi;
        d(4) = std::sin(p(0) * 2 * pi) * std::sin(p(1) * 2 * pi) * 2 * pi * 2 * pi;
        d(5) = std::cos(p(1) * 2 * pi) * (1 - std::cos(p(0) * 2 * pi)) * 2 * pi * 2 * pi;
        return d;
    };
    std::vector<std::string> meshNames = {
        "data/mesh/Uniform/UniformA0.msh",
        "data/mesh/Uniform/UniformA1.msh",
        "data/mesh/Uniform/UniformA2.msh",
        "data/mesh/Uniform/UniformA3.msh",
        "data/mesh/Uniform/UniformB0.msh",
        "data/mesh/Uniform/UniformB1.msh",
        "data/mesh/Uniform/UniformB2.msh",
        "data/mesh/Uniform/UniformB3.msh",
    };
    std::vector<Eigen::VectorXd> norms;
    norms.resize(meshNames.size() * 3);

    for (int iCase = 0; iCase < meshNames.size(); iCase++)
    {
        auto &mName = meshNames[iCase];
        std::shared_ptr<CompactFacedMeshSerialRW> mesh;
        CompactFacedMeshSerialRWBuild(mpi, mName, "data/out/debugmeshSO.plt", mesh);
        // InsertCheck(mpi, "AfterRead1");
        ImplicitFiniteVolume2D fv(mesh.get());
        VRFiniteVolume2D vfv(mesh.get(), &fv);

        vfv.initIntScheme();
        vfv.initMoment();
        vfv.initBaseDiffCache();
        vfv.initReconstructionMatVec();

        CRFiniteVolume2D cfv(vfv); //! mind the order!
        cfv.initReconstructionMatVec();

        ArrayLocal<VecStaticBatch<1u>> u;
        ArrayLocal<SemiVarMatrix<1u>> uRec, uRecNew, uRecCR;
        fv.BuildMean(u);
        vfv.BuildRec(uRec);
        uRecNew.Copy(uRec);
        cfv.BuildRec(uRecCR);

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
                        inc = fDiffs(pPhysics)(0);
                        inc *= Elem::DiNj2Jacobi(DiNj, coords)({0, 1}, {0, 1}).determinant();
                    });
                um /= fv.volumeLocal[iCell];
                e.p()(0) = um;
            });
        forEachInArray(
            *uRec.dist,
            [&](decltype(uRec.dist)::element_type::tComponent &e, DNDS::index iCell)
            {
                e.m().setZero();
            });

        u.InitPersistentPullClean();
        uRec.InitPersistentPullClean();
        // InsertCheck(mpi, "BeforeRec");
        double tstart = MPI_Wtime();
        for (int i = 0; i < 500; i++)
        {
            u.StartPersistentPullClean();
            uRec.StartPersistentPullClean();
            u.WaitPersistentPullClean();
            uRec.WaitPersistentPullClean();
            vfv.ReconstructionJacobiStep(u, uRec, uRecNew);
        }
        if (mpi.rank == 0)
            std::cout << "=== Rec time: " << MPI_Wtime() - tstart << std::endl;

        cfv.Reconstruction(u, uRec, uRecCR);
        Eigen::VectorXd norm1(6);
        Eigen::VectorXd norm2(6);
        Eigen::VectorXd normInf(6);
        norm1.setZero(), norm2.setZero(), normInf.setZero();
        // InsertCheck(mpi, "AfterRec");
        for (int iCell = 0; iCell < u.dist->size(); iCell++)
        {
            auto c2n = mesh->cell2nodeLocal[iCell];
            Eigen::MatrixXd coords;
            mesh->LoadCoords(c2n, coords);
            Elem::ElementManager eCell(mesh->cellAtrLocal[iCell][0].type, vfv.cellRecAtrLocal[iCell][0].intScheme);
            eCell.Integration(
                norm1,
                [&](Eigen::VectorXd &inc, int ng, Elem::tPoint &p, Elem::tDiFj &DiNj)
                {
                    Elem::tPoint pPhysics = coords * DiNj(0, Eigen::all).transpose();
                    int ndof = vfv.cellDiBjGaussCache[iCell][ng].cols();
                    int ndofCR = cfv.cellDiBjGaussCache[iCell][ng].cols();
                    Eigen::MatrixXd rec = vfv.cellDiBjGaussCache[iCell][ng].rightCols(ndof - 1) * uRec[iCell].m();
                    Eigen::MatrixXd recCR = cfv.cellDiBjGaussCache[iCell][ng].rightCols(ndofCR - 1) * uRecCR[iCell].m();
                    rec(0) += u[iCell].p()(0);
                    recCR(0) += u[iCell].p()(0);
                    auto vReal = fDiffs(pPhysics);
                    inc = (vReal - rec.topRows(6)).array().abs().matrix();
                    inc *= Elem::DiNj2Jacobi(DiNj, coords)({0, 1}, {0, 1}).determinant();
                });
            eCell.Integration(
                norm2,
                [&](Eigen::VectorXd &inc, int ng, Elem::tPoint &p, Elem::tDiFj &DiNj)
                {
                    Elem::tPoint pPhysics = coords * DiNj(0, Eigen::all).transpose();
                    int ndof = vfv.cellDiBjGaussCache[iCell][ng].cols();
                    int ndofCR = cfv.cellDiBjGaussCache[iCell][ng].cols();
                    Eigen::MatrixXd rec = vfv.cellDiBjGaussCache[iCell][ng].rightCols(ndof - 1) * uRec[iCell].m();
                    Eigen::MatrixXd recCR = cfv.cellDiBjGaussCache[iCell][ng].rightCols(ndofCR - 1) * uRecCR[iCell].m();
                    rec(0) += u[iCell].p()(0);
                    recCR(0) += u[iCell].p()(0);
                    auto vReal = fDiffs(pPhysics);
                    inc = (vReal - rec.topRows(6)).array().pow(2).matrix();
                    inc *= Elem::DiNj2Jacobi(DiNj, coords)({0, 1}, {0, 1}).determinant();
                });
        }
        // Eigen::VectorXd norm1R(6);
        // Eigen::VectorXd norm2R(6);
        // Eigen::VectorXd normInfR(6);
        // std::vector<double> norm1R;
        // std::vector<double> norm2R;
        // std::vector<double> normInfR;
        // norm1R.resize(6), norm2R.resize(6), normInfR.resize(6);
        norms[iCase * 3 + 0].resize(6);
        norms[iCase * 3 + 1].resize(6);
        norms[iCase * 3 + 2].resize(6);
        MPI_Reduce(norm1.data(), norms[iCase * 3 + 0].data(), 6, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
        MPI_Reduce(norm2.data(), norms[iCase * 3 + 1].data(), 6, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
        MPI_Reduce(normInf.data(), norms[iCase * 3 + 2].data(), 6, MPI_DOUBLE, MPI_MAX, 0, mpi.comm);

        std::move(*mesh);
        // if (mpi.rank == mpi.size - 1)
        // {
        //     std::cout << uRec[0].m() << std::endl;
        // }
    }

    for (int iCase = 0; iCase < meshNames.size(); iCase++)
    {
        auto &mName = meshNames[iCase];
        if (mpi.rank == 0)
        {
            std::cout << "=== === === === === === === === === === === === === === === === === === ===" << std::endl;
            std::cout << "Name: " << mName
                      //   << "  \nnorm1 " << norm1R[0] << norm1R[1] << norm1R[2]
                      //   << "  \nnorm2 " << norm2R[0] << norm2R[1] << norm2R[2]
                      //   << "  \nnormInf " << normInfR[0] << normInfR[1] << normInfR[2] << std::endl;
                      << "  \nnorm1 " << norms[iCase * 3 + 0].transpose()
                      << "  \nnorm2 " << norms[iCase * 3 + 1].transpose()
                      << "  \nnormInf " << norms[iCase * 3 + 2].transpose() << std::endl;
            std::cout << "=== === === === === === === === === === === === === === === === === === ===" << std::endl
                      << std::endl
                      << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}