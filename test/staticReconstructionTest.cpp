#include "../DNDS_FV_VR.hpp"
#include "../DNDS_FV_CR.hpp"

#include <cstdlib>
#include <iomanip>

int main(int argn, char *argv[])
{
    MPI_Init(&argn, &argv);
    using namespace DNDS;
    DNDS::MPIInfo mpi;
    mpi.setWorld();

    int nIter;
    double R;
    int ifLimit;
    int ifLimitIn;
    double jump;
    if (argn != 6)
    {
        std::cout << "need 5 args: num_iter, Rot angle, ifLimit, iflimitIn, jump !!\n";
        std::abort();
    }

    nIter = atoi(argv[1]);
    R = atof(argv[2]);
    ifLimit = atoi(argv[3]);
    ifLimitIn = atoi(argv[4]);
    jump = atof(argv[5]);
    if (mpi.rank == 0)
        std::cout << "N iter = " << nIter << " R = " << R
                  << " ifLimit = " << ifLimit
                  << " ifLimitIn = " << ifLimitIn
                  << " jump = " << jump << std::endl;

    // Debug::MPIDebugHold(mpi);

    auto fDiffs = [&](Elem::tPoint p) -> Eigen::VectorXd
    {
        Eigen::VectorXd d(6);
        d(0) = (1 - std::cos(p(0) * 2 * pi)) * (1 - std::cos(p(1) * 2 * pi));
        d(1) = std::sin(p(0) * 2 * pi) * (1 - std::cos(p(1) * 2 * pi)) * 2 * pi;
        d(2) = std::sin(p(1) * 2 * pi) * (1 - std::cos(p(0) * 2 * pi)) * 2 * pi;
        d(3) = std::cos(p(0) * 2 * pi) * (1 - std::cos(p(1) * 2 * pi)) * 2 * pi * 2 * pi;
        d(4) = std::sin(p(0) * 2 * pi) * std::sin(p(1) * 2 * pi) * 2 * pi * 2 * pi;
        d(5) = std::cos(p(1) * 2 * pi) * (1 - std::cos(p(0) * 2 * pi)) * 2 * pi * 2 * pi;

        //* use below as discontinuity test
        d(0) += p(0) > 0.2 && p(0) < 0.8 && p(1) > 0.2 && p(1) < 0.8 ? jump : 0;
        d(1) += 0;
        d(2) += 0;
        d(3) += 0;
        d(4) += 0;
        d(5) += 0;

        return d;
    };
    std::vector<std::string> meshNames = {
        // "data/mesh/Uniform/UniformA0.msh",
        // "data/mesh/Uniform/UniformA1.msh",
        // "data/mesh/Uniform/UniformA2.msh",
        // "data/mesh/Uniform/UniformA3.msh",
        // "data/mesh/Uniform/UniformA4.msh",
        "data/mesh/Uniform/UniformB0.msh",
        "data/mesh/Uniform/UniformB1.msh",
        "data/mesh/Uniform/UniformB2.msh",
        "data/mesh/Uniform/UniformB3.msh",
    };
    // std::vector<std::string> meshNames = {
    //     "data/mesh/Uniform/UniformAR00_0.msh",
    //     "data/mesh/Uniform/UniformAR00_1.msh",
    //     "data/mesh/Uniform/UniformAR00_2.msh",
    //     "data/mesh/Uniform/UniformAR00_3.msh",
    // };
    std::vector<Eigen::VectorXd> norms;
    norms.resize(meshNames.size() * 3);

    for (int iCase = 0; iCase < meshNames.size(); iCase++)
    {
        auto &mName = meshNames[iCase];
        std::shared_ptr<CompactFacedMeshSerialRW> mesh;
        CompactFacedMeshSerialRWBuild(mpi, mName, "data/out/debugmeshSO.plt", mesh, R);
        // InsertCheck(mpi, "AfterRead1");
        ImplicitFiniteVolume2D fv(mesh.get());
        VRFiniteVolume2D vfv(mesh.get(), &fv);
        vfv.setting.normWBAP = true;
        vfv.setting.orthogonalizeBase = false;
        

        vfv.initIntScheme();
        vfv.initMoment();
        vfv.initBaseDiffCache();
        vfv.initReconstructionMatVec();

        CRFiniteVolume2D cfv(vfv); //! mind the order!
        cfv.initReconstructionMatVec();
        // InsertCheck(mpi, "SDF 1");
        // ArrayLocal<VecStaticBatch<1u>> u;
        ArrayDOFV u;
        ArrayRecV uRec, uRecNew, uRecNew1, uRecOld, uRecCR;
        fv.BuildMean(u,1);
        vfv.BuildRec(uRec, 1);
        vfv.BuildRec(uRecNew, 1);
        vfv.BuildRec(uRecNew1, 1);
        vfv.BuildRec(uRecOld, 1);
        cfv.BuildRec(uRecCR, 1);

        ArrayLocal<Batch<real, 1>> ifUseLimiter;
        vfv.BuildIfUseLimiter(ifUseLimiter);

        // Eigen::ArrayXXd U1{{0,1,-1}};
        // Eigen::ArrayXXd U2{{1,0,1}};
        // std::cout << U1 << std::endl;
        // std::cout << U2 << std::endl;
        // Eigen::ArrayXXd U3;
        // vfv.FWBAP_L2_Biway(U1, U2, U3);
        // std::cout << U3 << std::endl;
        // // std::abort();

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
                e.p().setZero();
            });

        u.InitPersistentPullClean();
        uRec.InitPersistentPullClean();
        // InsertCheck(mpi, "BeforeRec");
        double tstart = MPI_Wtime();
        double tLimiter = 0;

        for (int i = 0; i < nIter; i++)
        {
            for (DNDS::index i = 0; i < uRecOld.dist->size(); i++)
            {
                uRecOld[i] = uRec[i];
            }

            u.StartPersistentPullClean();
            uRec.StartPersistentPullClean();
            u.WaitPersistentPullClean();
            uRec.WaitPersistentPullClean();
            vfv.ReconstructionJacobiStep(u, uRec, uRecNew);

            if (ifLimitIn)
            {
                double tstartA = MPI_Wtime();
                // vfv.ReconstructionWBAPLimitFacial(
                //     u, uRec, uRec, uRecF1, uRecF2, ifUseLimiter,
                //     [&](const Eigen::MatrixXd &uL, const Eigen::MatrixXd &uR, const Elem::tPoint &n)
                //     {
                //         return Eigen::MatrixXd::Identity(1, 1);
                //     },
                //     [&](const Eigen::MatrixXd &uL, const Eigen::MatrixXd &uR, const Elem::tPoint &n)
                //     {
                //         return Eigen::MatrixXd::Identity(1, 1);
                //     });
                vfv.ReconstructionWBAPLimitFacialV2(
                    u, uRec, uRec, uRecNew1, ifUseLimiter, true,
                    [&](const Eigen::MatrixXd &uL, const Eigen::MatrixXd &uR, const Elem::tPoint &n)
                    {
                        return Eigen::MatrixXd::Identity(1, 1);
                    },
                    [&](const Eigen::MatrixXd &uL, const Eigen::MatrixXd &uR, const Elem::tPoint &n)
                    {
                        return Eigen::MatrixXd::Identity(1, 1);
                    });

                tLimiter += MPI_Wtime() - tstartA;
            }
            real inc2 = 0;
            for (DNDS::index i = 0; i < uRecOld.dist->size(); i++)
            {
                inc2 += (uRecOld[i] - uRec[i]).array().pow(2).sum();
            }
            real inc2All = 0;
            MPI_Allreduce(&inc2, &inc2All, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
            if (mpi.rank == 0)
            {
                std::cout << mName << " inc2all  =  " << inc2All << std::endl;
            }
        }
        if (ifLimit)
        {
            double tstartA = MPI_Wtime();
            // vfv.ReconstructionWBAPLimitFacial(
            //     u, uRec, uRec, uRecF1, uRecF2, ifUseLimiter,
            //     [&](const Eigen::MatrixXd &uL, const Eigen::MatrixXd &uR, const Elem::tPoint &n)
            //     {
            //         return Eigen::MatrixXd::Identity(1, 1);
            //     },
            //     [&](const Eigen::MatrixXd &uL, const Eigen::MatrixXd &uR, const Elem::tPoint &n)
            //     {
            //         return Eigen::MatrixXd::Identity(1, 1);
            //     });
            vfv.ReconstructionWBAPLimitFacialV2(
                u, uRec, uRec, uRecNew1, ifUseLimiter, true,
                [&](const Eigen::MatrixXd &uL, const Eigen::MatrixXd &uR, const Elem::tPoint &n)
                {
                    return Eigen::MatrixXd::Identity(1, 1);
                },
                [&](const Eigen::MatrixXd &uL, const Eigen::MatrixXd &uR, const Elem::tPoint &n)
                {
                    return Eigen::MatrixXd::Identity(1, 1);
                });
            tLimiter += MPI_Wtime() - tstartA;
        }

        if (mpi.rank == 0)
            std::cout << "=== Rec time: " << MPI_Wtime() - tstart << "  === within: limiter time: " << tLimiter << std::endl;

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
                    int ndof = vfv.cellDiBjGaussBatch->operator[](iCell).m(ng).cols();
                    int ndofCR = cfv.cellDiBjGaussBatch->operator[](iCell).m(ng).cols();
                    Eigen::MatrixXd rec = vfv.cellDiBjGaussBatch->operator[](iCell).m(ng).rightCols(ndof - 1) * uRec[iCell];
                    Eigen::MatrixXd recCR = cfv.cellDiBjGaussBatch->operator[](iCell).m(ng).rightCols(ndofCR - 1) * uRecCR[iCell];
                    rec(0) += u[iCell](0);
                    recCR(0) += u[iCell](0);
                    auto vReal = fDiffs(pPhysics);
                    inc = (vReal - rec.topRows(6)).array().abs().matrix();
                    inc *= Elem::DiNj2Jacobi(DiNj, coords)({0, 1}, {0, 1}).determinant(); //! not doing to acquire element-wise error
                    normInf = normInf.array().cwiseMax((vReal - rec.topRows(6)).array().abs());
                });
            eCell.Integration(
                norm2,
                [&](Eigen::VectorXd &inc, int ng, Elem::tPoint &p, Elem::tDiFj &DiNj)
                {
                    Elem::tPoint pPhysics = coords * DiNj(0, Eigen::all).transpose();
                    int ndof = vfv.cellDiBjGaussBatch->operator[](iCell).m(ng).cols();
                    int ndofCR = cfv.cellDiBjGaussBatch->operator[](iCell).m(ng).cols();
                    Eigen::MatrixXd rec = vfv.cellDiBjGaussBatch->operator[](iCell).m(ng).rightCols(ndof - 1) * uRec[iCell];
                    Eigen::MatrixXd recCR = cfv.cellDiBjGaussBatch->operator[](iCell).m(ng).rightCols(ndofCR - 1) * uRecCR[iCell];
                    rec(0) += u[iCell](0);
                    recCR(0) += u[iCell](0);
                    auto vReal = fDiffs(pPhysics);
                    inc = (vReal - rec.topRows(6)).array().pow(2).matrix();
                    inc *= Elem::DiNj2Jacobi(DiNj, coords)({0, 1}, {0, 1}).determinant(); //! not doing to acquire element-wise error
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
        //     std::cout << uRec[0] << std::endl;
        // }
    }

    for (int iCase = 0; iCase < meshNames.size(); iCase++)
    {
        auto &mName = meshNames[iCase];
        if (mpi.rank == 0)
        {
            std::cout << "=== === === === === === === === === === === === === === === === === === ===" << std::endl;
            std::cout << "Name: " << mName << std::scientific << std::setprecision(15)
                      //   << "  \nnorm1 " << norm1R[0] << norm1R[1] << norm1R[2]
                      //   << "  \nnorm2 " << norm2R[0] << norm2R[1] << norm2R[2]
                      //   << "  \nnormInf " << normInfR[0] << normInfR[1] << normInfR[2] << std::endl;
                      << "  \nnorm1 " << norms[iCase * 3 + 0].transpose()
                      << "  \nnorm2 " << norms[iCase * 3 + 1].transpose().array().sqrt()
                      << "  \nnormInf " << norms[iCase * 3 + 2].transpose() << std::endl;
            std::cout << "=== === === === === === === === === === === === === === === === === === ===" << std::endl
                      << std::endl
                      << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}