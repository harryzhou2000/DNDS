#pragma once
#include "DNDS_Mesh.hpp"
#include "DNDS_FV_VR.hpp"
#include "DNDS_FV_CR.hpp"
#include "DNDS_ODE.hpp"
#include "DNDS_Scripting.hpp"
#include <iomanip>

namespace DNDS
{
    class EikonalEvaluator
    {

        int kAv = 4; //! change when order is changed!!
        CompactFacedMeshSerialRW *mesh = nullptr;
        ImplicitFiniteVolume2D *fv = nullptr;
        VRFiniteVolume2D *vfv = nullptr;
        CRFiniteVolume2D *cfv = nullptr;

    public:
        std::vector<real> lambdaCell;
        // std::vector<real> lambdaFace;

        EikonalEvaluator(CompactFacedMeshSerialRW *Nmesh, ImplicitFiniteVolume2D *Nfv, VRFiniteVolume2D *Nvfv, CRFiniteVolume2D *Ncfv)
            : kAv(vfv->P_ORDER + 1), mesh(Nmesh), fv(Nfv), vfv(Nvfv), cfv(Ncfv)
        {
            lambdaCell.resize(mesh->cell2nodeLocal.size()); // but only dist part are used, ghost part to not judge for it in facial iter
            // lambdaFace.resize(mesh->face2nodeLocal.size());
        }

        void EvaluateDt(std::vector<real> &dt, ArrayLocal<SemiVarMatrix<1>> &uRec, real CFL, real MaxDt = 1, bool UseLocaldt = false)
        {
            for (auto &i : lambdaCell)
                i = 0.0;
            for (index iFace = 0; iFace < mesh->face2nodeLocal.size(); iFace++)
            {
                auto f2c = mesh->face2cellLocal[iFace];
                auto faceDiBjCenterBatchElemVR = (*vfv->faceDiBjCenterBatch)[iFace];
                auto faceDiBjGaussBatchElemVR = (*vfv->faceDiBjGaussBatch)[iFace];
                Elem::tPoint unitNorm = vfv->faceNormCenter[iFace].normalized();

                index iCellL = f2c[0];
                Elem::tPoint gradL{0, 0, 0}, gradR{0, 0, 0};
                gradL({0, 1}) = faceDiBjCenterBatchElemVR.m(0)({1, 2}, Eigen::all).rightCols(faceDiBjCenterBatchElemVR.m(0).cols() - 1) * uRec[iCellL].m();
                if (f2c[1] != FACE_2_VOL_EMPTY) // can't be non local
                    gradR({0, 1}) = faceDiBjCenterBatchElemVR.m(1)({1, 2}, Eigen::all).rightCols(faceDiBjCenterBatchElemVR.m(1).cols() - 1) * uRec[f2c[1]].m();
                Elem::tPoint grad = (gradL + gradR) * 0.5;
                real lamFace = (std::abs(grad.dot(unitNorm)) + 1) * fv->faceArea[iFace];
                lambdaCell[iCellL] += lamFace;
                if (f2c[1] != FACE_2_VOL_EMPTY) // can't be non local
                    lambdaCell[f2c[1]] += lamFace;
            }
            real dtMin = veryLargeReal;
            for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
                // std::cout << fv->volumeLocal[iCell] << " " << (lambdaCell[iCell]) << " " << CFL << std::endl;
                // exit(0);
                dt[iCell] = std::min(CFL * fv->volumeLocal[iCell] / (lambdaCell[iCell] + 1e-10), MaxDt);
                dtMin = std::min(dtMin, dt[iCell]);
            }
            real dtMinall;
            MPI_Allreduce(&dtMin, &dtMinall, 1, DNDS_MPI_REAL, MPI_MIN, uRec.dist->getMPI().comm);
            if (!UseLocaldt)
            {
                for (auto &i : dt)
                    i = dtMinall;
            }
            // if (uRec.dist->getMPI().rank == 0)
            // log() << "dt: " << dtMin << std::endl;
        }

        /**
         * @brief
         * \param rhs overwritten;
         *
         */
        void EvaluateRHS(ArrayDOF<1u> &rhs, ArrayDOF<1u> &u, ArrayLocal<SemiVarMatrix<1u>> &uRec, ArrayLocal<SemiVarMatrix<1u>> &uRecCR)
        {
            for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
                real rhsValue = 0.;
                auto &cellRecAtr = vfv->cellRecAtrLocal[iCell][0];
                auto &cellAtr = mesh->cellAtrLocal[iCell][0];
                Elem::ElementManager eCell(cellAtr.type, cellRecAtr.intScheme);
                auto cellDiBjGaussBatchElemVR = (*vfv->cellDiBjGaussBatch)[iCell];
                auto cellDiBjGaussBatchElemCR = (*cfv->cellDiBjGaussBatch)[iCell];
                eCell.Integration(
                    rhsValue,
                    [&](real &inc, int ig, Elem::tPoint p, Elem::tDiFj &DiNj)
                    {
                        Elem::tPoint vrGrad{0, 0, 0}, crGrad{0, 0, 0};
                        vrGrad({0, 1}) = cellDiBjGaussBatchElemVR.m(ig)({1, 2}, Eigen::all).rightCols(uRec[iCell].m().rows()) * uRec[iCell].m();
                        crGrad({0, 1}) = cellDiBjGaussBatchElemCR.m(ig)({1, 2}, Eigen::all).rightCols(uRecCR[iCell].m().rows()) * uRecCR[iCell].m();
                        inc = 1 - std::fabs(vrGrad.dot(crGrad));
                        inc *= vfv->cellGaussJacobiDets[iCell][ig];
                    });
                rhs[iCell](0) = rhsValue / fv->volumeLocal[iCell];
                // std::cout << rhs[iCell](0) <<std::endl;
                // exit(0);
            }

            for (index iFace = 0; iFace < mesh->face2nodeLocal.size(); iFace++)
            {
                auto &faceRecAtr = vfv->faceRecAtrLocal[iFace][0];
                auto &faceAtr = mesh->faceAtrLocal[iFace][0];
                auto f2c = mesh->face2cellLocal[iFace];
                Elem::ElementManager eFace(faceAtr.type, faceRecAtr.intScheme);
                real flux = 0.;
                real dist;
                auto faceDiBjGaussBatchElemVR = (*vfv->faceDiBjGaussBatch)[iFace];

                if (f2c[1] != FACE_2_VOL_EMPTY)
                    dist = (vfv->cellCenters[f2c[0]] - vfv->cellCenters[f2c[1]]).norm();
                else
                    dist = (vfv->faceCenters[iFace] - vfv->cellCenters[f2c[0]]).norm() * 0;
                eFace.Integration(
                    flux,
                    [&](real &finc, int ig, Elem::tPoint &p, Elem::tDiFj &DiNj)
                    {
                        Elem::tPoint unitNorm = vfv->faceNorms[iFace][ig].normalized();
                        Eigen::Vector3d uRecVal{0, 0, 0}, uRecValL{0, 0, 0}, uRecValR{0, 0, 0}; // 2-d specific, no gradient z!
                        uRecValL = faceDiBjGaussBatchElemVR.m(ig * 2 + 0)({0, 1, 2}, Eigen::all).rightCols(uRec[f2c[0]].m().rows()) * uRec[f2c[0]].m();
                        if (f2c[1] != FACE_2_VOL_EMPTY)
                        {
                            uRecValR = faceDiBjGaussBatchElemVR.m(ig * 2 + 1)({0, 1, 2}, Eigen::all).rightCols(uRec[f2c[1]].m().rows()) * uRec[f2c[1]].m();
                            uRecVal = (uRecValL + uRecValR) * 0.5;
                        }
                        Elem::tPoint gradC{uRecVal(1), uRecVal(2), 0}; // 2-d specific, no gradient z!

                        real visEta = std::min(2. / 5. / (kAv * kAv), std::abs(1 - gradC.squaredNorm()));
                        if (visEta < 1. / 5. / (kAv * kAv))
                            visEta = 0.;
                        real visGam = 0.5;
                        visEta *= 0.2;

                        finc = visEta * visGam * (dist * unitNorm.dot(gradC) + (uRecValR(0) - uRecValL(0)) * 0.5);
                        finc *= vfv->faceNorms[iFace][ig].norm(); // don't forget this
                    });
                rhs[f2c[0]](0) += flux / fv->volumeLocal[f2c[0]];
                if (f2c[1] != FACE_2_VOL_EMPTY)
                    rhs[f2c[1]](0) -= flux / fv->volumeLocal[f2c[1]];
            }
            for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
                rhs[iCell](0) = std::min(std::max(rhs[iCell](0), -1e10), 1e10);
            }
        }

        void EvaluateResidual(real &res, ArrayDOF<1u> &rhs, real P = 1.)
        {

            if (P < largeReal)
            {
                real resc = 0;
                for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
                    resc += std::pow(std::fabs(rhs[iCell](0)), P);
                MPI_Allreduce(&resc, &res, 1, DNDS_MPI_REAL, MPI_SUM, rhs.dist->getMPI().comm);
                res = std::pow(res, 1. / P);
            }
            else
            {
                real resc = 0;
                for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
                    resc += std::max(std::fabs(rhs[iCell](0)), res);
                MPI_Allreduce(&resc, &res, 1, DNDS_MPI_REAL, MPI_MAX, rhs.dist->getMPI().comm);
            }
        }
    };

    class EikonalErrorEvaluator
    {
        CompactFacedMeshSerialRW *mesh = nullptr;
        ImplicitFiniteVolume2D *fv = nullptr;
        VRFiniteVolume2D *vfv = nullptr;
        real MaxD = 0;

    public:
        std::vector<real> sResult;
        EikonalErrorEvaluator(CompactFacedMeshSerialRW *nMesh, ImplicitFiniteVolume2D *nFv, VRFiniteVolume2D *nVfv, real nMaxD)
            : mesh(nMesh), fv(nFv), vfv(nVfv), MaxD(nMaxD)
        {
            MPIInfo mpi = mesh->mpi;
            sResult.resize(mesh->cell2nodeLocal.dist->size());

            index nBCPoint = 0;
            for (index iFace = 0; iFace < mesh->face2nodeLocal.dist->size(); iFace++)
                if (mesh->faceAtrLocal[iFace][0].iPhy == BoundaryType::Wall)
                {
                    Elem::ElementManager eFace(mesh->faceAtrLocal[iFace][0].type, vfv->faceRecAtrLocal[iFace][0].intScheme);
                    nBCPoint += eFace.getNInt();
                }
            Array<VecStaticBatch<6>> BCPointDist(VecStaticBatch<6>::Context(nBCPoint), mpi);
            index iFill = 0;
            for (index iFace = 0; iFace < mesh->face2nodeLocal.dist->size(); iFace++)
                if (mesh->faceAtrLocal[iFace][0].iPhy == BoundaryType::Wall)
                {
                    Elem::ElementManager eFace(mesh->faceAtrLocal[iFace][0].type, vfv->faceRecAtrLocal[iFace][0].intScheme);
                    Eigen::MatrixXd coords;
                    mesh->LoadCoords(mesh->face2nodeLocal[iFace], coords);
                    for (int ig = 0; ig < eFace.getNInt(); ig++)
                    {
                        Elem::tPoint pp;
                        eFace.GetIntPoint(ig, pp);
                        Elem::tDiFj DiNj(1, eFace.getNNode());
                        eFace.GetDiNj(pp, DiNj);
                        Elem::tPoint pPhyG = coords * DiNj.transpose();
                        BCPointDist[iFill].p()({0, 1, 2}) = pPhyG;
                        BCPointDist[iFill].p()({3, 4, 5}) = vfv->faceNorms[iFace][ig].normalized();
                        iFill++;
                    }
                }
            Array<VecStaticBatch<6>> BCPointFull(&BCPointDist);
            BCPointFull.createGlobalMapping();
            std::vector<index> fullPull(BCPointFull.pLGlobalMapping->globalSize());
            for (index i = 0; i < fullPull.size(); i++)
                fullPull[i] = i;
            BCPointFull.createGhostMapping(fullPull);
            BCPointFull.createMPITypes();
            BCPointFull.pullOnce();

            for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
                Elem::tPoint &pC = vfv->cellCenters[iCell];
                index imin = -1;
                real distMin = veryLargeReal;
                for (index isearch = 0; isearch < BCPointFull.size(); isearch++)
                {
                    real cdist = (BCPointFull[isearch].p()({0, 1, 2}) - pC).norm();
                    if (cdist < distMin)
                    {
                        distMin = cdist;
                        imin = isearch;
                    }
                }
                sResult[iCell] = std::fabs((BCPointFull[imin].p()({0, 1, 2}) - pC).dot(BCPointFull[imin].p()({3, 4, 5})));
                // sResult[iCell] = distMin;
            }
        }

        void EvaluateError(real &err, ArrayDOF<1u> &u, ArrayLocal<SemiVarMatrix<1u>> &uRec)
        {
            auto mpi = uRec.dist->getMPI();
            real volD = 0.0;
            real errD = 0.0;
            for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
                if (sResult[iCell] < MaxD)
                {
                    Eigen::MatrixXd recval = vfv->cellDiBjCenterBatch->operator[](iCell).m(0)({0}, Eigen::all).rightCols(uRec[iCell].m().rows()) * uRec[iCell].m();
                    real diff = (recval(0) + u[iCell](0)) - sResult[iCell];
                    errD += std::fabs(diff) / sResult[iCell];
                    volD += 1.;
                }
            }
            real vol;
            MPI_Allreduce(&volD, &vol, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            MPI_Allreduce(&errD, &err, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            err /= vol;
        }
    };

    class EikonalCRSolver
    {
        MPIInfo mpi;
        std::shared_ptr<CompactFacedMeshSerialRW> mesh;
        std::shared_ptr<ImplicitFiniteVolume2D> fv;
        std::shared_ptr<VRFiniteVolume2D> vfv;
        std::shared_ptr<CRFiniteVolume2D> cfv;
        std::shared_ptr<EikonalErrorEvaluator> err;

        ArrayDOF<1u> u;
        ArrayLocal<SemiVarMatrix<1u>> uRec, uRecNew, uRecCR;

        static const int nOUTS = 4;
        std::shared_ptr<Array<VecStaticBatch<nOUTS>>> outDist;
        std::shared_ptr<Array<VecStaticBatch<nOUTS>>> outSerial;

    public:
        EikonalCRSolver(const MPIInfo &nmpi) : mpi(nmpi)
        {
        }

        struct Configuration
        {
            int nTimeStep = 1000;
            int nConsoleCheck = 10;
            int nDataOut = 50;
            real CFL = 0.5;
            std::string mName = "data/mesh/NACA0012_WIDE_H3.msh";
            std::string outPltName = "data/out/debugData_";
            real err_dMax = 0.1;

            real res_base = 0;
        } config;

        void ConfigureFromJson(const std::string &jsonName)
        {
            rapidjson::Document doc;
            JSON::ReadFile(jsonName, doc);

            config.nTimeStep = doc["nTimeStep"].GetInt();
            if (mpi.rank == 0)
                log() << "JSON: nTimeStep = " << config.nTimeStep << std::endl;

            config.nConsoleCheck = doc["nConsoleCheck"].GetInt();
            if (mpi.rank == 0)
                log() << "JSON: nConsoleCheck = " << config.nConsoleCheck << std::endl;

            config.nDataOut = doc["nDataOut"].GetInt();
            if (mpi.rank == 0)
                log() << "JSON: nDataOut = " << config.nDataOut << std::endl;

            config.CFL = doc["CFL"].GetDouble();
            if (mpi.rank == 0)
                log() << "JSON: CFL = " << config.CFL << std::endl;

            config.mName = doc["meshFile"].GetString();
            if (mpi.rank == 0)
                log() << "JSON: meshFile = " << config.mName << std::endl;

            config.outPltName = doc["outPltName"].GetString();
            if (mpi.rank == 0)
                log() << "JSON: outPltName = " << config.outPltName << std::endl;

            config.err_dMax = doc["err_dMax"].GetDouble();
            if (mpi.rank == 0)
                log() << "JSON: err_dMax = " << config.err_dMax << std::endl;

            config.res_base = doc["res_base"].GetDouble();
            if (mpi.rank == 0)
                log() << "JSON: res_base = " << config.res_base << std::endl;
        }

        void ReadMeshAndInitialize()
        {
            // Debug::MPIDebugHold(mpi);
            CompactFacedMeshSerialRWBuild(mpi, config.mName, "data/out/debugmeshSO.plt", mesh);
            fv = std::make_shared<ImplicitFiniteVolume2D>(mesh.get());
            vfv = std::make_shared<VRFiniteVolume2D>(mesh.get(), fv.get());
            vfv->Initialization();

            cfv = std::make_shared<CRFiniteVolume2D>(*vfv);
            cfv->Initialization();

            fv->BuildMean(u);
            vfv->BuildRec(uRec);
            uRecNew.Copy(uRec);
            cfv->BuildRec(uRecCR);

            u.setConstant(0.0);

            outDist = std::make_shared<decltype(outDist)::element_type>(
                decltype(outDist)::element_type::tContext(mesh->cell2faceLocal.dist->size()), mpi);
            outSerial = std::make_shared<decltype(outDist)::element_type>(outDist.get());
            outSerial->BorrowGGIndexing(*mesh->cell2node);
            outSerial->createMPITypes();
            outSerial->initPersistentPull();

            err = std::make_shared<EikonalErrorEvaluator>(mesh.get(), fv.get(), vfv.get(), config.err_dMax);
        }

        void RunExplicitSSPRK4()
        {

            ODE::ExplicitSSPRK4LocalDt<decltype(u)> ode(
                u.dist->size(),
                [&](decltype(u) &data)
                {
                    data.resize(u.dist->size(), u.dist->getMPI());
                    data.CreateGhostCopyComm(mesh->cell2faceLocal);
                });
            EikonalEvaluator eval(mesh.get(), fv.get(), vfv.get(), cfv.get());
            uRec.InitPersistentPullClean();
            u.InitPersistentPullClean();
            // u.StartPersistentPullClean();
            double tstart = MPI_Wtime();
            double trec{0}, treccr{0}, tcomm{0}, trhs{0};
            int stepCount = 0;
            real resBaseC = config.res_base;
            for (int step = 1; step <= config.nTimeStep; step++)
            {
                ode.Step(
                    u,
                    [&](ArrayDOF<1u> &crhs, ArrayDOF<1u> &cx)
                    {
                        double tstartC = MPI_Wtime();
                        u.StartPersistentPullClean();
                        tcomm += MPI_Wtime() - tstartC;
                        double tstartB = MPI_Wtime();
                        u.WaitPersistentPullClean();
                        tcomm += MPI_Wtime() - tstartB;

                        double tstartA = MPI_Wtime();
                        vfv->ReconstructionJacobiStep(cx, uRec, uRecNew);
                        trec += MPI_Wtime() - tstartA;

                        double tstartF = MPI_Wtime();
                        uRec.StartPersistentPullClean();
                        tcomm += MPI_Wtime() - tstartF;

                        double tstartG = MPI_Wtime();
                        uRec.WaitPersistentPullClean();
                        tcomm += MPI_Wtime() - tstartG;

                        double tstartD = MPI_Wtime();
                        cfv->Reconstruction(cx, uRec, uRecCR);
                        treccr += MPI_Wtime() - tstartD;

                        double tstartE = MPI_Wtime();
                        eval.EvaluateRHS(crhs, cx, uRec, uRecCR);
                        trhs += MPI_Wtime() - tstartE;
                    },
                    [&](std::vector<real> &dt)
                    {
                        double tstartC = MPI_Wtime(); //! this also need to update!
                        u.StartPersistentPullClean();
                        tcomm += MPI_Wtime() - tstartC;
                        double tstartB = MPI_Wtime();
                        u.WaitPersistentPullClean();
                        tcomm += MPI_Wtime() - tstartB;

                        double tstartF = MPI_Wtime();
                        uRec.StartPersistentPullClean();
                        tcomm += MPI_Wtime() - tstartF;
                        double tstartG = MPI_Wtime();
                        uRec.WaitPersistentPullClean();
                        tcomm += MPI_Wtime() - tstartG;

                        eval.EvaluateDt(dt, uRec, config.CFL, 1e100, true);
                    });
                real res;
                eval.EvaluateResidual(res, ode.rhsbuf[0]);
                if (stepCount == 0 && resBaseC == 0)
                    resBaseC = res;

                real error;
                err->EvaluateError(error, u, uRec);

                if (step % config.nConsoleCheck == 0)
                {
                    double telapsed = MPI_Wtime() - tstart;
                    if (mpi.rank == 0)
                    {
                        auto fmt = log().flags();
                        log() << std::setprecision(6) << std::scientific
                              << "=== Step [" << step << "]   "
                              << "res \033[91m[" << res / resBaseC << "]\033[39m   "
                              << "err \033[93m[" << error << "]\033[39m   "
                              << std::setprecision(3) << std::fixed
                              << "Time [" << telapsed << "]   recTime [" << trec << "]   reccrTime [" << treccr
                              << "]   rhsTime [" << trhs << "]   commTime [" << tcomm << "]  " << std::endl;
                        log().setf(fmt);
                    }
                    tstart = MPI_Wtime();
                    trec = tcomm = treccr = trhs = 0.;
                }
                if (step % config.nDataOut == 0)
                {
                    PrintData(config.outPltName + std::to_string(step) + ".plt", ode);
                }
                stepCount++;
            }

            // u.WaitPersistentPullClean();
        }

        template <typename tODE>
        void PrintData(const std::string &fname, tODE &ode)
        {
            for (int iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
                Eigen::MatrixXd recval = vfv->cellDiBjCenterBatch->operator[](iCell).m(0)({0, 1, 2}, Eigen::all).rightCols(uRec[iCell].m().rows()) * uRec[iCell].m();
                (*outDist)[iCell][0] = recval(0) + u[iCell](0);
                (*outDist)[iCell][1] = recval(1);
                (*outDist)[iCell][2] = recval(2);
                (*outDist)[iCell][3] = err->sResult[iCell];
            }
            outSerial->startPersistentPull();
            outSerial->waitPersistentPull();
            const static std::vector<std::string> names{
                "sln", "dx", "dy", "sR"};
            mesh->PrintSerialPartPltASCIIDataArray(
                fname, 0, nOUTS, //! oprank = 0
                [&](int idata)
                { return names[idata]; },
                [&](int idata, index iv)
                {
                    return (*outSerial)[iv][idata];
                });
        }
    };

}