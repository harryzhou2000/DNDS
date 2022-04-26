#pragma once
#include "DNDS_Mesh.hpp"
#include "DNDS_FV_VR.hpp"
#include "DNDS_FV_CR.hpp"
#include "DNDS_ODE.hpp"
#include "DNDS_Scripting.hpp"

namespace DNDS
{
    class EikonalEvaluator
    {
    public:
        int kAv = 4; //! change when order is changed!!
        CompactFacedMeshSerialRW *mesh;
        ImplicitFiniteVolume2D *fv;
        VRFiniteVolume2D *vfv;
        CRFiniteVolume2D *cfv;

        std::vector<real> lambdaCell;
        // std::vector<real> lambdaFace;

        EikonalEvaluator(CompactFacedMeshSerialRW *Nmesh, ImplicitFiniteVolume2D *Nfv, VRFiniteVolume2D *Nvfv, CRFiniteVolume2D *Ncfv)
            : mesh(Nmesh), fv(Nfv), vfv(Nvfv), cfv(Ncfv)
        {
            lambdaCell.resize(mesh->cell2nodeLocal.size()); // but only dist part are used, ghost part to not judge for it in facial iter
            // lambdaFace.resize(mesh->face2nodeLocal.size());
        }

        void EvaluateDt(std::vector<real> &dt, ArrayCascadeLocal<SemiVarMatrix<1>> &uRec, real CFL, real MaxDt = 1, bool UseLocaldt = false)
        {
            for (auto &i : lambdaCell)
                i = 0.0;
            for (index iFace = 0; iFace < mesh->face2nodeLocal.size(); iFace++)
            {
                auto f2c = mesh->face2cellLocal[iFace];
                Elem::tPoint unitNorm = vfv->faceNormCenter[iFace].normalized();

                index iCellL = f2c[0];
                Elem::tPoint gradL{0, 0, 0}, gradR{0, 0, 0};
                gradL({0, 1}) = vfv->faceDiBjCenterCache[iFace].first({1, 2}, Eigen::all).rightCols(vfv->faceDiBjCenterCache[iFace].first.cols() - 1) * uRec[iCellL].m();
                if (f2c[1] != FACE_2_VOL_EMPTY) // can't be non local
                    gradR({0, 1}) = vfv->faceDiBjCenterCache[iFace].second({1, 2}, Eigen::all).rightCols(vfv->faceDiBjCenterCache[iFace].second.cols() - 1) * uRec[f2c[1]].m();
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
        void EvaluateRHS(ArrayDOF<1u> &rhs, ArrayDOF<1u> &u, ArrayCascadeLocal<SemiVarMatrix<1u>> uRec, ArrayCascadeLocal<SemiVarMatrix<1u>> uRecCR)
        {
            for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
                real rhsValue = 0.;
                auto &cellRecAtr = vfv->cellRecAtrLocal[iCell][0];
                auto &cellAtr = mesh->cellAtrLocal[iCell][0];
                Elem::ElementManager eCell(cellAtr.type, cellRecAtr.intScheme);
                eCell.Integration(
                    rhsValue,
                    [&](real &inc, int ig, Elem::tPoint p, Elem::tDiFj &DiNj)
                    {
                        Elem::tPoint vrGrad{0, 0, 0}, crGrad{0, 0, 0};
                        vrGrad({0, 1}) = vfv->cellDiBjGaussCache[iCell][ig]({1, 2}, Eigen::all).rightCols(uRec[iCell].m().rows()) * uRec[iCell].m();
                        crGrad({0, 1}) = cfv->cellDiBjGaussCache[iCell][ig]({1, 2}, Eigen::all).rightCols(uRecCR[iCell].m().rows()) * uRecCR[iCell].m();
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
                        uRecValL = vfv->faceDiBjGaussCache[iFace][ig * 2 + 0]({0, 1, 2}, Eigen::all).rightCols(uRec[f2c[0]].m().rows()) * uRec[f2c[0]].m();
                        if (f2c[1] != FACE_2_VOL_EMPTY)
                        {
                            uRecValR = vfv->faceDiBjGaussCache[iFace][ig * 2 + 1]({0, 1, 2}, Eigen::all).rightCols(uRec[f2c[1]].m().rows()) * uRec[f2c[1]].m();
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
                    resc += std::max(std::fabs(rhs[iCell](0)), res), P;
                MPI_Allreduce(&resc, &res, 1, DNDS_MPI_REAL, MPI_MAX, rhs.dist->getMPI().comm);
            }
        }
    };

    class EikonalCRSolver
    {
        MPIInfo mpi;
        std::shared_ptr<CompactFacedMeshSerialRW> mesh;
        std::shared_ptr<ImplicitFiniteVolume2D> fv;
        std::shared_ptr<VRFiniteVolume2D> vfv;
        std::shared_ptr<CRFiniteVolume2D> cfv;

        ArrayDOF<1u> u;
        ArrayCascadeLocal<SemiVarMatrix<1u>> uRec, uRecNew, uRecCR;

        std::shared_ptr<ArrayCascade<VecStaticBatch<3>>> outDist;
        std::shared_ptr<ArrayCascade<VecStaticBatch<3>>> outSerial;

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
        }

        void ReadMeshAndInitialize()
        {
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

                if (step % config.nConsoleCheck == 0)
                {
                    double telapsed = MPI_Wtime() - tstart;
                    if (mpi.rank == 0)
                        log() << "=== Step [" << step << "]   "
                              << "res [" << res << "]   Time [" << telapsed << "]   recTime[" << trec << "]   reccrTime[" << treccr
                              << "]   rhsTime[" << trhs << "]   commTime[" << tcomm << "]  " << std::endl;
                    tstart = MPI_Wtime();
                    trec = tcomm = treccr = trhs = 0.;
                }
                if (step % config.nDataOut == 0)
                {
                    PrintData(config.outPltName + std::to_string(step) + ".plt", ode);
                }
            }

            // u.WaitPersistentPullClean();
        }

        template <typename tODE>
        void PrintData(const std::string &fname, tODE &ode)
        {
            for (int iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
                Eigen::MatrixXd recval = vfv->cellDiBjCenterCache[iCell]({0, 1, 2}, Eigen::all).rightCols(uRec[iCell].m().rows()) * uRec[iCell].m();
                (*outDist)[iCell][0] = recval(0) + u[iCell](0);
                (*outDist)[iCell][1] = recval(1);
                (*outDist)[iCell][2] = recval(2);
                // (*outDist)[iCell][2] = ;
            }
            outSerial->startPersistentPull();
            outSerial->waitPersistentPull();
            const static std::vector<std::string> names{
                "sln", "dx", "dy"};
            mesh->PrintSerialPartPltASCIIDataArray(
                fname, 0, 3, //! oprank = 0
                [&](int idata)
                { return names[idata]; },
                [&](int idata, index iv)
                {
                    return (*outSerial)[iv][idata];
                });
        }
    };

}