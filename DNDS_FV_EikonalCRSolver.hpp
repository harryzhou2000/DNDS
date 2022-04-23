#pragma once
#include "DNDS_Mesh.hpp"
#include "DNDS_FV_VR.hpp"
#include "DNDS_FV_CR.hpp"
#include "DNDS_ODE.hpp"

namespace DNDS
{
    class EikonalEvaluator
    {
    public:
        int kAv = 4;
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
                    gradR({0, 1}) = vfv->faceDiBjCenterCache[iFace].second({1, 2}, Eigen::all).rightCols(vfv->faceDiBjCenterCache[iFace].first.cols() - 1) * uRec[f2c[1]].m();
                Elem::tPoint grad = (gradL + gradR) * 0.5;
                real lamFace = std::abs(grad.dot(unitNorm)) * fv->faceArea[iFace];
                lambdaCell[iCellL] += lamFace;
                if (f2c[1] != FACE_2_VOL_EMPTY) // can't be non local
                    lambdaCell[f2c[1]] += lamFace;
            }
            real dtMin = veryLargeReal;
            for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
                dt[iCell] = std::min(CFL * fv->volumeLocal[iCell] / (lambdaCell[iCell] + 1e-10), MaxDt);
                dtMin = std::min(dtMin, dt[iCell]);
            }
            if (!UseLocaldt)
            {
                for (auto &i : dt)
                    i = dtMin;
            }
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
                        inc = 1 - std::abs(vrGrad.dot(crGrad));
                        inc *= vfv->cellGaussJacobiDets[iCell][ig];
                    });
                rhs[iCell](0) = rhsValue / fv->volumeLocal[iCell];
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
                    dist = (vfv->faceCenters[iFace] - vfv->cellCenters[f2c[0]]).norm() * 2;
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

                        finc = dist * visEta * visGam * unitNorm.dot(gradC);
                        finc *= vfv->faceNorms[iFace][ig].norm(); // don't forget this
                    });
                rhs[f2c[0]](0) += flux / fv->volumeLocal[f2c[0]];
                if (f2c[1] != FACE_2_VOL_EMPTY)
                    rhs[f2c[1]](0) -= flux / fv->volumeLocal[f2c[0]];
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

        void ReadMeshAndInitialize(const std::string &mName)
        {
            CompactFacedMeshSerialRWBuild(mpi, mName, "data/out/debugmeshSO.plt", mesh);
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
            real CFL = 0.1;
            uRec.InitPersistentPullClean();
            u.InitPersistentPullClean();
            uRec.StartPersistentPullClean();
            u.StartPersistentPullClean();

            for (int step = 1; step <= 10; step++)
            {
                ode.Step(
                    u,
                    [&](ArrayDOF<1u> &crhs, ArrayDOF<1u> &cx)
                    {
                        uRec.WaitPersistentPullClean();
                        u.WaitPersistentPullClean();
                        vfv->ReconstructionJacobiStep(cx, uRec, uRecNew);
                        cfv->Reconstruction(cx, uRec, uRecCR);
                        uRec.StartPersistentPullClean();
                        u.StartPersistentPullClean();

                        eval.EvaluateRHS(crhs, cx, uRec, uRecCR);
                    },
                    [&](std::vector<real> &dt)
                    {
                        eval.EvaluateDt(dt, uRec, CFL, 0.1);
                    });
                real res;
                eval.EvaluateResidual(res, ode.rhsbuf[0]);
                if (mpi.rank == 0)
                    log() << "=== Step [" << step << "]   "
                          << "res [" << res << "]" << std::endl;
                PrintData("data/out/debugData_" + std::to_string(step) +".plt");
            }
            

            uRec.WaitPersistentPullClean();
            u.WaitPersistentPullClean();
        }

        void PrintData(const std::string &fname)
        {
            for (int iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
                Eigen::MatrixXd recval = vfv->cellDiBjCenterCache[iCell]({0, 1, 2}, Eigen::all).rightCols(uRec[iCell].m().rows()) * uRec[iCell].m();
                (*outDist)[iCell][0] = recval(0);
                (*outDist)[iCell][1] = recval(1);
                (*outDist)[iCell][1] = recval(2);
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