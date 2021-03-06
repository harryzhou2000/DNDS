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
    public:
        CompactFacedMeshSerialRW *mesh = nullptr;
        ImplicitFiniteVolume2D *fv = nullptr;
        VRFiniteVolume2D *vfv = nullptr;
        CRFiniteVolume2D *cfv = nullptr;
        int kAv = 0;

        std::vector<real> lambdaCell;
        // std::vector<real> lambdaFace;

        struct Setting
        {
            real visScale = 1;
            real visScaleIn = 1;
            real isiScale = 1;
            real isiScaleIn = 1;
            real isiCutDown = 0.5;
            real ekCutDown = 0.5;
        } settings;

        EikonalEvaluator(CompactFacedMeshSerialRW *Nmesh, ImplicitFiniteVolume2D *Nfv, VRFiniteVolume2D *Nvfv, CRFiniteVolume2D *Ncfv)
            : mesh(Nmesh), fv(Nfv), vfv(Nvfv), cfv(Ncfv), kAv(Nvfv->P_ORDER + 1)
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
                dt[iCell] = std::min(CFL * fv->volumeLocal[iCell] / (lambdaCell[iCell] + 0.0000000001), MaxDt);
                dtMin = std::min(dtMin, dt[iCell]);
            }
            real dtMinall;
            MPI_Allreduce(&dtMin, &dtMinall, 1, DNDS_MPI_REAL, MPI_MIN, uRec.dist->getMPI().comm);
            // if (uRec.dist->getMPI().rank == 0)
            //     std::cout << "dt min is " << dtMinall << std::endl;
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
                        // inc = 1 - std::fabs(crGrad.dot(crGrad));
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
                    dist = (vfv->faceCenters[iFace] - vfv->cellCenters[f2c[0]]).norm() * 1;
                eFace.Integration(
                    flux,
                    [&](real &finc, int ig, Elem::tPoint &p, Elem::tDiFj &DiNj)
                    {
                        int nDiff = vfv->faceWeights->operator[](iFace).size();
                        Elem::tPoint unitNorm = vfv->faceNorms[iFace][ig].normalized();
                        Eigen::VectorXd uRecVal(nDiff), uRecValL(nDiff), uRecValR(nDiff), uRecValJump(nDiff); // 2-d specific, no gradient z!
                        uRecVal.setZero(), uRecValJump.setZero();
                        uRecValL = faceDiBjGaussBatchElemVR.m(ig * 2 + 0).rightCols(uRec[f2c[0]].m().rows()) * uRec[f2c[0]].m();

                        if (f2c[1] != FACE_2_VOL_EMPTY)
                        {
                            uRecValR = faceDiBjGaussBatchElemVR.m(ig * 2 + 1).rightCols(uRec[f2c[1]].m().rows()) * uRec[f2c[1]].m();
                            uRecVal = (uRecValL + uRecValR) * 0.5;
                            uRecValJump = (uRecValL - uRecValR) * 0.5;
                        }

                        Eigen::MatrixXd IJI, ISI;
                        vfv->FFaceFunctional(iFace, ig, uRecVal, uRecVal, (*vfv->faceWeights)[iFace], ISI);
                        vfv->FFaceFunctional(iFace, ig, uRecValJump, uRecValJump, (*vfv->faceWeights)[iFace], IJI);
                        // std::cout << "===" << IJI << ISI << std::endl;

                        Elem::tPoint gradC{uRecVal(1), uRecVal(2), 0}; // 2-d specific, no gradient z!

                        real visEta = std::min(2. / 5. / (kAv * kAv) * settings.visScaleIn, std::abs(1 - gradC.squaredNorm()));
                        if (visEta < 2. / 5. / (kAv * kAv) * settings.ekCutDown * settings.visScaleIn)
                            visEta = 0.;
                        visEta *= settings.visScale;

                        real visEta2 = std::min(std::pow(IJI(0, 0) / (ISI(0, 0) + 1e-8), 0.75), 2. / 5. / (kAv * kAv) * settings.isiScaleIn);
                        if (visEta2 < 2. / 5. / (kAv * kAv) * settings.isiCutDown * settings.isiScaleIn)
                            visEta2 = 0.;
                        visEta2 *= settings.isiScale;

                        visEta += visEta2;

                        // visEta = settings.visScale * 2. / 5. / (kAv * kAv);

                        real visGam = 0.5;

                        finc = visEta * visGam * (dist * unitNorm.dot(gradC) + (uRecValR(0) - uRecValL(0)) * .0);
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

        void PoissonInit(ArrayDOF<1u> &u, ArrayDOF<1u> &unew, int nIter, real alpha = 0.1)
        {
            for (int step = 1; step <= nIter; step++)
            {
                real incMax = 0;
                u.StartPersistentPullClean();
                u.WaitPersistentPullClean();
                for (index iCell = 0; iCell < u.dist->size(); iCell++)
                {
                    auto &c2f = mesh->cell2faceLocal[iCell];

                    Elem::tPoint gradAll{0, 0, 0};
                    bool hasUpper = false;
                    auto &cLeft = vfv->cellCenters[iCell];
                    real aC = 0;
                    real aRHSU = 0;
                    real uC = u[iCell](0);
                    real uDMax = 0;
                    real dmin = veryLargeReal;
                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                    {
                        // this is a repeated code block START
                        index iFace = c2f[ic2f];
                        auto &f2c = (*mesh->face2cellPair)[iFace];
                        index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                        index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                        auto &faceAttribute = mesh->faceAtrLocal[iFace][0];
                        auto &cFace = vfv->faceCenters[iFace];
                        auto fN = vfv->faceNormCenter[iFace].normalized() * (iCellAtFace == 0 ? 1. : -1.);

                        if (iCellOther != FACE_2_VOL_EMPTY)
                        {
                            real D = std::abs((cLeft - vfv->cellCenters[iCellOther]).dot(fN));
                            dmin = std::min(D, dmin);
                            // std::cout << u.dist->size() << " " << u.ghost->size() << " "
                            //           << iCell << " " << iCellOther << " " << u[iCellOther](0) << " " << uC << std::endl;
                            // if (u[iCellOther](0) != 0)
                            // {
                            //     forEachInArrayPair(*u.pair, [&](decltype(u.dist)::element_type::tComponent &e, index iCell)
                            //                           { std::cout << "UPPair: " << e << std::endl; });
                            //     exit(0);
                            // }
                            // inc += (u[iCellOther](0) - uC) / D * fv->faceArea[iFace];
                            aRHSU += u[iCellOther](0) / D * fv->faceArea[iFace];
                            aC += 1 / D * fv->faceArea[iFace];
                            uDMax = std::max(uDMax, std::abs(u[iCellOther](0) - uC));
                        }
                        else
                        {
                            real D = (cLeft - cFace).norm() * 2;
                            dmin = std::min(D, dmin);
                            // std::cout << D << std::endl;
                            if (faceAttribute.iPhy == BoundaryType::Wall)
                            {
                                aRHSU += 0 / D * fv->faceArea[iFace];
                                aC += 1 / D * fv->faceArea[iFace];
                                uDMax = std::max(uDMax, std::abs(0 - uC));
                            }
                            else if (faceAttribute.iPhy == BoundaryType::Farfield)
                            {
                                // inc += (0 - uC) / D * fv->faceArea[iFace];
                                // uDMax = std::max(uDMax, std::abs(0 - uC));
                                aRHSU += uC / D * fv->faceArea[iFace];
                                aC += 1 / D * fv->faceArea[iFace];
                            }
                            else
                            {
                                assert(false);
                            }
                        }
                    }
                    // real vinc = ((inc) / fv->volumeLocal[iCell] - 1);
                    // unew[iCell](0) += std::min(alpha, uDMax / (std::abs(vinc) + 1e-10)) * vinc;
                    // unew[iCell](0) += dmin * dmin * alpha * vinc;
                    u[iCell](0) = (1 - alpha) * u[iCell](0) + alpha * (aRHSU + fv->volumeLocal[iCell] * 1) / aC;
                    // std::cout << " great" << unew[iCell](0) << " ";
                    incMax = std::max(std::abs(uC - u[iCell](0)), incMax);
                }
                // for (index iCell = 0; iCell < u.dist->size(); iCell++)
                //     u[iCell] = unew[iCell];

                real incMaxR;
                // std::cout << incMax << std::endl;
                MPI_Allreduce(&incMax, &incMaxR, 1, DNDS_MPI_REAL, MPI_MAX, u.ghost->getMPI().comm);

                if (u.ghost->getMPI().rank == 0 && step % 100 == 0)
                    log() << "Poisson Solve Step [" << step << "]: RHSMax = [" << incMaxR << "]" << std::endl;
            }

            u.StartPersistentPullClean();
            u.WaitPersistentPullClean();
            for (index iCell = 0; iCell < u.dist->size(); iCell++)
            {
                auto &c2f = mesh->cell2faceLocal[iCell];
                Elem::tPoint grad{0, 0, 0};
                real uC = u[iCell](0);

                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    // this is a repeated code block START
                    index iFace = c2f[ic2f];
                    auto &f2c = (*mesh->face2cellPair)[iFace];
                    index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                    index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                    auto &faceAttribute = mesh->faceAtrLocal[iFace][0];
                    auto &cFace = vfv->faceCenters[iFace];
                    auto &cLeft = vfv->cellCenters[iCell];
                    auto fN = vfv->faceNormCenter[iFace].normalized() * (iCellAtFace == 0 ? 1. : -1.);

                    if (iCellOther != FACE_2_VOL_EMPTY)
                    {
                        real D = std::abs((cLeft - vfv->cellCenters[iCellOther]).dot(fN));
                        grad += ((u[iCellOther](0) + uC) * 0.5 * fv->faceArea[iFace]) * fN;
                    }
                    else
                    {
                        real D = (cLeft - cFace).norm() * 2;
                        if (faceAttribute.iPhy == BoundaryType::Wall)
                        {
                            grad += ((0 + uC) * 0.5 * fv->faceArea[iFace]) * fN;
                        }
                        else if (faceAttribute.iPhy == BoundaryType::Farfield)
                        {
                            grad += ((uC + uC) * 0.5 * fv->faceArea[iFace]) * fN;
                        }
                        else
                        {
                            assert(false);
                        }
                    }
                }
                grad.array() /= fv->volumeLocal[iCell];
                real nGrad = grad.squaredNorm();
                unew[iCell](0) = std::sqrt(std::max(nGrad + 2 * u[iCell](0), 0.0)) - std::sqrt(nGrad);
                // unew[iCell] = u[iCell];
                // std::cout << "out " << u[iCell];
            }

            

            for (int i = 0; i < 10; i++)
            {
                unew.StartPersistentPullClean();
                unew.WaitPersistentPullClean();
                for (index iCell = 0; iCell < u.dist->size(); iCell++)
                {
                    auto &c2f = mesh->cell2faceLocal[iCell];
                    Elem::tPoint grad{0, 0, 0};
                    real uC = unew[iCell](0);

                    real uSum{0}, nSum{0};

                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                    {
                        // this is a repeated code block START
                        index iFace = c2f[ic2f];
                        auto &f2c = (*mesh->face2cellPair)[iFace];
                        index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                        index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                        auto &faceAttribute = mesh->faceAtrLocal[iFace][0];
                        auto &cFace = vfv->faceCenters[iFace];
                        auto &cLeft = vfv->cellCenters[iCell];
                        auto fN = vfv->faceNormCenter[iFace].normalized() * (iCellAtFace == 0 ? 1. : -1.);

                        if (iCellOther != FACE_2_VOL_EMPTY)
                        {
                            uSum += unew[iCellOther](0);
                            nSum += 1;

                            uSum += uC;
                            nSum += 1;
                        }
                        else
                        {
                            real D = (cLeft - cFace).norm() * 2;
                            if (faceAttribute.iPhy == BoundaryType::Wall)
                            {
                                uSum += 0;
                                nSum += 1;

                                uSum += uC;
                                nSum += 1;
                            }
                            else if (faceAttribute.iPhy == BoundaryType::Farfield)
                            {
                                uSum += uC;
                                nSum += 1;

                                uSum += uC;
                                nSum += 1;
                            }
                            else
                            {
                                assert(false);
                            }
                        }
                    }
                    u[iCell](0) = uSum / nSum;
                }
                for (index iCell = 0; iCell < u.dist->size(); iCell++)
                    unew[iCell] = u[iCell];
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
            const int NSampleLine = 6;

            index nBCPoint = 0;
            for (index iFace = 0; iFace < mesh->face2nodeLocal.dist->size(); iFace++)
                if (mesh->faceAtrLocal[iFace][0].iPhy == BoundaryType::Wall)
                {
                    Elem::ElementManager eFace(mesh->faceAtrLocal[iFace][0].type, vfv->faceRecAtrLocal[iFace][0].intScheme);
                    nBCPoint += NSampleLine; //! knowing as line //eFace.getNInt()
                }
            Array<VecStaticBatch<6>> BCPointDist(VecStaticBatch<6>::Context(nBCPoint), mpi);
            index iFill = 0;
            for (index iFace = 0; iFace < mesh->face2nodeLocal.dist->size(); iFace++)
                if (mesh->faceAtrLocal[iFace][0].iPhy == BoundaryType::Wall)
                {
                    Elem::ElementManager eFace(mesh->faceAtrLocal[iFace][0].type, vfv->faceRecAtrLocal[iFace][0].intScheme);
                    Eigen::MatrixXd coords;
                    mesh->LoadCoords(mesh->face2nodeLocal[iFace], coords);
                    for (int ig = 0; ig < NSampleLine; ig++) //! knowing as line //eFace.getNInt()
                    {
                        Elem::tPoint pp;
                        // eFace.GetIntPoint(ig, pp);
                        pp << -1.0 + ig * 2.0 / double(NSampleLine - 1), 0, 0;
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
            real minResult = veryLargeReal;
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
                //! assuming being a line and continuous
                if ((imin % NSampleLine) > 0 && (imin % NSampleLine) < NSampleLine - 1 && distMin < 300 * MaxD)
                {
                    Elem::tPoint p0 = BCPointFull[imin - 1].p()({0, 1, 2});
                    Elem::tPoint p1 = BCPointFull[imin].p()({0, 1, 2});
                    Elem::tPoint p2 = BCPointFull[imin + 1].p()({0, 1, 2});
                    //! using auto to receive here could cause segfault in -O3, GPU704
                    auto d = [&](real pPx) -> real
                    {
                        return ((pPx * (pPx - 1) * 0.5) * p0 + (1 - pPx * pPx) * p1 + (pPx * (pPx + 1) * 0.5) * p2 - pC).squaredNorm();
                    };
                    auto dd = [&](real pPx) -> real
                    {
                        return 2 * ((pPx * (pPx - 1) * 0.5) * p0 + (1 - pPx * pPx) * p1 + (pPx * (pPx + 1) * 0.5) * p2 - pC)
                                       .dot(((2 * pPx - 1) * 0.5) * p0 + (-2 * pPx) * p1 + ((2 * pPx + 1) * 0.5) * p2);
                    };

                    real pPxc = 0.0;

                    real dc = d(pPxc);
                    real ddxm = std::max(std::fabs(dd(-1)), std::fabs(dd(1)));
                    int i;
                    for (i = 0; i < 10000; i++)
                    {
                        real ddxc = dd(pPxc);
                        real inc = std::max(std::min(ddxc / ddxm, 1.), -1.) * 0.1;
                        pPxc -= inc;
                        // std::cout << d(pPxc) << std::endl;
                        if (std::abs(inc) < 1e-10)
                            break;
                    }

                    real de = d(pPxc);
                    // std::cout << imin << " " << i << " sDC " << std::sqrt(dc) << " sDE " << std::sqrt(de) << " NC "
                    //           << std::fabs((BCPointFull[imin].p()({0, 1, 2}) - pC).dot(BCPointFull[imin].p()({3, 4, 5}))) << std::endl;
                    // exit(0);
                    sResult[iCell] = std::sqrt(de);
                }
                else
                {
                    // sResult[iCell] = std::fabs((BCPointFull[imin].p()({0, 1, 2}) - pC).dot(BCPointFull[imin].p()({3, 4, 5})));
                    sResult[iCell] = distMin;
                }
                // sResult[iCell] = distMin;
                minResult = std::min(minResult, sResult[iCell]);
            }
            // std::cout << minResult << " M \n";
        }

        void EvaluateError(real &err, real &errMax, ArrayDOF<1u> &u, ArrayLocal<SemiVarMatrix<1u>> &uRec)
        {
            auto mpi = uRec.dist->getMPI();
            real volD = 0.0;
            real errD = 0.0;
            real errDM = 0.0;
            for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
                if (sResult[iCell] < MaxD)
                {
                    Eigen::MatrixXd recval = vfv->cellDiBjCenterBatch->operator[](iCell).m(0)({0}, Eigen::all).rightCols(uRec[iCell].m().rows()) * uRec[iCell].m();
                    real diff = (recval(0) + u[iCell](0)) - sResult[iCell];
                    errD += std::fabs(diff) / sResult[iCell];
                    volD += 1.;
                    errDM = std::max(errDM, std::fabs(diff) / sResult[iCell]);
                }
            }
            real vol;
            MPI_Allreduce(&volD, &vol, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            MPI_Allreduce(&errD, &err, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            MPI_Allreduce(&errDM, &errMax, 1, DNDS_MPI_REAL, MPI_MAX, mpi.comm);
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

        ArrayDOF<1u> u, uPoisson;
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
            int recOrder = 2;
            int nInternalRecStep = 1;
            int nTimeStep = 1000;
            int nConsoleCheck = 10;
            int nDataOut = 50;

            real CFL = 0.5;

            real meshRotZ = 0;
            std::string mName = "data/mesh/NACA0012_WIDE_H3.msh";
            std::string outPltName = "data/out/debugData_";
            std::string outLogName = "data/out/debugData_";
            real err_dMax = 0.1;

            real res_base = 0;

            VRFiniteVolume2D::Setting vfvSetting;

            int nDropVisScale;
            real vDropVisScale;
            EikonalEvaluator::Setting eikonalSetting;

            int curvilinearOneStep = 500;
            int curvilinearRepeatInterval = 500;
            int curvilinearRepeatNum = 10;

            int curvilinearRestartNstep = 100;
            real curvilinearRange = 0.1;

            bool useLocalDt = true;
            int nForceLocalStartStep = -1;
        } config;

        void ConfigureFromJson(const std::string &jsonName)
        {
            rapidjson::Document doc;
            JSON::ReadFile(jsonName, doc);

            assert(doc["nInternalRecStep"].IsInt());
            config.nInternalRecStep = doc["nInternalRecStep"].GetInt();
            if (mpi.rank == 0)
            {
                log() << "JSON: nInternalRecStep = " << config.nInternalRecStep << std::endl;
            }

            assert(doc["recOrder"].IsInt());
            config.recOrder = doc["recOrder"].GetInt();
            if (mpi.rank == 0)
            {
                log() << "JSON: recOrder = " << config.recOrder << std::endl;
                if (config.recOrder < 2 || config.recOrder > 3)
                {
                    log() << "Eikonal Error: recOrder bad! " << std::endl;
                    abort();
                }
            }

            assert(doc["nTimeStep"].IsInt());
            config.nTimeStep = doc["nTimeStep"].GetInt();
            if (mpi.rank == 0)
                log() << "JSON: nTimeStep = " << config.nTimeStep << std::endl;

            assert(doc["nConsoleCheck"].IsInt());
            config.nConsoleCheck = doc["nConsoleCheck"].GetInt();
            if (mpi.rank == 0)
                log() << "JSON: nConsoleCheck = " << config.nConsoleCheck << std::endl;

            assert(doc["nDataOut"].IsInt());
            config.nDataOut = doc["nDataOut"].GetInt();
            if (mpi.rank == 0)
                log() << "JSON: nDataOut = " << config.nDataOut << std::endl;

            assert(doc["CFL"].IsNumber());
            config.CFL = doc["CFL"].GetDouble();
            if (mpi.rank == 0)
                log() << "JSON: CFL = " << config.CFL << std::endl;

            assert(doc["meshRotZ"].IsNumber());
            config.meshRotZ = doc["meshRotZ"].GetDouble();
            if (mpi.rank == 0)
                log() << "JSON: meshRotZ = " << config.meshRotZ << std::endl;

            assert(doc["meshFile"].IsString());
            config.mName = doc["meshFile"].GetString();
            if (mpi.rank == 0)
                log() << "JSON: meshFile = " << config.mName << std::endl;

            assert(doc["outLogName"].IsString());
            config.outLogName = doc["outLogName"].GetString();
            if (mpi.rank == 0)
                log() << "JSON: outLogName = " << config.outLogName << std::endl;

            assert(doc["outPltName"].IsString());
            config.outPltName = doc["outPltName"].GetString();
            if (mpi.rank == 0)
                log() << "JSON: outPltName = " << config.outPltName << std::endl;

            assert(doc["err_dMax"].IsNumber());
            config.err_dMax = doc["err_dMax"].GetDouble();
            if (mpi.rank == 0)
                log() << "JSON: err_dMax = " << config.err_dMax << std::endl;

            assert(doc["res_base"].IsNumber());
            config.res_base = doc["res_base"].GetDouble();
            if (mpi.rank == 0)
                log() << "JSON: res_base = " << config.res_base << std::endl;

            assert(doc["useLocalDt"].IsBool());
            config.useLocalDt = doc["useLocalDt"].GetBool();
            if (mpi.rank == 0)
                log() << "JSON: useLocalDt = " << config.useLocalDt << std::endl;

            assert(doc["nForceLocalStartStep"].IsInt());
            config.nForceLocalStartStep = doc["nForceLocalStartStep"].GetInt();
            if (mpi.rank == 0)
            {
                log() << "JSON: nForceLocalStartStep = " << config.nForceLocalStartStep << std::endl;
            }

            if (doc["vfvSetting"].IsObject())
            {
                if (doc["vfvSetting"]["SOR_Instead"].IsBool())
                {
                    config.vfvSetting.SOR_Instead = doc["vfvSetting"]["SOR_Instead"].GetBool();
                    if (mpi.rank == 0)
                        log() << "JSON: vfvSetting.SOR_Instead = " << config.vfvSetting.SOR_Instead << std::endl;
                }
                if (doc["vfvSetting"]["SOR_InverseScanning"].IsBool())
                {
                    config.vfvSetting.SOR_InverseScanning = doc["vfvSetting"]["SOR_InverseScanning"].GetBool();
                    if (mpi.rank == 0)
                        log() << "JSON: vfvSetting.SOR_InverseScanning = " << config.vfvSetting.SOR_InverseScanning << std::endl;
                }
                if (doc["vfvSetting"]["SOR_RedBlack"].IsBool())
                {
                    config.vfvSetting.SOR_RedBlack = doc["vfvSetting"]["SOR_RedBlack"].GetBool();
                    if (mpi.rank == 0)
                        log() << "JSON: vfvSetting.SOR_RedBlack = " << config.vfvSetting.SOR_RedBlack << std::endl;
                }
                if (doc["vfvSetting"]["JacobiRelax"].IsNumber())
                {
                    config.vfvSetting.JacobiRelax = doc["vfvSetting"]["JacobiRelax"].GetDouble();
                    if (mpi.rank == 0)
                        log() << "JSON: vfvSetting.JacobiRelax = " << config.vfvSetting.JacobiRelax << std::endl;
                }

                if (doc["vfvSetting"]["tangWeight"].IsNumber())
                {
                    config.vfvSetting.tangWeight = doc["vfvSetting"]["tangWeight"].GetDouble();
                    if (mpi.rank == 0)
                        log() << "JSON: vfvSetting.tangWeight = " << config.vfvSetting.tangWeight << std::endl;
                }

                if (doc["vfvSetting"]["anistropicLengths"].IsBool())
                {
                    config.vfvSetting.anistropicLengths = doc["vfvSetting"]["anistropicLengths"].GetBool();
                    if (mpi.rank == 0)
                        log() << "JSON: vfvSetting.anistropicLengths = " << config.vfvSetting.anistropicLengths << std::endl;
                }

                if (doc["vfvSetting"]["baseCenterType"].IsString())
                {
                    std::string centerOpt = doc["vfvSetting"]["baseCenterType"].GetString();
                    config.vfvSetting.baseCenterTypeName = centerOpt;
                    if (centerOpt == "Param")
                        config.vfvSetting.baseCenterType = VRFiniteVolume2D::Setting::BaseCenterType::Paramcenter;
                    else if (centerOpt == "Bary")
                        config.vfvSetting.baseCenterType = VRFiniteVolume2D::Setting::BaseCenterType::Barycenter;
                    else
                        assert(false);
                    if (mpi.rank == 0)
                        log() << "JSON: vfvSetting.baseCenterType = " << config.vfvSetting.baseCenterTypeName << std::endl;
                }

                if (doc["vfvSetting"]["weightSchemeGeom"].IsString())
                {
                    std::string centerOpt = doc["vfvSetting"]["weightSchemeGeom"].GetString();
                    config.vfvSetting.weightSchemeGeomName = centerOpt;
                    if (centerOpt == "None")
                        config.vfvSetting.weightSchemeGeom = VRFiniteVolume2D::Setting::WeightSchemeGeom::None;
                    else if (centerOpt == "D")
                        config.vfvSetting.weightSchemeGeom = VRFiniteVolume2D::Setting::WeightSchemeGeom::D;
                    else if (centerOpt == "S")
                        config.vfvSetting.weightSchemeGeom = VRFiniteVolume2D::Setting::WeightSchemeGeom::S;
                    else
                        assert(false);
                    if (mpi.rank == 0)
                        log() << "JSON: vfvSetting.weightSchemeGeom = " << config.vfvSetting.weightSchemeGeomName << std::endl;
                }

                if (doc["vfvSetting"]["scaleMLargerPortion"].IsNumber())
                {
                    config.vfvSetting.scaleMLargerPortion = doc["vfvSetting"]["scaleMLargerPortion"].GetDouble();
                    if (mpi.rank == 0)
                        log() << "JSON: vfvSetting.scaleMLargerPortion = " << config.vfvSetting.scaleMLargerPortion << std::endl;
                }
                if (doc["vfvSetting"]["wallWeight"].IsNumber())
                {
                    config.vfvSetting.wallWeight = doc["vfvSetting"]["wallWeight"].GetDouble();
                    if (mpi.rank == 0)
                        log() << "JSON: vfvSetting.wallWeight = " << config.vfvSetting.wallWeight << std::endl;
                }
                if (doc["vfvSetting"]["farWeight"].IsNumber())
                {
                    config.vfvSetting.farWeight = doc["vfvSetting"]["farWeight"].GetDouble();
                    if (mpi.rank == 0)
                        log() << "JSON: vfvSetting.farWeight = " << config.vfvSetting.farWeight << std::endl;
                }
                if (doc["vfvSetting"]["curvilinearOrder"].IsInt())
                {
                    config.vfvSetting.curvilinearOrder = doc["vfvSetting"]["curvilinearOrder"].GetInt();
                    if (mpi.rank == 0)
                        log() << "JSON: vfvSetting.curvilinearOrder = " << config.vfvSetting.curvilinearOrder << std::endl;
                }
            }
            if (doc["nDropVisScale"].IsInt())
            {
                config.nDropVisScale = doc["nDropVisScale"].GetInt();
                if (mpi.rank == 0)
                    log() << "JSON: nDropVisScale = " << config.nDropVisScale << std::endl;
            }

            if (doc["vDropVisScale"].IsNumber())
            {
                config.vDropVisScale = doc["vDropVisScale"].GetDouble();
                if (mpi.rank == 0)
                    log() << "JSON: vDropVisScale = " << config.vDropVisScale << std::endl;
            }

            if (doc["eikonalSetting"].IsObject())
            {
                if (doc["eikonalSetting"]["visScale"].IsNumber())
                {
                    config.eikonalSetting.visScale = doc["eikonalSetting"]["visScale"].GetDouble();
                    if (mpi.rank == 0)
                        log() << "JSON: eikonalSetting.visScale = " << config.eikonalSetting.visScale << std::endl;
                }
                if (doc["eikonalSetting"]["visScaleIn"].IsNumber())
                {
                    config.eikonalSetting.visScaleIn = doc["eikonalSetting"]["visScaleIn"].GetDouble();
                    if (mpi.rank == 0)
                        log() << "JSON: eikonalSetting.visScaleIn = " << config.eikonalSetting.visScaleIn << std::endl;
                }
                if (doc["eikonalSetting"]["ekCutDown"].IsNumber())
                {
                    config.eikonalSetting.ekCutDown = doc["eikonalSetting"]["ekCutDown"].GetDouble();
                    if (mpi.rank == 0)
                        log() << "JSON: eikonalSetting.ekCutDown = " << config.eikonalSetting.ekCutDown << std::endl;
                }
                if (doc["eikonalSetting"]["isiScale"].IsNumber())
                {
                    config.eikonalSetting.isiScale = doc["eikonalSetting"]["isiScale"].GetDouble();
                    if (mpi.rank == 0)
                        log() << "JSON: eikonalSetting.isiScale = " << config.eikonalSetting.isiScale << std::endl;
                }
                if (doc["eikonalSetting"]["isiScaleIn"].IsNumber())
                {
                    config.eikonalSetting.isiScaleIn = doc["eikonalSetting"]["isiScaleIn"].GetDouble();
                    if (mpi.rank == 0)
                        log() << "JSON: eikonalSetting.isiScaleIn = " << config.eikonalSetting.isiScaleIn << std::endl;
                }
                if (doc["eikonalSetting"]["isiCutDown"].IsNumber())
                {
                    config.eikonalSetting.isiCutDown = doc["eikonalSetting"]["isiCutDown"].GetDouble();
                    if (mpi.rank == 0)
                        log() << "JSON: eikonalSetting.isiCutDown = " << config.eikonalSetting.isiCutDown << std::endl;
                }
            }

            if (doc["curvilinearOneStep"].IsInt())
            {
                config.curvilinearOneStep = doc["curvilinearOneStep"].GetInt();
                if (mpi.rank == 0)
                    log() << "JSON: curvilinearOneStep = " << config.curvilinearOneStep << std::endl;
            }
            if (doc["curvilinearRestartNstep"].IsInt())
            {
                config.curvilinearRestartNstep = doc["curvilinearRestartNstep"].GetInt();
                if (mpi.rank == 0)
                    log() << "JSON: curvilinearRestartNstep = " << config.curvilinearRestartNstep << std::endl;
            }
            if (doc["curvilinearRepeatInterval"].IsInt())
            {
                config.curvilinearRepeatInterval = doc["curvilinearRepeatInterval"].GetInt();
                if (mpi.rank == 0)
                    log() << "JSON: curvilinearRepeatInterval = " << config.curvilinearRepeatInterval << std::endl;
            }
            if (doc["curvilinearRepeatNum"].IsInt())
            {
                config.curvilinearRepeatNum = doc["curvilinearRepeatNum"].GetInt();
                if (mpi.rank == 0)
                    log() << "JSON: curvilinearRepeatNum = " << config.curvilinearRepeatNum << std::endl;
            }
            if (doc["curvilinearRange"].IsNumber())
            {
                config.curvilinearRange = doc["curvilinearRange"].GetDouble();
                if (mpi.rank == 0)
                    log() << "JSON: curvilinearRange = " << config.curvilinearRange << std::endl;
            }
        }

        void ReadMeshAndInitialize()
        {
            // Debug::MPIDebugHold(mpi);
            CompactFacedMeshSerialRWBuild(mpi, config.mName, "data/out/debugmeshSO.plt", mesh, config.meshRotZ);
            fv = std::make_shared<ImplicitFiniteVolume2D>(mesh.get());
            vfv = std::make_shared<VRFiniteVolume2D>(mesh.get(), fv.get(), config.recOrder);
            vfv->setting = config.vfvSetting; //* currently only copies, could upgrade to referencing
            vfv->Initialization();

            cfv = std::make_shared<CRFiniteVolume2D>(*vfv);
            cfv->Initialization();

            fv->BuildMean(u);
            fv->BuildMean(uPoisson);
            vfv->BuildRec(uRec);
            uRecNew.Copy(uRec);
            cfv->BuildRec(uRecCR);

            u.setConstant(0.0);
            uPoisson.setConstant(0);

            outDist = std::make_shared<decltype(outDist)::element_type>(
                decltype(outDist)::element_type::tContext(mesh->cell2faceLocal.dist->size()), mpi);
            outSerial = std::make_shared<decltype(outDist)::element_type>(outDist.get());
            outSerial->BorrowGGIndexing(*mesh->cell2node);
            outSerial->createMPITypes();
            outSerial->initPersistentPull();

            err = std::make_shared<EikonalErrorEvaluator>(mesh.get(), fv.get(), vfv.get(), config.err_dMax);
            forEachInArray(*u.dist, [&](decltype(u.dist)::element_type::tComponent &e, index iCell)
                           { e.p()(0) = err->sResult[iCell] * 0; });
        }

        void RunExplicitSSPRK4()
        {

            ODE::ExplicitSSPRK3LocalDt<decltype(u)> ode(
                u.dist->size(),
                [&](decltype(u) &data)
                {
                    data.resize(u.dist->size(), u.dist->getMPI());
                    data.CreateGhostCopyComm(mesh->cell2faceLocal);
                });
            EikonalEvaluator eval(mesh.get(), fv.get(), vfv.get(), cfv.get());
            std::ofstream logErr(config.outLogName + ".log");
            eval.settings = config.eikonalSetting;
            uRec.InitPersistentPullClean();
            u.InitPersistentPullClean();
            // u.StartPersistentPullClean();
            double tstart = MPI_Wtime();
            double trec{0}, treccr{0}, tcomm{0}, trhs{0};
            int stepCount = 0;
            real resBaseC = config.res_base;

            // Doing Poisson Init:
            uPoisson.InitPersistentPullClean();
            // forEachInArrayPair(*uPoisson.pair, [&](decltype(uPoisson.dist)::element_type::tComponent &e, index iCell)
            //                    { std::cout << "UPPair: " << e << std::endl; });
            eval.PoissonInit(uPoisson, u, 10000, 1);
            for (int iRec = 0; iRec < 20; iRec++)
            {
                vfv->ReconstructionJacobiStep(u, uRec, uRecNew);
                uRec.StartPersistentPullClean();
                uRec.WaitPersistentPullClean();
            }

            int curvilinearNum = 0;
            int curvilinearStepper = 0;
            for (int step = 1; step <= config.nTimeStep; step++)
            {
                if (step == config.nForceLocalStartStep)
                    config.useLocalDt = true;
                if (step == config.nDropVisScale)
                    eval.settings.visScale *= config.vDropVisScale;

                curvilinearStepper++;
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

                        for (int iRec = 0; iRec < config.nInternalRecStep; iRec++)
                        {
                            double tstartA = MPI_Wtime();
                            vfv->ReconstructionJacobiStep(cx, uRec, uRecNew);
                            trec += MPI_Wtime() - tstartA;

                            double tstartF = MPI_Wtime();
                            uRec.StartPersistentPullClean();
                            tcomm += MPI_Wtime() - tstartF;

                            double tstartG = MPI_Wtime();
                            uRec.WaitPersistentPullClean();
                            tcomm += MPI_Wtime() - tstartG;
                        }

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

                        eval.EvaluateDt(dt, uRec, config.CFL, 1e100, config.useLocalDt);
                    });
                real res;
                eval.EvaluateResidual(res, ode.rhsbuf[0]);
                if (stepCount == 0 && resBaseC == 0)
                    resBaseC = res;

                real error, errorMax;

                if (step % config.nConsoleCheck == 0)
                {
                    err->EvaluateError(error, errorMax, u, uRec);
                    double telapsed = MPI_Wtime() - tstart;
                    if (mpi.rank == 0)
                    {
                        auto fmt = log().flags();
                        log() << std::setprecision(6) << std::scientific
                              << "=== Step [" << step << "]   "
                              << "res \033[91m[" << res / resBaseC << "]\033[39m   "
                              << "err \033[93m[" << error << "]\033[39m   "
                              << "errM \033[93m[" << errorMax << "]\033[39m   "
                              << std::setprecision(3) << std::fixed
                              << "Time [" << telapsed << "]   recTime [" << trec << "]   reccrTime [" << treccr
                              << "]   rhsTime [" << trhs << "]   commTime [" << tcomm << "]  " << std::endl;
                        log().setf(fmt);
                        logErr << step << "\t" << std::setprecision(9) << std::scientific
                               << res / resBaseC << "\t"
                               << error << "\t"
                               << errorMax << std::endl;
                    }
                    tstart = MPI_Wtime();
                    trec = tcomm = treccr = trhs = 0.;
                }
                if (step % config.nDataOut == 0)
                {
                    PrintData(config.outPltName + std::to_string(step) + ".plt", ode);
                }
#ifdef USE_LOCAL_COORD_CURVILINEAR
                if ((curvilinearStepper == config.curvilinearOneStep && curvilinearNum == 0) ||
                    (curvilinearStepper == config.curvilinearRepeatInterval && (curvilinearNum > 0 && curvilinearNum < config.curvilinearRepeatNum)))
                {
                    assert(!vfv->setting.anistropicLengths);
                    curvilinearStepper = 0;
                    curvilinearNum++;

                    forEachInArray(
                        *vfv->uCurve.dist,
                        [&](decltype(vfv->uCurve.dist)::element_type::tComponent &e, index iCell)
                        {
                            if (u[iCell](0) > config.curvilinearRange)
                                return;
                            auto em = e.m();

                            em.setZero();
                            em(0, 0) = em(1, 1) = 1.0;
                            int nZetaDof = em.rows();

                            auto &cellAtr = mesh->cellAtrLocal[iCell][0];
                            auto &cellAtrRec = vfv->cellRecAtrLocal[iCell][0];
                            auto eCell = Elem::ElementManager(cellAtr.type, cellAtrRec.intScheme);
                            Eigen::MatrixXd coords;
                            mesh->LoadCoords(mesh->cell2nodeLocal[iCell], coords);
                            Elem::tPoint sScale = vfv->CoordMinMaxScale(coords);
                            Elem::tPoint center = vfv->getCellCenter(iCell);

                            Eigen::MatrixXd A(nZetaDof, nZetaDof);
                            A.setZero();
                            eCell.Integration(
                                A,
                                [&](Eigen::MatrixXd &inc, int ig, Elem::tPoint pparam, Elem::tDiFj &DiNj)
                                {
                                    Eigen::MatrixXd incFull;
                                    Eigen::MatrixXd DiBj(6, nZetaDof + 1); //*remember add 1 Dof for constvalue-base
                                    vfv->FDiffBaseValue(
                                        iCell, eCell, coords, DiNj,
                                        pparam, center, sScale,
                                        Eigen::VectorXd::Zero(nZetaDof + 1),
                                        DiBj);
                                    Eigen::MatrixXd DiBjSlice = DiBj({1, 2}, Eigen::all);
                                    Eigen::MatrixXd DiBjSlice2 = DiBj({3, 4, 5}, Eigen::all);
                                    Eigen::VectorXd Weights(6);
                                    real L = sScale(0);
                                    Weights << 0, L, L, L * L, 2 * L * L, L * L;

                                    // incFull = DiBjSlice.transpose() * DiBjSlice + DiBjSlice2.transpose();
                                    incFull = DiBj.transpose() * Weights.asDiagonal() * DiBj;

                                    inc = incFull.bottomRightCorner(incFull.rows() - 1, incFull.cols() - 1);
                                    inc *= vfv->cellGaussJacobiDets[iCell][ig];
                                });

                            // std::cout << "Amat good \n"
                            //           << std::endl;

                            Eigen::MatrixXd b(nZetaDof, 2);
                            b.setZero();
                            eCell.Integration(
                                b,
                                [&](Eigen::MatrixXd &inc, int ig, Elem::tPoint pparam, Elem::tDiFj &DiNj)
                                {
                                    Eigen::MatrixXd incFull;
                                    Eigen::MatrixXd DiBj(6, nZetaDof + 1);
                                    Elem::tPoint pPhysics = coords * DiNj(0, Eigen::all).transpose();
                                    vfv->FDiffBaseValue(
                                        iCell, eCell, coords, DiNj,
                                        pparam, center, sScale,
                                        Eigen::VectorXd::Zero(nZetaDof + 1),
                                        DiBj);
                                    Eigen::MatrixXd DiBjSlice = DiBj({1, 2}, Eigen::all); //? why can't use auto to recieve
                                    // Eigen::MatrixXd DiBjSlice0 = DiBj({0}, Eigen::all);
                                    // real recVal = (vfv->cellDiBjGaussBatch->operator[](iCell).m(ig)({0}, Eigen::all).rightCols(uRec[iCell].m().rows()) * uRec[iCell].m())(0);
                                    Eigen::Vector2d recGrad = vfv->cellDiBjGaussBatch->operator[](iCell).m(ig)({1, 2}, Eigen::all).rightCols(uRec[iCell].m().rows()) * uRec[iCell].m();
                                    Eigen::Matrix2d recGrad01;
                                    recGrad01.col(0) = recGrad;
                                    recGrad01.col(1)(0) = -recGrad(1), recGrad01.col(1)(1) = recGrad(0);
                                    incFull = DiBjSlice.transpose() * recGrad01;

                                    Eigen::VectorXd Weights(6);
                                    real L = sScale(0);
                                    Weights << 0, L, L, L * L, 2 * L * L, L * L;
                                    Eigen::MatrixXd recAll = vfv->cellDiBjGaussBatch->operator[](iCell).m(ig)({0, 1, 2, 3, 4, 5}, Eigen::all).rightCols(uRec[iCell].m().rows()) *
                                                             uRec[iCell].m();
                                    Eigen::MatrixXd recAll2(6, 2);
                                    recAll2.col(0) = recAll;
                                    recAll2.col(1).setZero();
                                    recAll2.col(1)(1) = -recAll(2), recAll2.col(1)(2) = recAll(1);
                                    incFull = DiBj.transpose() * Weights.asDiagonal() * recAll2;

                                    inc = incFull.bottomRows(incFull.rows() - 1);
                                    inc *= vfv->cellGaussJacobiDets[iCell][ig];
                                });
                            Eigen::MatrixXd Ainv;
                            HardEigen::EigenLeastSquareInverse(A, Ainv);
                            em = Ainv * b;
                            Eigen::MatrixXd lengths = em({0, 1}, Eigen::all).colwise().norm();
                            real length0 = lengths.norm() / std::sqrt(2);
                            length0 = sScale(0);
                            em /= length0;

                            // std::cout << "REC " << uRec[iCell].m().transpose();
                            // std::cout << " EM \n"
                            //           << std::scientific << std::setprecision(6) << em.transpose() << std::endl;
                            // exit(123);
                        });

                    vfv->uCurve.StartPersistentPullClean();
                    vfv->uCurve.WaitPersistentPullClean();
                    // InsertCheck(mpi, "CHECK VFVRENEW B");
                    vfv->Initialization_RenewBase();
                    // InsertCheck(mpi, "CHECK VFVRENEW");
                    cfv = std::make_shared<CRFiniteVolume2D>(*vfv);
                    // InsertCheck(mpi, "CHECK CFVDONE");
                    cfv->Initialization();
                    // std::cout << cfv->baseMoments.size() << "cfv- "<< cfv->faceNormCenter[0].size()
                    eval.cfv = cfv.get();
                    forEachInArray(
                        *uRec.dist,
                        [&](decltype(uRec.dist)::element_type::tComponent &e, index iCell)
                        {
                            e.m().setZero();
                        });
                    for (int i = 0; i < config.curvilinearRestartNstep; i++)
                    {
                        uRec.StartPersistentPullClean();
                        uRec.WaitPersistentPullClean();
                        vfv->ReconstructionJacobiStep(u, uRec, uRecNew);
                        if (mpi.rank == 0)
                            log() << "--- Restart Reconstruction " << i << std::endl;
                    }
                }
#endif

                stepCount++;
            }

            // u.WaitPersistentPullClean();
            logErr.close();
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