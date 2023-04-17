#include "DNDS_FV_EulerEvaluator.hpp"

#include "DNDS_GasGen.hpp"
#include "DNDS_HardEigen.h"

namespace DNDS
{

    // Eigen::Vector<real, -1> EulerEvaluator::CompressRecPart(
    //     const Eigen::Vector<real, -1> &umean,
    //     const Eigen::Vector<real, -1> &uRecInc)

    //! evaluates dt and facial spectral radius
    template <EulerModel model>
    void EulerEvaluator<model>::EvaluateDt(std::vector<real> &dt,
                                           ArrayDOFV<nVars_Fixed> &u,
                                           real CFL, real &dtMinall, real MaxDt,
                                           bool UseLocaldt)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        InsertCheck(u.dist->getMPI(), "EvaluateDt 1");
        for (auto &i : lambdaCell)
            i = 0.0;

        for (index iFace = 0; iFace < mesh->face2nodeLocal.size(); iFace++)
        {
            auto f2c = mesh->face2cellLocal[iFace];
            auto faceDiBjCenterBatchElemVR = (*vfv->faceDiBjCenterBatch)[iFace];
            TVec unitNorm = vfv->faceNormCenter[iFace](Seq012).normalized();

            index iCellL = f2c[0];
            auto UL = u[iCellL];
            TU uMean = UL;
            real pL, asqrL, HL, pR, asqrR, HR;
            TVec vL = UL(Seq123) / UL(0);
            TVec vR = vL;
            Gas::IdealGasThermal(UL(I4), UL(0), vL.squaredNorm(),
                                 settings.idealGasProperty.gamma,
                                 pL, asqrL, HL);
            pR = pL, HR = HL, asqrR = asqrL;
            if (f2c[1] != FACE_2_VOL_EMPTY)
            {
                auto UR = u[f2c[1]];
                uMean = (uMean + UR) * 0.5;
                vR = UR(Seq123) / UR(0);
                Gas::IdealGasThermal(UR(I4), UR(0), vR.squaredNorm(),
                                     settings.idealGasProperty.gamma,
                                     pR, asqrR, HR);
            }
            assert(uMean(0) > 0);
            TVec veloMean = (uMean(Seq123).array() / uMean(0)).matrix();
            // real veloNMean = veloMean.dot(unitNorm); // original
            real veloNMean = 0.5 * (vL + vR).dot(unitNorm); // paper

            // real ekFixRatio = 0.001;
            // Eigen::Vector3d velo = uMean({1, 2, 3}) / uMean(0);
            // real vsqr = velo.squaredNorm();
            // real Ek = vsqr * 0.5 * uMean(0);
            // real Efix = Ek * ekFixRatio;
            // real e = uMean(4) - Ek;
            // if (e < 0)
            //     e = 0.5 * Efix;
            // else if (e < Efix)
            //     e = (e * e + Efix * Efix) / (2 * Efix);
            // uMean(4) = Ek + e;

            real pMean, asqrMean, HMean;
            Gas::IdealGasThermal(uMean(I4), uMean(0), veloMean.squaredNorm(),
                                 settings.idealGasProperty.gamma,
                                 pMean, asqrMean, HMean);

            pMean = (pL + pR) * 0.5;
            real aMean = sqrt(settings.idealGasProperty.gamma * pMean / uMean(0)); // paper

            // assert(asqrMean >= 0);
            // real aMean = std::sqrt(asqrMean); // original
            real lambdaConvection = std::abs(veloNMean) + aMean;

            // ! refvalue:
            real muRef = settings.idealGasProperty.muGas;

            real gamma = settings.idealGasProperty.gamma;
            real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * uMean(0));
            real muf = settings.idealGasProperty.muGas *
                       std::pow(T / settings.idealGasProperty.TRef, 1.5) *
                       (settings.idealGasProperty.TRef + settings.idealGasProperty.CSutherland) /
                       (T + settings.idealGasProperty.CSutherland);
            if constexpr (model == NS_SA)
            {
                real cnu1 = 7.1;
                real Chi = uMean(I4 + 1) * muRef / muf;
#ifdef USE_NS_SA_NEGATIVE_MODEL
                if (Chi < 10)
                    Chi = 0.05 * std::log(1 + std::exp(20 * Chi));
#endif
                real Chi3 = std::pow(Chi, 3);
                real fnu1 = Chi3 / (Chi3 + std::pow(cnu1, 3));
                muf *= std::max((1 + Chi * fnu1), 1.0);
            }
            real lamVis = muf / uMean(0) *
                          std::max(4. / 3., gamma / settings.idealGasProperty.prGas);

            // Elem::tPoint gradL{0, 0, 0},gradR{0, 0, 0};
            // gradL({0, 1}) = faceDiBjCenterBatchElemVR.m(0)({1, 2}, Eigen::all).rightCols(faceDiBjCenterBatchElemVR.m(0).cols() - 1) * uRec[iCellL];
            // if (f2c[1] != FACE_2_VOL_EMPTY) // can't be non local
            //     gradR({0, 1}) = faceDiBjCenterBatchElemVR.m(1)({1, 2}, Eigen::all).rightCols(faceDiBjCenterBatchElemVR.m(1).cols() - 1) * uRec[f2c[1]];
            // Elem::tPoint grad = (gradL + gradR) * 0.5;

            real lamFace = lambdaConvection * fv->faceArea[iFace];

            real area = fv->faceArea[iFace];
            real areaSqr = area * area;
            real volR = fv->volumeLocal[iCellL];
            // lambdaCell[iCellL] += lamFace + 2 * lamVis * areaSqr / fv->volumeLocal[iCellL];
            if (f2c[1] != FACE_2_VOL_EMPTY) // can't be non local
                                            // lambdaCell[f2c[1]] += lamFace + 2 * lamVis * areaSqr / fv->volumeLocal[f2c[1]],
                volR = fv->volumeLocal[f2c[1]];

            lambdaFace[iFace] = lambdaConvection + lamVis * area * (1. / fv->volumeLocal[iCellL] + 1. / volR);
            lambdaFaceC[iFace] = std::abs(veloNMean) + lamVis * area * (1. / fv->volumeLocal[iCellL] + 1. / volR); // passive part
            lambdaFaceVis[iFace] = lamVis * area * (1. / fv->volumeLocal[iCellL] + 1. / volR);

            if (f2c[0] == 10756)
            {
                std::cout << "----Lambdas" << std::setprecision(16) << iFace << std::endl;
                std::cout << lambdaConvection << std::endl;
                std::cout << lambdaFaceVis[iFace] << std::endl;
                std::cout << veloNMean << " " << aMean << std::endl;
                std::cout << gamma << " " << pMean << " " << uMean(0) << std::endl;
            }

            lambdaCell[iCellL] += lambdaFace[iFace] * fv->faceArea[iFace];
            if (f2c[1] != FACE_2_VOL_EMPTY) // can't be non local
                lambdaCell[f2c[1]] += lambdaFace[iFace] * fv->faceArea[iFace];

            deltaLambdaFace[iFace] = std::abs((vR - vL).dot(unitNorm)) + std::sqrt(std::abs(asqrR - asqrL)) * 0.7071;
        }
        real dtMin = veryLargeReal;
        for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
        {
            // std::cout << fv->volumeLocal[iCell] << " " << (lambdaCell[iCell]) << " " << CFL << std::endl;
            // exit(0);
            dt[iCell] = std::min(CFL * fv->volumeLocal[iCell] / (lambdaCell[iCell] + 1e-100), MaxDt);
            dtMin = std::min(dtMin, dt[iCell]);
            if (iCell == 10756)
            {
                std::cout << "dt " << std::setprecision(16) << dt[iCell] << " " << settings.nTimeFilterPass;
                std::cout << std::endl;
            }
        }

        MPI_Allreduce(&dtMin, &dtMinall, 1, DNDS_MPI_REAL, MPI_MIN, u.dist->getMPI().comm);

        for (int iPass = 1; iPass <= settings.nTimeFilterPass; iPass++)
        {
            for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
                auto &c2f = mesh->cell2faceLocal[iCell];
                real dtC = dt[iCell];
                real dtC_N = 1.0;
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    auto &f2c = (*mesh->face2cellPair)[iFace];
                    index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                    index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                    if (iCellOther != FACE_2_VOL_EMPTY && iCellOther < mesh->cell2nodeLocal.dist->size())
                        dtC += dt[iCell] + dt[iCellOther], dtC_N += 2.0;
                }
                dt[iCell] = dtC / dtC_N;
            }
        }

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
    template void EulerEvaluator<NS>::EvaluateDt(std::vector<real> &dt,
                                                 ArrayDOFV<nVars_Fixed> &u,
                                                 real CFL, real &dtMinall, real MaxDt,
                                                 bool UseLocaldt);
    template void EulerEvaluator<NS_SA>::EvaluateDt(std::vector<real> &dt,
                                                    ArrayDOFV<nVars_Fixed> &u,
                                                    real CFL, real &dtMinall, real MaxDt,
                                                    bool UseLocaldt);
    template void EulerEvaluator<NS_2D>::EvaluateDt(std::vector<real> &dt,
                                                    ArrayDOFV<nVars_Fixed> &u,
                                                    real CFL, real &dtMinall, real MaxDt,
                                                    bool UseLocaldt);

#define IF_NOT_NOREC (1)
    template <EulerModel model>
    void EulerEvaluator<model>::EvaluateRHS(ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u,
                                            ArrayRecV &uRec, real t)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        InsertCheck(u.dist->getMPI(), "EvaluateRHS 1");
        int cnvars = nVars;
        typename Setting::RiemannSolverType rsType = settings.rsType;
        for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
        {
            rhs[iCell].setZero();
        }
        TU fluxWallSumLocal;
        fluxWallSumLocal.setZero(cnvars);
        fluxWallSum.setZero(cnvars);
        nFaceReducedOrder = 0;

        for (index iFace = 0; iFace < mesh->face2nodeLocal.size(); iFace++)
        {
            auto &faceRecAtr = vfv->faceRecAtrLocal[iFace][0];
            auto &faceAtr = mesh->faceAtrLocal[iFace][0];
            auto f2c = mesh->face2cellLocal[iFace];
            Elem::ElementManager eFace(faceAtr.type, faceRecAtr.intScheme);
#ifdef USE_FLUX_BALANCE_TERM
            Eigen::Matrix<real, nVars_Fixed, 3, Eigen::ColMajor> fluxEs(cnvars, 3);
#else
            Eigen::Matrix<real, nVars_Fixed, 1, Eigen::ColMajor> fluxEs(cnvars, 1);
#endif

            fluxEs.setZero();
            auto faceDiBjGaussBatchElemVR = (*vfv->faceDiBjGaussBatch)[iFace];

            auto f2n = mesh->face2nodeLocal[iFace];
            Eigen::MatrixXd coords;
            mesh->LoadCoords(f2n, coords);

            Elem::SummationNoOp noOp;
            bool faceOrderReducedL = false;
            bool faceOrderReducedR = false;

#ifdef USE_TOTAL_REDUCED_ORDER_CELL
            eFace.Integration(
                noOp,
                [&](decltype(noOp) &finc, int ig, Elem::tPoint &p, Elem::tDiFj &DiNj)
                {
                    if (!faceOrderReducedL)
                    {
                        TU ULxy =
                            faceDiBjGaussBatchElemVR.m(ig * 2 + 0).row(0).rightCols(uRec[f2c[0]].rows()) *
                            uRec[f2c[0]] * IF_NOT_NOREC;
                        ULxy = CompressRecPart(u[f2c[0]], ULxy, faceOrderReducedL);
                    }

                    if (f2c[1] != FACE_2_VOL_EMPTY)
                    {
                        if (!faceOrderReducedR)
                        {
                            TU URxy =
                                faceDiBjGaussBatchElemVR.m(ig * 2 + 1).row(0).rightCols(uRec[f2c[1]].rows()) *
                                uRec[f2c[1]] * IF_NOT_NOREC;
                            URxy = CompressRecPart(u[f2c[1]], URxy, faceOrderReducedR);
                        }
                    }
                });
#endif

            eFace.Integration(
                fluxEs,
                [&](decltype(fluxEs) &finc, int ig, Elem::tPoint &p, Elem::tDiFj &DiNj)
                {
                    int nDiff = vfv->faceWeights->operator[](iFace).size();
                    TVec unitNorm = vfv->faceNorms[iFace][ig](Seq012).normalized();
                    TMat normBase = Elem::NormBuildLocalBaseV(unitNorm);
                    PerformanceTimer::Instance().StartTimer(PerformanceTimer::LimiterB);
                    // TU ULxy =
                    //     faceDiBjGaussBatchElemVR.m(ig * 2 + 0).row(0).rightCols(uRec[f2c[0]].rows()) *
                    //     uRec[f2c[0]] * IF_NOT_NOREC;
                    // UL += u[f2c[0]]; //! do not forget the mean value
                    // bool compressed = false;
                    // ULxy = CompressRecPart(u[f2c[0]], ULxy, compressed);
                    TU ULxy = u[f2c[0]];
                    bool pointOrderReducedL = false;
                    bool pointOrderReducedR = false;
                    if (!faceOrderReducedL)
                    {
                        // ULxy +=
                        //     faceDiBjGaussBatchElemVR.m(ig * 2 + 0).row(0).rightCols(uRec[f2c[0]].rows()) *
                        //     uRec[f2c[0]] * IF_NOT_NOREC;
                        ULxy = CompressRecPart(ULxy,
                                               faceDiBjGaussBatchElemVR.m(ig * 2 + 0).row(0).rightCols(uRec[f2c[0]].rows()) *
                                                   uRec[f2c[0]] * IF_NOT_NOREC,
                                               pointOrderReducedL);
                    }

                    TU ULMeanXy = u[f2c[0]];
                    TU URMeanXy;

                    TU URxy;
#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
                    TDiffU GradULxy, GradURxy;
                    GradULxy.resize(Eigen::NoChange, cnvars);
                    GradURxy.resize(Eigen::NoChange, cnvars);
                    GradULxy.setZero(), GradURxy.setZero();
                    // if (iFace == 16404)
                    // {
                    //     std::cout << "face DIBJ: " << std::endl;
                    //     std::cout << faceDiBjGaussBatchElemVR.m(ig * 2 + 0) << std::endl;
                    //     std::cout << "urec\n";
                    //     std::cout << uRec[f2c[0]] << std::endl;
                    // }

                    if constexpr (gdim == 2)
                        GradULxy({0, 1}, Eigen::all) =
                            faceDiBjGaussBatchElemVR.m(ig * 2 + 0)({1, 2}, Eigen::seq(Eigen::fix<1>, Eigen::last)) *
                            uRec[f2c[0]] * IF_NOT_NOREC; // 2d here
                    else
                        GradULxy({0, 1, 2}, Eigen::all) =
                            faceDiBjGaussBatchElemVR.m(ig * 2 + 0)({1, 2, 3}, Eigen::seq(Eigen::fix<1>, Eigen::last)) *
                            uRec[f2c[0]] * IF_NOT_NOREC; // 3d here

#endif
                    real minVol = fv->volumeLocal[f2c[0]];
                    // InsertCheck(u.dist->getMPI(), "RHS inner 2");

                    if (f2c[1] != FACE_2_VOL_EMPTY)
                    {
                        // URxy =
                        //     faceDiBjGaussBatchElemVR.m(ig * 2 + 1).row(0).rightCols(uRec[f2c[1]].rows()) *
                        //     uRec[f2c[1]] * IF_NOT_NOREC;
                        // // UR += u[f2c[1]];
                        // URxy = CompressRecPart(u[f2c[1]], URxy, compressed);
                        URxy = u[f2c[1]];
                        if (!faceOrderReducedR)
                        {
                            // URxy +=
                            //     faceDiBjGaussBatchElemVR.m(ig * 2 + 1).row(0).rightCols(uRec[f2c[1]].rows()) *
                            //     uRec[f2c[1]] * IF_NOT_NOREC;
                            URxy = CompressRecPart(URxy,
                                                   faceDiBjGaussBatchElemVR.m(ig * 2 + 1).row(0).rightCols(uRec[f2c[1]].rows()) *
                                                       uRec[f2c[1]] * IF_NOT_NOREC,
                                                   pointOrderReducedR);
                        }

                        URMeanXy = u[f2c[1]];
#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
                        if constexpr (gdim == 2)
                            GradURxy({0, 1}, Eigen::all) =
                                faceDiBjGaussBatchElemVR.m(ig * 2 + 1)({1, 2}, Eigen::seq(Eigen::fix<1>, Eigen::last)) *
                                uRec[f2c[1]] * IF_NOT_NOREC; // 2d here
                        else
                            GradURxy({0, 1, 2}, Eigen::all) =
                                faceDiBjGaussBatchElemVR.m(ig * 2 + 1)({1, 2, 3}, Eigen::seq(Eigen::fix<1>, Eigen::last)) *
                                uRec[f2c[1]] * IF_NOT_NOREC; // 3d here

#endif

                        minVol = std::min(minVol, fv->volumeLocal[f2c[1]]);
                    }
                    else if (true)
                    {
                        URxy = generateBoundaryValue(
                            ULxy,
                            unitNorm,
                            normBase,
                            vfv->faceCenters[iFace](Seq012),
                            t,
                            BoundaryType(faceAtr.iPhy), true);
#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
                        GradURxy = GradULxy;
#endif
                        URMeanXy = generateBoundaryValue(
                            ULMeanXy,
                            unitNorm,
                            normBase,
                            vfv->faceCenters[iFace](Seq012),
                            t,
                            BoundaryType(faceAtr.iPhy), false);
                    }
                    PerformanceTimer::Instance().EndTimer(PerformanceTimer::LimiterB);
                    // UR = URxy;
                    // UL = ULxy;
                    // UR({1, 2, 3}) = normBase.transpose() * UR({1, 2, 3});
                    // UL({1, 2, 3}) = normBase.transpose() * UL({1, 2, 3});
                    real distGRP = minVol / fv->faceArea[iFace] * 2;
#ifdef USE_DISABLE_DIST_GRP_FIX_AT_WALL
                    distGRP +=
                        (faceAtr.iPhy == BoundaryType::Wall_Euler || faceAtr.iPhy == BoundaryType::Wall_NoSlip
                             ? veryLargeReal
                             : 0.0);
#endif
                    // real distGRP = (vfv->cellBaries[f2c[0]] -
                    //                 (f2c[1] != FACE_2_VOL_EMPTY
                    //                      ? vfv->cellBaries[f2c[1]]
                    //                      : 2 * vfv->faceCenters[iFace] - vfv->cellBaries[f2c[0]]))
                    //                    .norm();
                    // InsertCheck(u.dist->getMPI(), "RHS inner 1");
                    TU UMeanXy = 0.5 * (ULxy + URxy);

#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
                    TDiffU GradUMeanXy = (GradURxy + GradULxy) * 0.5 +
                                         (1.0 / distGRP) *
                                             (unitNorm * (URxy - ULxy).transpose());

#else
                    TDiffU GradUMeanXy;
#endif
                    // if (u.dist->getMPI().rank == 0)
                    // {
                    //     if constexpr (model == NS)
                    //     {
                    //         finc = fluxFace(
                    //             TU{1, 0, 0, 0, 2.5},
                    //             TU{1, 1, 0, 0, 7.5},
                    //             GradUMeanXy,
                    //             unitNorm,
                    //             normBase,
                    //             BoundaryType(faceAtr.iPhy),
                    //             rsType,
                    //             iFace, ig);

                    //         std::cout << "F::: \n " << GradUMeanXy << "\n  " << unitNorm << "\n   " << normBase << std::endl;
                    //         std::cout << finc << std::endl;
                    //     }
                    //     else if constexpr (model == NS_2D)
                    //     {
                    //         finc = fluxFace(
                    //             TU{1, 0, 0, 2.5},
                    //             TU{1, 1, 0, 7.5},
                    //             GradUMeanXy,
                    //             unitNorm,
                    //             normBase,
                    //             BoundaryType(faceAtr.iPhy),
                    //             rsType,
                    //             iFace, ig);
                    //         std::cout << "F::: \n " << GradUMeanXy << "\n  " << unitNorm << "\n   " << normBase << std::endl;
                    //         std::cout << finc << std::endl;
                    //     }
                    // }

                    // exit(-1);

                    TU FLFix, FRFix;
                    FLFix.setZero(), FRFix.setZero();

                    TU fincC = fluxFace(
                        ULxy,
                        URxy,
                        ULMeanXy,
                        URMeanXy,
                        GradUMeanXy,
                        unitNorm,
                        normBase,
                        FLFix, FRFix,
                        BoundaryType(faceAtr.iPhy),
                        rsType,
                        iFace, ig);

                    // if (iFace == 16404)
                    // {
                    //     std::cout << "RHS face " << iFace << std::endl << std::setprecision(10);
                    //     std::cout << fincC.transpose() << std::endl;
                    //     std::cout << ULxy.transpose() << std::endl;
                    //     std::cout << URxy.transpose() << std::endl;
                    //     std::cout << u[f2c[0]].transpose() << std::endl;
                    //     std::cout << u[f2c[1]].transpose() << std::endl;
                    //     std::cout << GradULxy << std::endl;
                    //     std::cout << GradULxy << std::endl;
                    //     std::cout << GradUMeanXy << std::endl;
                    //     std::cout << f2c[0] << " " << f2c[1] << " " << distGRP << std::endl;
                    //     std::cout << lambdaFace[iFace] << std::endl;
                    //     exit(-1);
                    // }
                    if (f2c[0] == 10756)
                    {
                        std::cout << "RLR " << ULxy.transpose() << " " << URxy.transpose() << " " << std::endl;
                        std::cout << "RO " << pointOrderReducedL << " " << pointOrderReducedR << std::endl;
                    }

                    finc(Eigen::all, 0) = fincC;
#ifdef USE_FLUX_BALANCE_TERM
                    finc(Eigen::all, 1) = FLFix;
                    finc(Eigen::all, 2) = FRFix;
#endif

                    finc *= vfv->faceNorms[iFace][ig].norm(); // don't forget this

                    if (pointOrderReducedL)
                        nFaceReducedOrder++, faceOrderReducedL = false;
                    if (pointOrderReducedR)
                        nFaceReducedOrder++, faceOrderReducedR = false;
                    if (faceOrderReducedL)
                        nFaceReducedOrder++;
                    if (faceOrderReducedR)
                        nFaceReducedOrder++;
                });

            if (f2c[0] == 10756)
            {
                std::cout << std::setprecision(16)
                          << fluxEs(Eigen::all, 0).transpose() << std::endl;
                // exit(-1);
            }

            rhs[f2c[0]] += fluxEs(Eigen::all, 0) / fv->volumeLocal[f2c[0]];
            if (f2c[1] != FACE_2_VOL_EMPTY)
                rhs[f2c[1]] -= fluxEs(Eigen::all, 0) / fv->volumeLocal[f2c[1]];
#ifdef USE_FLUX_BALANCE_TERM
            rhs[f2c[0]] -= fluxEs(Eigen::all, 1) / fv->volumeLocal[f2c[0]];
            if (f2c[1] != FACE_2_VOL_EMPTY)
                rhs[f2c[1]] += fluxEs(Eigen::all, 2) / fv->volumeLocal[f2c[1]];
#endif

            if (faceAtr.iPhy == BoundaryType::Wall_NoSlip || faceAtr.iPhy == BoundaryType::Wall_Euler)
            {
                fluxWallSumLocal -= fluxEs(Eigen::all, 0);
            }
        }
        // quick aux: reduce the wall flux sum
        MPI_Allreduce(fluxWallSumLocal.data(), fluxWallSum.data(), fluxWallSum.size(), DNDS_MPI_REAL, MPI_SUM, u.dist->getMPI().comm);

        InsertCheck(u.dist->getMPI(), "EvaluateRHS After Flux");

#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_SOURCE_TERM
        for (index iCell = 0; iCell < jacobianCellSourceDiag.size(); iCell++) // force zero source jacobian
            jacobianCellSourceDiag[iCell].setZero();

        for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
        {
            auto &cellRecAtr = vfv->cellRecAtrLocal[iCell][0];
            auto &cellAtr = mesh->cellAtrLocal[iCell][0];
            Elem::ElementManager eCell(cellAtr.type, cellRecAtr.intScheme);
            auto cellDiBjGaussBatchElemVR = (*vfv->cellDiBjGaussBatch)[iCell];

            Eigen::Vector<real, nvarsFixedMultipy<nVars_Fixed, 2>()> sourceV(cnvars * 2); // now includes sourcejacobian diag
            sourceV.setZero();

            Elem::SummationNoOp noOp;
            bool cellOrderReduced = false;

#ifdef USE_TOTAL_REDUCED_ORDER_CELL
            eCell.Integration(
                noOp,
                [&](decltype(noOp) &finc, int ig, Elem::tPoint &p, Elem::tDiFj &DiNj)
                {
                    if (!cellOrderReduced)
                    {
                        TU ULxy =
                            cellDiBjGaussBatchElemVR.m(ig).row(0).rightCols(uRec[iCell].rows()) *
                            uRec[iCell] * IF_NOT_NOREC;
                        ULxy = CompressRecPart(u[iCell], ULxy, cellOrderReduced);
                    }
                });
#endif

            eCell.Integration(
                sourceV,
                [&](decltype(sourceV) &finc, int ig, Elem::tPoint &p, Elem::tDiFj &DiNj)
                {
                    TDiffU GradU;
                    GradU.resize(Eigen::NoChange, cnvars);
                    GradU.setZero();
                    PerformanceTimer::Instance().StartTimer(PerformanceTimer::LimiterB);
                    if constexpr (gdim == 2)
                        GradU({0, 1}, Eigen::all) =
                            cellDiBjGaussBatchElemVR.m(ig)({1, 2}, Eigen::seq(Eigen::fix<1>, Eigen::last)) *
                            uRec[iCell] * IF_NOT_NOREC; // 2d specific
                    else
                        GradU({0, 1, 2}, Eigen::all) =
                            cellDiBjGaussBatchElemVR.m(ig)({1, 2, 3}, Eigen::seq(Eigen::fix<1>, Eigen::last)) *
                            uRec[iCell] * IF_NOT_NOREC; // 3d specific

                    bool pointOrderReduced;
                    TU ULxy = u[iCell];
                    if (!cellOrderReduced)
                    {
                        // ULxy += cellDiBjGaussBatchElemVR.m(ig).row(0).rightCols(uRec[iCell].rows()) *
                        //         uRec[iCell] * IF_NOT_NOREC;
                        ULxy = CompressRecPart(ULxy,
                                               cellDiBjGaussBatchElemVR.m(ig).row(0).rightCols(uRec[iCell].rows()) *
                                                   uRec[iCell] * IF_NOT_NOREC,
                                               pointOrderReduced);
                    }
                    PerformanceTimer::Instance().EndTimer(PerformanceTimer::LimiterB);

                    // bool compressed = false;
                    // ULxy = CompressRecPart(u[iCell], ULxy, compressed); //! do not forget the mean value

                    finc.resizeLike(sourceV);
                    if constexpr (nVars_Fixed > 0)
                    {
                        finc(Eigen::seq(Eigen::fix<0>, Eigen::fix<nVars_Fixed - 1>)) =
                            source(
                                ULxy,
                                GradU,
                                iCell, ig);
                        finc(Eigen::seq(Eigen::fix<nVars_Fixed>, Eigen::fix<2 * nVars_Fixed - 1>)) =
                            sourceJacobianDiag(
                                ULxy,
                                GradU,
                                iCell, ig);
                    }
                    else
                    {
                        finc(Eigen::seq(0, cnvars - 1)) =
                            source(
                                ULxy,
                                GradU,
                                iCell, ig);
                        finc(Eigen::seq(cnvars, 2 * cnvars - 1)) =
                            sourceJacobianDiag(
                                ULxy,
                                GradU,
                                iCell, ig);
                    }

                    finc *= vfv->cellGaussJacobiDets[iCell][ig]; // don't forget this
                    if (finc.hasNaN() || (!finc.allFinite()))
                    {
                        std::cout << finc.transpose() << std::endl;
                        std::cout << ULxy.transpose() << std::endl;
                        std::cout << GradU << std::endl;
                        assert(false);
                    }
                });
            if constexpr (nVars_Fixed > 0)
            {
                rhs[iCell] += sourceV(Eigen::seq(Eigen::fix<0>, Eigen::fix<nVars_Fixed - 1>)) / fv->volumeLocal[iCell];
                jacobianCellSourceDiag[iCell] = sourceV(Eigen::seq(Eigen::fix<nVars_Fixed>, Eigen::fix<2 * nVars_Fixed - 1>)) / fv->volumeLocal[iCell];
            }
            else
            {
                rhs[iCell] += sourceV(Eigen::seq(0, cnvars - 1)) / fv->volumeLocal[iCell];
                jacobianCellSourceDiag[iCell] = sourceV(Eigen::seq(cnvars, 2 * cnvars - 1)) / fv->volumeLocal[iCell];
            }
            // if (iCell == 18195)
            // {
            //     std::cout << rhs[iCell].transpose() << std::endl;
            // }
        }
#endif
        InsertCheck(u.dist->getMPI(), "EvaluateRHS -1");
    }

    template void EulerEvaluator<NS>::EvaluateRHS(ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u,
                                                  ArrayRecV &uRec, real t);
    template void EulerEvaluator<NS_SA>::EvaluateRHS(ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u,
                                                     ArrayRecV &uRec, real t);
    template void EulerEvaluator<NS_2D>::EvaluateRHS(ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u,
                                                     ArrayRecV &uRec, real t);

    template <EulerModel model>
    void EulerEvaluator<model>::LUSGSADMatrixInit(std::vector<real> &dTau, real dt, real alphaDiag, ArrayDOFV<nVars_Fixed> &u,
                                                  int jacobianCode,
                                                  real t)
    {
        // assert(false);                                                      // TODO: to support model expanding
        // for (index iCell = 0; iCell < mesh->cell2nodeLocal.size(); iCell++) // includes ghost
        // {
        //     if (iCell < mesh->cell2nodeLocal.dist->size())
        //         jacobianCell[iCell] = Eigen::Matrix<real, 5, 5>::Identity() *
        //                               (fv->volumeLocal[iCell] / dTau[iCell] +
        //                                fv->volumeLocal[iCell] / dt);
        //     else
        //         jacobianCell[iCell].setConstant(UnInitReal);
        // }
        // for (index iFace = 0; iFace < mesh->face2nodeLocal.size(); iFace++)
        // {
        //     auto &faceRecAtr = vfv->faceRecAtrLocal[iFace][0];
        //     auto &faceAtr = mesh->faceAtrLocal[iFace][0];
        //     auto f2c = mesh->face2cellLocal[iFace];

        //     TU UL, UR, ULxy, URxy;
        //     ULxy = u[f2c[0]];

        //     Elem::tPoint unitNorm = vfv->faceNormCenter[iFace].normalized();
        //     Elem::tJacobi normBase = Elem::NormBuildLocalBaseV(unitNorm);
        //     Elem::tPoint centerRL;
        //     real volumeInverse;

        //     if (f2c[1] != FACE_2_VOL_EMPTY)
        //     {
        //         UR = URxy = u[f2c[1]];
        //         UR({1, 2, 3}) = normBase.transpose() * UR({1, 2, 3});

        //         centerRL = vfv->getCellCenter(f2c[1]) - vfv->getCellCenter(f2c[0]);
        //         volumeInverse = 0.5 / fv->volumeLocal[f2c[1]] + 0.5 / fv->volumeLocal[f2c[0]];
        //     }
        //     else if (true)
        //     {
        //         URxy = generateBoundaryValue(
        //             ULxy,
        //             unitNorm,
        //             normBase,
        //             vfv->faceCenters[iFace],
        //             t,
        //             BoundaryType(faceAtr.iPhy));

        //         centerRL = (vfv->faceCenters[iFace] - vfv->getCellCenter(f2c[0])) * 2;
        //         volumeInverse = 1.0 / fv->volumeLocal[f2c[0]];
        //     }
        //     UL = ULxy, UR = URxy;
        //     UL({1, 2, 3}) = normBase.transpose() * UL({1, 2, 3});
        //     UR({1, 2, 3}) = normBase.transpose() * UR({1, 2, 3});

        //     Eigen::Matrix<real, 5, 1> F;
        //     Eigen::Matrix<real, 5, 5> dFdUL, dFdUR;

        //     Gas::RoeFlux_IdealGas_HartenYee_AutoDiffGen(
        //         UL, UR, settings.idealGasProperty.gamma, F, dFdUL, dFdUR,
        //         [&]() {});
        //     if (F.hasNaN() || dFdUL.hasNaN() || dFdUR.hasNaN())
        //     {
        //         std::cout << "F \n"
        //                   << F << std::endl;
        //         std::cout << "GL \n"
        //                   << dFdUL << std::endl;
        //         std::cout << "GR \n"
        //                   << dFdUR << std::endl;
        //         std::cout << "UL \n"
        //                   << UL << std::endl;
        //         std::cout << "UR \n"
        //                   << UR << std::endl;
        //         assert(false);
        //     }
        //     F({1, 2, 3}) = normBase * F({1, 2, 3});
        //     dFdUL.transposeInPlace();
        //     dFdUR.transposeInPlace(); // -> dF_idUj
        //     dFdUL({1, 2, 3}, Eigen::all) = normBase * dFdUL({1, 2, 3}, Eigen::all);
        //     dFdUR({1, 2, 3}, Eigen::all) = normBase * dFdUR({1, 2, 3}, Eigen::all);
        //     dFdUL(Eigen::all, {1, 2, 3}) *= normBase.transpose();
        //     dFdUR(Eigen::all, {1, 2, 3}) *= normBase.transpose();

        //     if (jacobianCode == 2)
        //     {
        //         //*** USES vis jacobian
        //         // Elem::tPoint vGrad = unitNorm / (unitNorm.dot(centerRL));
        //         // Elem::tPoint vGrad = centerRL / centerRL.squaredNorm();
        //         Elem::tPoint vGrad = unitNorm * (fv->faceArea[iFace] * volumeInverse);
        //         TU UC = (ULxy + URxy) * 0.5;
        //         TDiffU gradU = vGrad * (URxy - ULxy).transpose();
        //         TU Fvis;
        //         TJacobianU dFvisDu, dFvisDUL, dFvisDUR;
        //         Eigen::Matrix<real, 3, nVars_Fixed>0?nVars_Fixed*4,-1> dFvisDGu;

        //         real k = settings.idealGasProperty.CpGas * settings.idealGasProperty.muGas / settings.idealGasProperty.prGas;
        //         Gas::ViscousFlux_IdealGas_N_AutoDiffGen(UC, gradU, unitNorm, false,
        //                                                 settings.idealGasProperty.gamma, settings.idealGasProperty.muGas,
        //                                                 k, settings.idealGasProperty.CpGas,
        //                                                 Fvis, dFvisDu, dFvisDGu);

        //         Eigen::Matrix<real, 5, 5> dFvisDuDiff;
        //         dFvisDuDiff({0, 1, 2, 3, 4}, {0}).setZero();
        //         dFvisDuDiff({0, 1, 2, 3, 4}, {1, 2, 3, 4}) = (vGrad.transpose() * dFvisDGu).reshaped<Eigen::ColMajor>(5, 4);

        //         dFvisDUL = (dFvisDu * 0.5 - dFvisDuDiff).transpose();
        //         dFvisDUR = (dFvisDu * 0.5 + dFvisDuDiff).transpose();
        //         // std::cout << "lamFaceVis " << lambdaFaceVis[iFace] << " dFvisDuDiff\n"
        //         //           << dFvisDuDiff << std::endl;
        //         // assert(false);
        //         // //* A
        //         // dFvisDUL(0, 0) = -0.5 * lambdaFaceVis[iFace];
        //         // dFvisDUR(0, 0) = 0.5 * lambdaFaceVis[iFace];
        //         //* B
        //         dFvisDUL -= 0.5 * lambdaFaceVis[iFace] * Eigen::Matrix<real, 5, 5>::Identity();
        //         dFvisDUR += 0.5 * lambdaFaceVis[iFace] * Eigen::Matrix<real, 5, 5>::Identity();
        //         // //* C
        //         // dFvisDUL *= 0.5;
        //         // dFvisDUR *= 0.5;
        //         // dFvisDUL -= 0.5 * 0.5 * lambdaFaceVis[iFace] * Eigen::Matrix<real, 5, 5>::Identity();
        //         // dFvisDUR += 0.5 * 0.5 * lambdaFaceVis[iFace] * Eigen::Matrix<real, 5, 5>::Identity();

        //         dFdUFace[iFace]({0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}) = dFdUL - dFvisDUL;
        //         dFdUFace[iFace]({5, 6, 7, 8, 9}, {0, 1, 2, 3, 4}) = dFdUR - dFvisDUR;

        //         jacobianFace[iFace]({0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}) =
        //             fv->faceArea[iFace] *
        //             (-dFdUL + dFvisDUL) * alphaDiag; // right: use minus version
        //         jacobianFace[iFace]({5, 6, 7, 8, 9}, {0, 1, 2, 3, 4}) =
        //             fv->faceArea[iFace] *
        //             (dFdUR - dFvisDUR) * alphaDiag; // left: uses plus version
        //         //*** USES vis jacobian
        //     }
        //     else if (jacobianCode == 1)
        //     {
        //         jacobianFace[iFace]({0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}) =
        //             fv->faceArea[iFace] *
        //             (-dFdUL - 0.5 * lambdaFaceVis[iFace] * Eigen::Matrix<real, 5, 5>::Identity()) * alphaDiag; // right: use minus version
        //         jacobianFace[iFace]({5, 6, 7, 8, 9}, {0, 1, 2, 3, 4}) =
        //             fv->faceArea[iFace] *
        //             (dFdUR - 0.5 * lambdaFaceVis[iFace] * Eigen::Matrix<real, 5, 5>::Identity()) * alphaDiag; // left: uses plus version
        //     }

        //     jacobianCell[f2c[0]] -= jacobianFace[iFace]({0, 1, 2, 3, 4}, {0, 1, 2, 3, 4});
        //     if (f2c[1] != FACE_2_VOL_EMPTY)
        //     {
        //         jacobianCell[f2c[1]] -= jacobianFace[iFace]({5, 6, 7, 8, 9}, {0, 1, 2, 3, 4});
        //     }
        //     else if (faceAtr.iPhy == BoundaryType::Farfield ||
        //              faceAtr.iPhy == BoundaryType::Special_DMRFar ||
        //              faceAtr.iPhy == BoundaryType::Special_RTFar)
        //     {
        //         // jacobianCell[f2c[0]];
        //         // nothing
        //     }
        //     else if (faceAtr.iPhy == BoundaryType::Wall_Euler)
        //     {
        //         Eigen::Matrix<real, 5, 5> jacobianRL = -jacobianFace[iFace]({5, 6, 7, 8, 9}, {0, 1, 2, 3, 4});
        //         jacobianRL(Eigen::all, {1, 2, 3}) *= normBase;
        //         jacobianRL(Eigen::all, {1}) *= -1;
        //         jacobianRL(Eigen::all, {1, 2, 3}) *= normBase.transpose();
        //         jacobianCell[f2c[0]] -= jacobianRL;
        //     }
        //     else if (faceAtr.iPhy == BoundaryType::Wall_NoSlip)
        //     {
        //         Eigen::Matrix<real, 5, 5> jacobianRL = -jacobianFace[iFace]({5, 6, 7, 8, 9}, {0, 1, 2, 3, 4});
        //         jacobianRL(Eigen::all, {1, 2, 3}) *= -1;
        //         jacobianCell[f2c[0]] -= jacobianRL;
        //     }
        //     else if (faceAtr.iPhy == BoundaryType::Wall)
        //     {
        //         std::cout << "Wall is not a proper bc" << std::endl;
        //         assert(false);
        //     }
        //     else
        //     {
        //         assert(false);
        //     }
        // }
        // for (index iCell = 0; iCell < mesh->cell2nodeLocal.size(); iCell++) // includes ghost
        // {
        //     if (iCell < mesh->cell2nodeLocal.dist->size())
        //     {
        //         jacobianCellInv[iCell] = jacobianCell[iCell].fullPivLu().inverse();

        //         if (jacobianCell[iCell].hasNaN() || jacobianCellInv[iCell].hasNaN() ||
        //             (!(jacobianCell[iCell].allFinite() && jacobianCellInv[iCell].allFinite())))
        //         {
        //             std::cout << "JCInv\n"
        //                       << jacobianCellInv[iCell] << std::endl;
        //             std::cout << "JC\n"
        //                       << jacobianCell[iCell] << std::endl;
        //             assert(false);
        //         }
        //     }
        //     else
        //         jacobianCellInv[iCell].setConstant(UnInitReal);
        // }
    }
    template void EulerEvaluator<NS>::LUSGSADMatrixInit(std::vector<real> &dTau, real dt, real alphaDiag, ArrayDOFV<nVars_Fixed> &u,
                                                        int jacobianCode,
                                                        real t);
    template void EulerEvaluator<NS_SA>::LUSGSADMatrixInit(std::vector<real> &dTau, real dt, real alphaDiag, ArrayDOFV<nVars_Fixed> &u,
                                                           int jacobianCode,
                                                           real t);
    template void EulerEvaluator<NS_2D>::LUSGSADMatrixInit(std::vector<real> &dTau, real dt, real alphaDiag, ArrayDOFV<nVars_Fixed> &u,
                                                           int jacobianCode,
                                                           real t);

    template <EulerModel model>
    void EulerEvaluator<model>::LUSGSADMatrixVec(ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &AuInc)
    {
        // assert(false); // TODO: to support model expanding
        // for (index iScan = 0; iScan < mesh->cell2nodeLocal.dist->size(); iScan++)
        // {
        //     index iCell = iScan;
        //     // iCell = (*vfv->SOR_iScan2iCell)[iCell];//TODO: add rb-sor

        //     auto &c2f = mesh->cell2faceLocal[iCell];
        //     Eigen::Vector<real, 5> uIncNewBuf;
        //     uIncNewBuf.setZero(); // norhs

        //     if (uInc[iCell].hasNaN())
        //     {
        //         std::cout << uInc[iCell] << std::endl;
        //         assert(false);
        //     }

        //     for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
        //     {
        //         index iFace = c2f[ic2f];
        //         auto &f2c = (*mesh->face2cellPair)[iFace];
        //         index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
        //         index iCellAtFace = f2c[0] == iCell ? 0 : 1;
        //         if (iCellOther != FACE_2_VOL_EMPTY)
        //         {
        //             Eigen::Matrix<real, 5, 5> jacobianOther = iCellAtFace
        //                                                           ? jacobianFace[iFace]({0, 1, 2, 3, 4}, {0, 1, 2, 3, 4})
        //                                                           : jacobianFace[iFace]({5, 6, 7, 8, 9}, {0, 1, 2, 3, 4});
        //             uIncNewBuf += jacobianOther * uInc[iCellOther](Eigen::seq(0, 4));
        //         }
        //     }
        //     AuInc[iCell](Eigen::seq(0, 4)) = jacobianCell[iCell] * uInc[iCell](Eigen::seq(0, 4)) + uIncNewBuf;
        // }
    }

    template void EulerEvaluator<NS>::LUSGSADMatrixVec(ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &AuInc);
    template void EulerEvaluator<NS_SA>::LUSGSADMatrixVec(ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &AuInc);
    template void EulerEvaluator<NS_2D>::LUSGSADMatrixVec(ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &AuInc);

    /**
     * @brief waring! rhs here is with volume, different from UpdateLUSGSADForward
     **/
    template <EulerModel model>
    void EulerEvaluator<model>::UpdateLUSGSADForward(ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew)
    {
        // assert(false); // TODO: to support model expanding
        // index nCellDist = mesh->cell2nodeLocal.dist->size();
        // for (index iScan = 0; iScan < nCellDist; iScan++)
        // {
        //     index iCell = iScan;
        //     iCell = (*vfv->SOR_iScan2iCell)[iCell]; // TODO: add rb-sor

        //     auto &c2f = mesh->cell2faceLocal[iCell];
        //     Eigen::Vector<real, 5> uIncNewBuf;
        //     uIncNewBuf = rhs[iCell](Eigen::seq(0, 4));

        //     for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
        //     {
        //         index iFace = c2f[ic2f];
        //         auto &f2c = (*mesh->face2cellPair)[iFace];
        //         index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
        //         index iCellAtFace = f2c[0] == iCell ? 0 : 1;
        //         if (iCellOther != FACE_2_VOL_EMPTY)
        //         {
        //             index iScanOther = iCellOther < nCellDist
        //                                    ? (*vfv->SOR_iCell2iScan)[iCellOther]
        //                                    : iScan + 1;
        //             if (iScanOther < iScan)
        //             {
        //                 Eigen::Matrix<real, 5, 5> jacobianOther = iCellAtFace
        //                                                               ? jacobianFace[iFace]({0, 1, 2, 3, 4}, {0, 1, 2, 3, 4})
        //                                                               : jacobianFace[iFace]({5, 6, 7, 8, 9}, {0, 1, 2, 3, 4});
        //                 uIncNewBuf -= jacobianOther * uInc[iCellOther](Eigen::seq(0, 4));
        //             }
        //         }
        //     }

        //     uIncNew[iCell](Eigen::seq(0, 4)) = jacobianCellInv[iCell] * uIncNewBuf;

        //     // fix rho increment
        //     // if (u[iCell](0) + uIncNew[iCell](0) < u[iCell](0) * 1e-5)
        //     //     uIncNew[iCell](0) = -u[iCell](0) * (1 - 1e-5);
        //     uIncNew[iCell](Eigen::seq(0, 4)) = CompressRecPart(u[iCell](Eigen::seq(0, 4)), uIncNew[iCell](Eigen::seq(0, 4))) - u[iCell](Eigen::seq(0, 4));
        // }
    }

    template void EulerEvaluator<NS>::UpdateLUSGSADForward(ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew);
    template void EulerEvaluator<NS_SA>::UpdateLUSGSADForward(ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew);
    template void EulerEvaluator<NS_2D>::UpdateLUSGSADForward(ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew);

    template <EulerModel model>
    void EulerEvaluator<model>::UpdateLUSGSADBackward(ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew)
    {
        // assert(false); // TODO: to support model expanding
        // index nCellDist = mesh->cell2nodeLocal.dist->size();
        // for (index iScan = nCellDist - 1; iScan >= 0; iScan--)
        // {
        //     index iCell = iScan;
        //     iCell = (*vfv->SOR_iScan2iCell)[iCell]; // TODO: add rb-sor

        //     auto &c2f = mesh->cell2faceLocal[iCell];
        //     Eigen::Vector<real, 5> uIncNewBuf;
        //     // uIncNewBuf = rhs[iCell];
        //     uIncNewBuf.setZero(); // Back

        //     for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
        //     {
        //         index iFace = c2f[ic2f];
        //         auto &f2c = (*mesh->face2cellPair)[iFace];
        //         index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
        //         index iCellAtFace = f2c[0] == iCell ? 0 : 1;
        //         if (iCellOther != FACE_2_VOL_EMPTY)
        //         {
        //             index iScanOther = iCellOther < nCellDist
        //                                    ? (*vfv->SOR_iCell2iScan)[iCellOther]
        //                                    : iScan + 1;
        //             if (iScanOther > iScan) // Back
        //             {
        //                 Eigen::Matrix<real, 5, 5> jacobianOther = iCellAtFace
        //                                                               ? jacobianFace[iFace]({0, 1, 2, 3, 4}, {0, 1, 2, 3, 4})
        //                                                               : jacobianFace[iFace]({5, 6, 7, 8, 9}, {0, 1, 2, 3, 4});
        //                 uIncNewBuf -= jacobianOther * uInc[iCellOther](Eigen::seq(0, 4));
        //             }
        //         }
        //     }

        //     uIncNew[iCell](Eigen::seq(0, 4)) += jacobianCellInv[iCell] * uIncNewBuf; // Back

        //     uIncNew[iCell](Eigen::seq(0, 4)) = CompressRecPart(u[iCell](Eigen::seq(0, 4)), uIncNew[iCell](Eigen::seq(0, 4))) - u[iCell](Eigen::seq(0, 4));
        // }
    }

    template void EulerEvaluator<NS>::UpdateLUSGSADBackward(ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew);
    template void EulerEvaluator<NS_SA>::UpdateLUSGSADBackward(ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew);
    template void EulerEvaluator<NS_2D>::UpdateLUSGSADBackward(ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew);

    template <EulerModel model>
    void EulerEvaluator<model>::LUSGSMatrixInit(std::vector<real> &dTau, real dt, real alphaDiag,
                                                ArrayDOFV<nVars_Fixed> &u, ArrayRecV &uRec,
                                                int jacobianCode,
                                                real t)
    {
        // TODO: for code0: flux jacobian with lambdaFace, and source jacobian with integration, only diagpart dealt with
        assert(jacobianCode == 0);
        int cnvars = nVars;
        for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
        {
            auto &cellRecAtr = vfv->cellRecAtrLocal[iCell][0];
            auto &cellAtr = mesh->cellAtrLocal[iCell][0];
            Elem::ElementManager eCell(cellAtr.type, cellRecAtr.intScheme);
            auto cellDiBjGaussBatchElemVR = (*vfv->cellDiBjGaussBatch)[iCell];
            auto &c2f = mesh->cell2faceLocal[iCell];
#ifndef DNDS_FV_EULEREVALUATOR_USE_SCALAR_JACOBIAN
            jacobianCell[iCell].setIdentity();
#endif

            // LUSGS diag part
            real fpDivisor = 1.0 / dTau[iCell] + 1.0 / dt; //!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                fpDivisor += (0.5 * alphaDiag) * fv->faceArea[iFace] * lambdaFace[iFace] / fv->volumeLocal[iCell];
            }
#ifndef DNDS_FV_EULEREVALUATOR_USE_SCALAR_JACOBIAN
            jacobianCell[iCell] *= fpDivisor; //! all passive vars use same diag for flux part
#else
            jacobianCell_Scalar[iCell] = fpDivisor;
#endif
            // std::cout << fpDivisor << std::endl;

            // jacobian diag

#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_SOURCE_TERM
            jacobianCell[iCell] += alphaDiag * jacobianCellSourceDiag[iCell].asDiagonal();
#endif
            //! assuming diagonal here!
#ifndef DNDS_FV_EULEREVALUATOR_USE_SCALAR_JACOBIAN
            jacobianCellInv[iCell] = jacobianCell[iCell].diagonal().array().inverse().matrix().asDiagonal();
#else
            jacobianCellInv_Scalar[iCell] = 1. / fpDivisor;
#endif
            // jacobianCellInv[iCell] = jacobianCell[iCell].partialPivLu().inverse();

            // std::cout << "jacobian Diag\n"
            //           << jacobianCell[iCell] << std::endl;
            // std::cout << dTau[iCell] << "\n";
        }
        // exit(-1);
    }
    template void EulerEvaluator<NS>::LUSGSMatrixInit(std::vector<real> &dTau, real dt, real alphaDiag,
                                                      ArrayDOFV<nVars_Fixed> &u, ArrayRecV &uRec,
                                                      int jacobianCode,
                                                      real t);
    template void EulerEvaluator<NS_SA>::LUSGSMatrixInit(std::vector<real> &dTau, real dt, real alphaDiag,
                                                         ArrayDOFV<nVars_Fixed> &u, ArrayRecV &uRec,
                                                         int jacobianCode,
                                                         real t);
    template void EulerEvaluator<NS_2D>::LUSGSMatrixInit(std::vector<real> &dTau, real dt, real alphaDiag,
                                                         ArrayDOFV<nVars_Fixed> &u, ArrayRecV &uRec,
                                                         int jacobianCode,
                                                         real t);

    template <EulerModel model>
    void EulerEvaluator<model>::LUSGSMatrixVec(real alphaDiag, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &AuInc)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        InsertCheck(u.dist->getMPI(), "LUSGSMatrixVec 1");
        int cnvars = nVars;
        for (index iScan = 0; iScan < mesh->cell2nodeLocal.dist->size(); iScan++)
        {
            index iCell = iScan;
            // iCell = (*vfv->SOR_iScan2iCell)[iCell];//TODO: add rb-sor

            auto &c2f = mesh->cell2faceLocal[iCell];
            TU uIncNewBuf(cnvars);
            uIncNewBuf.setZero(); // norhs
            auto uINCi = uInc[iCell];

            if (uINCi.hasNaN())
            {
                std::cout << uINCi << std::endl;
                assert(false);
            }

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto &f2c = (*mesh->face2cellPair)[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                if (iCellOther != FACE_2_VOL_EMPTY)
                {

                    if (true)
                    {
                        auto uINCj = uInc[iCellOther];
                        auto uj = u[iCellOther];
                        TU fInc;
                        {
                            TVec unitNorm = vfv->faceNormCenter[iFace](Seq012).normalized() *
                                            (iCellAtFace ? -1 : 1); // faces out

                            // fInc = fluxJacobian0_Right(
                            //            u[iCellOther],
                            //            unitNorm,
                            //            BoundaryType::Inner) *
                            //        uInc[iCellOther]; //! always inner here
                            fInc = fluxJacobian0_Right_Times_du(
                                uj,
                                unitNorm,
                                BoundaryType::Inner, uINCj, lambdaFace[iFace], lambdaFaceC[iFace]); //! always inner here
                        }

                        uIncNewBuf -= (0.5 * alphaDiag) * fv->faceArea[iFace] / fv->volumeLocal[iCell] *
                                      (fInc);
                        if (uIncNewBuf.hasNaN() || (!uIncNewBuf.allFinite()))
                        {
                            std::cout
                                << fInc.transpose() << std::endl
                                << uInc[iCellOther].transpose() << std::endl;
                            assert(!(uIncNewBuf.hasNaN() || (!uIncNewBuf.allFinite())));
                        }
                    }
                }
            }
            // uIncNewBuf /= fpDivisor;
            // uIncNew[iCell] = uIncNewBuf;
            auto AuIncI = AuInc[iCell];
#ifndef DNDS_FV_EULEREVALUATOR_USE_SCALAR_JACOBIAN
            AuIncI = jacobianCell[iCell] * uInc[iCell] - uIncNewBuf;
#else
            AuIncI = jacobianCell_Scalar[iCell] * uInc[iCell] - uIncNewBuf;
#endif
            if (AuIncI.hasNaN())
            {
                std::cout << AuIncI.transpose() << std::endl
                          << uINCi.transpose() << std::endl
                          << u[iCell].transpose() << std::endl
                          << jacobianCell[iCell] << std::endl
                          << iCell << std::endl;
                assert(!AuInc[iCell].hasNaN());
            }
        }
        InsertCheck(u.dist->getMPI(), "LUSGSMatrixVec -1");
    }

    template void EulerEvaluator<NS>::LUSGSMatrixVec(real alphaDiag, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &AuInc);
    template void EulerEvaluator<NS_SA>::LUSGSMatrixVec(real alphaDiag, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &AuInc);
    template void EulerEvaluator<NS_2D>::LUSGSMatrixVec(real alphaDiag, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &AuInc);

    template <EulerModel model>
    void EulerEvaluator<model>::UpdateLUSGSForward(real alphaDiag,
                                                   ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        InsertCheck(u.dist->getMPI(), "UpdateLUSGSForward 1");
        int cnvars = nVars;
        index nCellDist = mesh->cell2nodeLocal.dist->size();
        for (index iScan = 0; iScan < nCellDist; iScan++)
        {
            index iCell = iScan;
            iCell = (*vfv->SOR_iScan2iCell)[iCell]; // TODO: add rb-sor

            auto &c2f = mesh->cell2faceLocal[iCell];
            TU uIncNewBuf(nVars);
            auto RHSI = rhs[iCell];
            // std::cout << rhs[iCell](0) << std::endl;
            uIncNewBuf = RHSI;

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto &f2c = (*mesh->face2cellPair)[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                if (iCellOther != FACE_2_VOL_EMPTY)
                {

                    index iScanOther = iCellOther < nCellDist
                                           ? (*vfv->SOR_iCell2iScan)[iCellOther]
                                           : iScan + 1;
                    if (iScanOther < iScan)
                    {
                        TU fInc;
                        auto uINCj = uInc[iCellOther];

                        {
                            TVec unitNorm = vfv->faceNormCenter[iFace](Seq012).normalized() *
                                            (iCellAtFace ? -1 : 1); // faces out

                            // fInc = fluxJacobian0_Right(
                            //            u[iCellOther],
                            //            unitNorm,
                            //            BoundaryType::Inner) *
                            //        uInc[iCellOther]; //! always inner here

                            // uINCj << 0.01997442663, -0.1624173545, -0.1349337526, 0, 2.22936795, -0.03193140561;
                            fInc = fluxJacobian0_Right_Times_du(
                                u[iCellOther],
                                unitNorm,
                                BoundaryType::Inner, uINCj, lambdaFace[iFace], lambdaFaceC[iFace]); //! always inner here
                            // std::cout << uINCj.transpose() << " " << fInc.transpose() << std::endl;
                            // if (uInc[iCellOther](0) > 1e-5 && iCellOther == 10756)
                            // {

                            //     std::cout << iCellOther << " " << alphaDiag << std::endl;
                            //     std::cout << std::setprecision(10);
                            //     std::cout << u[iCellOther].transpose() << std::endl;
                            //     std::cout << uInc[iCellOther].transpose() << std::endl;
                            //     std::cout << (0.5 * alphaDiag) * fv->faceArea[iFace] / fv->volumeLocal[iCell] *
                            //                      (fInc).transpose()
                            //               << std::endl;
                            //     std::cout << vfv->cellBaries[iCell].transpose() << std::endl;
                            //     exit(1);
                            // }
                        }

                        uIncNewBuf -= (0.5 * alphaDiag) * fv->faceArea[iFace] / fv->volumeLocal[iCell] *
                                      (fInc);
                        // std::cout << "Forward " << iCell << "-" << iCellOther << " " << (0.5 * alphaDiag) * fv->faceArea[iFace] / fv->volumeLocal[iCell] * (fInc(0));
                        // std::cout << std::endl;
                        if (uIncNewBuf.hasNaN() || (!uIncNewBuf.allFinite()))
                        {
                            std::cout << RHSI.transpose() << std::endl
                                      << fInc.transpose() << std::endl
                                      << uINCj.transpose() << std::endl;
                            assert(!(uIncNewBuf.hasNaN() || (!uIncNewBuf.allFinite())));
                        }
                        if (iCell == 10756)
                        {
                            std::cout << std::setprecision(16) << "??? dUother " << (0.5 * alphaDiag) * fv->faceArea[iFace] / fv->volumeLocal[iCell] * fInc.transpose()
                                      << std::endl;
                        }
                    }
                }
            }
            auto uIncNewI = uIncNew[iCell];
#ifndef DNDS_FV_EULEREVALUATOR_USE_SCALAR_JACOBIAN
            uIncNewI = jacobianCellInv[iCell] * uIncNewBuf;
#else
            uIncNewI = jacobianCellInv_Scalar[iCell] * uIncNewBuf;
#endif
            // std::cout << "Forward " << iCell << "-- " << uIncNewI(0) << std::endl;
            // if (iCell == 10756)
            // {
            //     std::cout << std::setprecision(10);
            //     std::cout << uIncNewI.transpose() << std::endl;
            //     std::cout << jacobianCellInv[iCell] << std::endl;
            //     std::cout << uIncNewBuf.transpose() << std::endl;
            //     std::cout << RHSI.z() << std::endl;
            // }
            if (uIncNewI.hasNaN())
            {
                std::cout << uIncNewI.transpose() << std::endl
                          << jacobianCellInv[iCell] << std::endl
                          << iCell << std::endl;
                assert(!uIncNew[iCell].hasNaN());
            }

            // fix rho increment
            // if (u[iCell](0) + uIncNew[iCell](0) < u[iCell](0) * 1e-5)
            //     uIncNew[iCell](0) = -u[iCell](0) * (1 - 1e-5);
            // uIncNewI = CompressInc(u[iCell], uIncNewI, RHSI); //! disabled for trusting the following Backward fix

            // std::cout << "JCI\n";
            // std::cout << jacobianCellInv[iCell] << std::endl;

            // if (iCell == 0)
            //     assert(false);
            // uIncNewI(0) = 0;
            // u[iCell](0) = 1 + iCell * 1e-8;

            if (iCell == 20636 || iCell == 10756)
                std::cout << std::setprecision(16) << "??? " << uIncNewI(0) << " RHS " << RHSI(0) << " dF " << uIncNewBuf(0) - RHSI(0) << " IJ " << jacobianCellInv[iCell](0, 0) << std::endl;
        }
        InsertCheck(u.dist->getMPI(), "UpdateLUSGSForward -1");
        // exit(-1);
    }

    template void EulerEvaluator<NS>::UpdateLUSGSForward(real alphaDiag,
                                                         ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew);
    template void EulerEvaluator<NS_SA>::UpdateLUSGSForward(real alphaDiag,
                                                            ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew);
    template void EulerEvaluator<NS_2D>::UpdateLUSGSForward(real alphaDiag,
                                                            ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew);

    template <EulerModel model>
    void EulerEvaluator<model>::UpdateLUSGSBackward(real alphaDiag,
                                                    ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        InsertCheck(u.dist->getMPI(), "UpdateLUSGSBackward 1");
        int cnvars = nVars;
        index nCellDist = mesh->cell2nodeLocal.dist->size();
        for (index iScan = nCellDist - 1; iScan >= 0; iScan--)
        {
            index iCell = iScan;
            iCell = (*vfv->SOR_iScan2iCell)[iCell];

            auto &c2f = mesh->cell2faceLocal[iCell];
            TU uIncNewBuf(cnvars);
            uIncNewBuf.setZero(); // backward
            auto RHSI = rhs[iCell];

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto &f2c = (*mesh->face2cellPair)[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                if (iCellOther != FACE_2_VOL_EMPTY)
                {

                    index iScanOther = iCellOther < nCellDist
                                           ? (*vfv->SOR_iCell2iScan)[iCellOther]
                                           : iScan + 1;
                    if (iScanOther > iScan) // backward
                    {
                        TU fInc;

                        {
                            TVec unitNorm = vfv->faceNormCenter[iFace](Seq012).normalized() *
                                            (iCellAtFace ? -1 : 1); // faces out

                            // fInc = fluxJacobian0_Right(
                            //            u[iCellOther],
                            //            unitNorm,
                            //            BoundaryType::Inner) *
                            //        uInc[iCellOther]; //! always inner here
                            fInc = fluxJacobian0_Right_Times_du(
                                u[iCellOther],
                                unitNorm,
                                BoundaryType::Inner, uInc[iCellOther], lambdaFace[iFace], lambdaFaceC[iFace]); //! always inner here
                        }

                        uIncNewBuf -= (0.5 * alphaDiag) * fv->faceArea[iFace] / fv->volumeLocal[iCell] *
                                      (fInc);
                        // std::cout << "BackWard " << iCell << "-" << iCellOther << " " << (0.5 * alphaDiag) * fv->faceArea[iFace] / fv->volumeLocal[iCell] * (fInc(0));
                        // std::cout << std::endl;
                    }
                }
            }
            auto uIncNewI = uIncNew[iCell];
#ifndef DNDS_FV_EULEREVALUATOR_USE_SCALAR_JACOBIAN
            uIncNewI += jacobianCellInv[iCell] * uIncNewBuf; // backward
#else
            uIncNewI += jacobianCellInv_Scalar[iCell] * uIncNewBuf; // backward
#endif
            if (iCell == 20636 || iCell == 10756)
                std::cout << std::setprecision(16) << "!!! " << uIncNewI(0) << std::endl;
            // std::cout << "BackWard " << iCell << "-- " << uIncNewI(0) << std::endl;

            // fix rho increment
            // if (u[iCell](0) + uIncNew[iCell](0) < u[iCell](0) * 1e-5)
            //     uIncNew[iCell](0) = -u[iCell](0) * (1 - 1e-5);
            // uIncNewI = CompressInc(u[iCell], uIncNewI, RHSI);
        }
        InsertCheck(u.dist->getMPI(), "UpdateLUSGSBackward -1");
    }

    template void EulerEvaluator<NS>::UpdateLUSGSBackward(real alphaDiag,
                                                          ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew);
    template void EulerEvaluator<NS_SA>::UpdateLUSGSBackward(real alphaDiag,
                                                             ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew);
    template void EulerEvaluator<NS_2D>::UpdateLUSGSBackward(real alphaDiag,
                                                             ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew);

    template <EulerModel model>
    void EulerEvaluator<model>::UpdateSGS(real alphaDiag,
                                          ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew, bool ifForward)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        int cnvars = nVars;
        for (index iScan = 0; iScan < mesh->cell2nodeLocal.dist->size(); iScan++)
        {
            index iCell;
            if (ifForward)
                iCell = iScan;
            else
                iCell = mesh->cell2nodeLocal.dist->size() - 1 - iScan;
            auto &c2f = mesh->cell2faceLocal[iCell];
            TU uIncNewBuf;
            // uIncNewBuf.setZero(); // backward
            uIncNewBuf = rhs[iCell]; // full

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto &f2c = (*mesh->face2cellPair)[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                if (iCellOther != FACE_2_VOL_EMPTY)
                {
                    // if (true) // full
                    if ((ifForward && iCellOther < iCell) || ((!ifForward) && iCellOther > iCell))
                    {
                        TU fInc;

                        {
                            TVec unitNorm = vfv->faceNormCenter[iFace](Seq012).normalized() *
                                            (iCellAtFace ? -1 : 1); // faces out

                            // fInc = fluxJacobian0_Right(
                            //            u[iCellOther],
                            //            unitNorm,
                            //            BoundaryType::Inner) *
                            //        uInc[iCellOther]; //! always inner here
                            fInc = fluxJacobian0_Right_Times_du(
                                u[iCellOther],
                                unitNorm,
                                BoundaryType::Inner, uInc[iCellOther], lambdaFace[iFace], lambdaFaceC[iFace]); //! always inner here
                        }

                        uIncNewBuf -= (0.5 * alphaDiag) * fv->faceArea[iFace] / fv->volumeLocal[iCell] *
                                      (fInc);
                    }
                }
            }
            real relax = 1;
            uIncNew[iCell] = jacobianCellInv[iCell] * uIncNewBuf * relax + uInc[iCell] * (1 - relax); // full

            // fix rho increment
            // if (u[iCell](0) + uIncNew[iCell](0) < u[iCell](0) * 1e-5)
            //     uIncNew[iCell](0) = -u[iCell](0) * (1 - 1e-5);
        }
    }

    template void EulerEvaluator<NS>::UpdateSGS(real alphaDiag,
                                                ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew, bool ifForward);
    template void EulerEvaluator<NS_SA>::UpdateSGS(real alphaDiag,
                                                   ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew, bool ifForward);
    template void EulerEvaluator<NS_2D>::UpdateSGS(real alphaDiag,
                                                   ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew, bool ifForward);

    template <EulerModel model>
    void EulerEvaluator<model>::FixUMaxFilter(ArrayDOFV<nVars_Fixed> &u)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        // TODO: make spacial filter jacobian
        return; // ! nofix shortcut
        real scaleRhoCutoff = 0.0001;
        real scaleEInternalCutOff = 0.0001;
        real rhoMax = 0.0;
        real rhoMeanU = 0.0;
        real rhoMeanL = 0.0;
        for (index iCell = 0; iCell < u.size(); iCell++)
        {
            rhoMax = std::max(u[iCell](0), rhoMax);
            rhoMeanU += u[iCell](0);
            rhoMeanL += 1;
        }
        real rhoMaxR, rhoMeanUR, rhoMeanLR;
        MPI_Allreduce(&rhoMax, &rhoMaxR, 1, DNDS_MPI_REAL, MPI_MAX, u.dist->getMPI().comm);
        MPI_Allreduce(&rhoMeanU, &rhoMeanUR, 1, DNDS_MPI_REAL, MPI_SUM, u.dist->getMPI().comm);
        MPI_Allreduce(&rhoMeanL, &rhoMeanLR, 1, DNDS_MPI_REAL, MPI_SUM, u.dist->getMPI().comm);
        real eMax = 0.0;
        real eMeanU = 0.0;
        real eMeanL = 0.0;
        real rhoMean = rhoMeanUR / rhoMeanLR;
        real rhoTC = scaleRhoCutoff * rhoMean;
        real rhoTCSqr = rhoTC * rhoTC;
        for (index iCell = 0; iCell < u.dist->size(); iCell++)
        {
            real rhoT = rhoTC;
            real rhoTSqr = rhoTCSqr;

            auto c2f = mesh->cell2faceLocal[iCell];
            real rhoMU = 0.0;
            real rhoML = 0.0;
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto f2c = mesh->face2cellLocal[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                if (iCellOther != FACE_2_VOL_EMPTY)
                {
                    rhoMU += u[iCellOther](0);
                    rhoML += 1;
                }
            }
            rhoT = scaleRhoCutoff * rhoMU / rhoML;
            rhoT = rhoTC; //! using global
            rhoTSqr = rhoT * rhoT;

            real uScale = 1;
            real rhoOld = u[iCell](0);
            if (rhoOld <= 0)
                u[iCell](0) = rhoT * 0.5, uScale = 0;
            else if (rhoOld <= rhoT)
            {
                u[iCell](0) = (rhoOld * rhoOld + rhoTSqr) / (2 * rhoT);
                uScale = u[iCell](0) / rhoOld;
            }
            u[iCell](Seq123) *= uScale;
            real e = u[iCell](I4) - 0.5 * u[iCell](Seq123).squaredNorm() / (u[iCell](0) + verySmallReal);
            eMax = std::max(e, eMax);
            eMeanU += e;
            eMeanL += 1;
        }
        real eMaxR, eMeanUR, eMeanLR;
        MPI_Allreduce(&eMax, &eMaxR, 1, DNDS_MPI_REAL, MPI_MAX, u.dist->getMPI().comm);
        MPI_Allreduce(&eMeanU, &eMeanUR, 1, DNDS_MPI_REAL, MPI_SUM, u.dist->getMPI().comm);
        MPI_Allreduce(&eMeanL, &eMeanLR, 1, DNDS_MPI_REAL, MPI_SUM, u.dist->getMPI().comm);
        real eMean = eMeanUR / eMeanLR;
        real eTC = scaleEInternalCutOff * eMean;
        real eTSqrC = eTC * eTC;
        u.StartPersistentPullClean();
        u.WaitPersistentPullClean();
        for (index iCell = 0; iCell < u.dist->size(); iCell++)
        {
            real eT = eTC;
            real eTSqr = eTSqrC;

            auto c2f = mesh->cell2faceLocal[iCell];
            real eMU = 0.0;
            real eML = 0.0;
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto f2c = mesh->face2cellLocal[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                if (iCellOther != FACE_2_VOL_EMPTY)
                {
                    real EkOther = 0.5 * u[iCellOther](Seq123).squaredNorm() / (u[iCellOther](0) + verySmallReal);
                    eMU += u[iCellOther](I4) - EkOther;
                    eML += 1;
                }
            }
            eT = scaleEInternalCutOff * eMU / eML;
            eT = eTC; //! using global
            eTSqr = eT * eT;

            real Ek = 0.5 * u[iCell](Seq123).squaredNorm() / (u[iCell](0) + verySmallReal);
            real e = u[iCell](I4) - Ek;
            if (e <= 0)
                e = eT * 0.5;
            else if (e <= eT)
                e = (e * e + eTSqr) / (2 * eT);
            // eNew + Ek = e + Eknew
            real EkNew = Ek - e + (u[iCell](I4) - Ek);
            if (EkNew > 0)
            {
                real uScale = sqrt(EkNew / Ek);
                u[iCell](Seq123) *= uScale;
            }
            else
            {
                u[iCell](Seq123) *= 0;
                u[iCell](I4) -= EkNew;
            }
            // u[iCell](4) =Ek + e;
        }
    }

    template void EulerEvaluator<NS>::FixUMaxFilter(ArrayDOFV<nVars_Fixed> &u);
    template void EulerEvaluator<NS_SA>::FixUMaxFilter(ArrayDOFV<nVars_Fixed> &u);
    template void EulerEvaluator<NS_2D>::FixUMaxFilter(ArrayDOFV<nVars_Fixed> &u);

    template <EulerModel model>
    void EulerEvaluator<model>::EvaluateResidual(Eigen::Vector<real, -1> &res, ArrayDOFV<nVars_Fixed> &rhs, index P, bool volWise)
    {
        res.resize(nVars);
        if (P < 3)
        {
            TU resc;
            resc.resizeLike(rhs[0]);
            resc.setZero();

            for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
                if (rhs[iCell].hasNaN() || (!rhs[iCell].allFinite()))
                {
                    std::cout << rhs[iCell] << std::endl;
                    assert(false);
                }
                if (volWise)
                    resc += rhs[iCell].array().abs().pow(P).matrix() * fv->volumeLocal[iCell];
                else
                    resc += rhs[iCell].array().abs().pow(P).matrix();
            }
            MPI_Allreduce(resc.data(), res.data(), res.size(), DNDS_MPI_REAL, MPI_SUM, rhs.dist->getMPI().comm);
            res = res.array().pow(1.0 / P).matrix();
            // std::cout << res << std::endl;
        }
        else
        {
            TU resc;
            resc.resizeLike(rhs[0]);
            resc.setZero();
            for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
                resc = resc.array().max(rhs[iCell].array().abs()).matrix();
            MPI_Allreduce(resc.data(), res.data(), res.size(), DNDS_MPI_REAL, MPI_MAX, rhs.dist->getMPI().comm);
        }
    }

    template void EulerEvaluator<NS>::EvaluateResidual(Eigen::Vector<real, -1> &res, ArrayDOFV<nVars_Fixed> &rhs, index P, bool volWise);
    template void EulerEvaluator<NS_SA>::EvaluateResidual(Eigen::Vector<real, -1> &res, ArrayDOFV<nVars_Fixed> &rhs, index P, bool volWise);
    template void EulerEvaluator<NS_2D>::EvaluateResidual(Eigen::Vector<real, -1> &res, ArrayDOFV<nVars_Fixed> &rhs, index P, bool volWise);
}
