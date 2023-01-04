#include "DNDS_FV_EulerEvaluator.hpp"

#include "DNDS_GasGen.hpp"
#include "DNDS_HardEigen.h"

namespace DNDS
{

    // Eigen::Vector<real, -1> EulerEvaluator::CompressRecPart(
    //     const Eigen::Vector<real, -1> &umean,
    //     const Eigen::Vector<real, -1> &uRecInc)

    //! evaluates dt and facial spectral radius
    void EulerEvaluator::EvaluateDt(std::vector<real> &dt,
                                    ArrayDOFV &u,
                                    real CFL, real &dtMinall, real MaxDt,
                                    bool UseLocaldt)
    {
        InsertCheck(u.dist->getMPI(), "EvaluateDt 1");
        for (auto &i : lambdaCell)
            i = 0.0;

        for (index iFace = 0; iFace < mesh->face2nodeLocal.size(); iFace++)
        {
            auto f2c = mesh->face2cellLocal[iFace];
            auto faceDiBjCenterBatchElemVR = (*vfv->faceDiBjCenterBatch)[iFace];
            Elem::tPoint unitNorm = vfv->faceNormCenter[iFace].normalized();

            index iCellL = f2c[0];
            Eigen::Vector<real, -1> uMean = u[iCellL];
            real pL, asqrL, HL, pR, asqrR, HR;
            Gas::tVec vL = u[iCellL]({1, 2, 3}) / u[iCellL](0);
            Gas::tVec vR = vL;
            Gas::IdealGasThermal(u[iCellL](4), u[iCellL](0), vL.squaredNorm(),
                                 settings.idealGasProperty.gamma,
                                 pL, asqrL, HL);
            pR = pL, HR = HL, asqrR = asqrL;
            if (f2c[1] != FACE_2_VOL_EMPTY)
            {
                uMean = (uMean + u[f2c[1]]) * 0.5;
                vR = u[f2c[1]]({1, 2, 3}) / u[f2c[1]](0);
                Gas::IdealGasThermal(u[f2c[1]](4), u[f2c[1]](0), vR.squaredNorm(),
                                     settings.idealGasProperty.gamma,
                                     pR, asqrR, HR);
            }
            assert(uMean(0) > 0);
            Gas::tVec veloMean = (uMean({1, 2, 3}).array() / uMean(0)).matrix();
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
            Gas::IdealGasThermal(uMean(4), uMean(0), veloMean.squaredNorm(),
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
            if (model == NS_SA)
            {
                real cnu1 = 7.1;
                real Chi = uMean(5) * muRef / muf;
                if (Chi < 10)
                    Chi = 0.05 * std::log(1 + std::exp(20 * Chi));
                real Chi3 = std::pow(Chi, 3);
                real fnu1 = Chi3 / (Chi3 + std::pow(cnu1, 3));
                muf *= (1 + Chi * fnu1);
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
            lambdaCell[iCellL] += lamFace + 2 * lamVis * areaSqr / fv->volumeLocal[iCellL];
            if (f2c[1] != FACE_2_VOL_EMPTY) // can't be non local
                lambdaCell[f2c[1]] += lamFace + 2 * lamVis * areaSqr / fv->volumeLocal[f2c[1]],
                    volR = fv->volumeLocal[f2c[1]];

            lambdaFace[iFace] = lambdaConvection + lamVis * area * (1. / fv->volumeLocal[iCellL] + 1. / volR);
            lambdaFaceVis[iFace] = lamVis * area * (1. / fv->volumeLocal[iCellL] + 1. / volR);

            deltaLambdaFace[iFace] = std::abs((vR - vL).dot(unitNorm)) + std::sqrt(std::abs(asqrR - asqrL)) * 0.7071;
        }
        real dtMin = veryLargeReal;
        for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
        {
            // std::cout << fv->volumeLocal[iCell] << " " << (lambdaCell[iCell]) << " " << CFL << std::endl;
            // exit(0);
            dt[iCell] = std::min(CFL * fv->volumeLocal[iCell] / (lambdaCell[iCell] + 1e-100), MaxDt);
            dtMin = std::min(dtMin, dt[iCell]);
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

#define IF_NOT_NOREC (1)
    void EulerEvaluator::EvaluateRHS(ArrayDOFV &rhs, ArrayDOFV &u,
                                     ArrayRecV &uRec, real t)
    {
        InsertCheck(u.dist->getMPI(), "EvaluateRHS 1");
        int cnvars = nVars;
        Setting::RiemannSolverType rsType = settings.rsType;
        for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
        {
            rhs[iCell].setZero();
        }
        Eigen::Vector<real, -1> fluxWallSumLocal;
        fluxWallSumLocal.setZero(cnvars);
        fluxWallSum.setZero(cnvars);

        for (index iFace = 0; iFace < mesh->face2nodeLocal.size(); iFace++)
        {
            auto &faceRecAtr = vfv->faceRecAtrLocal[iFace][0];
            auto &faceAtr = mesh->faceAtrLocal[iFace][0];
            auto f2c = mesh->face2cellLocal[iFace];
            Elem::ElementManager eFace(faceAtr.type, faceRecAtr.intScheme);
            Eigen::Vector<real, -1> flux(cnvars);
            flux.setZero();
            auto faceDiBjGaussBatchElemVR = (*vfv->faceDiBjGaussBatch)[iFace];

            auto f2n = mesh->face2nodeLocal[iFace];
            Eigen::MatrixXd coords;
            mesh->LoadCoords(f2n, coords);

            eFace.Integration(
                flux,
                [&](decltype(flux) &finc, int ig, Elem::tPoint &p, Elem::tDiFj &DiNj)
                {
                    int nDiff = vfv->faceWeights->operator[](iFace).size();
                    Elem::tPoint unitNorm = vfv->faceNorms[iFace][ig].normalized();
                    Elem::tJacobi normBase = Elem::NormBuildLocalBaseV(unitNorm);

                    Eigen::Vector<real, -1> ULxy =
                        faceDiBjGaussBatchElemVR.m(ig * 2 + 0).row(0).rightCols(uRec[f2c[0]].rows()) *
                        uRec[f2c[0]] * IF_NOT_NOREC;
                    // UL += u[f2c[0]]; //! do not forget the mean value
                    ULxy = CompressRecPart(u[f2c[0]], ULxy);

                    Eigen::Vector<real, -1> URxy;

                    Eigen::Matrix<real, 3, -1> GradULxy, GradURxy;
                    GradULxy.resize(Eigen::NoChange, cnvars);
                    GradURxy.resize(Eigen::NoChange, cnvars);
                    GradULxy.setZero(), GradURxy.setZero();

                    GradULxy({0, 1}, Eigen::all) =
                        faceDiBjGaussBatchElemVR.m(ig * 2 + 0)({1, 2}, Eigen::seq(1, Eigen::last)) *
                        uRec[f2c[0]] * IF_NOT_NOREC; // ! 2d here

                    real minVol = fv->volumeLocal[f2c[0]];
                    // InsertCheck(u.dist->getMPI(), "RHS inner 2");

                    if (f2c[1] != FACE_2_VOL_EMPTY)
                    {
                        URxy =
                            faceDiBjGaussBatchElemVR.m(ig * 2 + 1).row(0).rightCols(uRec[f2c[1]].rows()) *
                            uRec[f2c[1]] * IF_NOT_NOREC;
                        // UR += u[f2c[1]];
                        URxy = CompressRecPart(u[f2c[1]], URxy);

                        GradURxy({0, 1}, Eigen::all) =
                            faceDiBjGaussBatchElemVR.m(ig * 2 + 1)({1, 2}, Eigen::seq(1, Eigen::last)) *
                            uRec[f2c[1]] * IF_NOT_NOREC; // ! 2d here

                        minVol = std::min(minVol, fv->volumeLocal[f2c[1]]);
                    }
                    else if (true)
                    {
                        URxy = generateBoundaryValue(
                            ULxy,
                            unitNorm,
                            normBase,
                            vfv->faceCenters[iFace],
                            t,
                            BoundaryType(faceAtr.iPhy));
                        GradURxy = GradULxy;
                    }
                    // UR = URxy;
                    // UL = ULxy;
                    // UR({1, 2, 3}) = normBase.transpose() * UR({1, 2, 3});
                    // UL({1, 2, 3}) = normBase.transpose() * UL({1, 2, 3});

                    real distGRP = minVol / fv->faceArea[iFace] * 2;
                    // real distGRP = (vfv->cellBaries[f2c[0]] -
                    //                 (f2c[1] != FACE_2_VOL_EMPTY
                    //                      ? vfv->cellBaries[f2c[1]]
                    //                      : 2 * vfv->faceCenters[iFace] - vfv->cellBaries[f2c[0]]))
                    //                    .norm();
                    // InsertCheck(u.dist->getMPI(), "RHS inner 1");
                    Eigen::VectorXd UMeanXy = 0.5 * (ULxy + URxy);
                    Eigen::Matrix<real, 3, -1> GradUMeanXy = (GradURxy + GradULxy) * 0.5 +
                                                             (1.0 / distGRP) *
                                                                 (unitNorm * (URxy - ULxy).transpose());
                    finc = fluxFace(
                        ULxy,
                        URxy,
                        GradUMeanXy,
                        unitNorm,
                        normBase,
                        BoundaryType(faceAtr.iPhy),
                        rsType,
                        iFace, ig);

                    finc *= vfv->faceNorms[iFace][ig].norm(); // don't forget this
                });

            rhs[f2c[0]] += flux / fv->volumeLocal[f2c[0]];
            if (f2c[1] != FACE_2_VOL_EMPTY)
                rhs[f2c[1]] -= flux / fv->volumeLocal[f2c[1]];

            if (faceAtr.iPhy == BoundaryType::Wall_NoSlip || faceAtr.iPhy == BoundaryType::Wall_Euler)
            {
                fluxWallSumLocal -= flux;
            }
        }
        // quick aux: reduce the wall flux sum
        MPI_Allreduce(fluxWallSumLocal.data(), fluxWallSum.data(), fluxWallSum.size(), DNDS_MPI_REAL, MPI_SUM, u.dist->getMPI().comm);

        InsertCheck(u.dist->getMPI(), "EvaluateRHS After Flux");

        for (index iCell = 0; iCell < jacobianCellSourceDiag.size(); iCell++) // force zero source jacobian
            jacobianCellSourceDiag[iCell].setZero();

        for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
        {
            auto &cellRecAtr = vfv->cellRecAtrLocal[iCell][0];
            auto &cellAtr = mesh->cellAtrLocal[iCell][0];
            Elem::ElementManager eCell(cellAtr.type, cellRecAtr.intScheme);
            auto cellDiBjGaussBatchElemVR = (*vfv->cellDiBjGaussBatch)[iCell];

            Eigen::Vector<real, -1> sourceV(cnvars * 2); // now includes sourcejacobian diag
            sourceV.setZero();

            eCell.Integration(
                sourceV,
                [&](decltype(sourceV) &finc, int ig, Elem::tPoint &p, Elem::tDiFj &DiNj)
                {
                    Eigen::Matrix<real, 3, -1> GradU;
                    GradU.resize(Eigen::NoChange, cnvars);
                    GradU.setZero();
                    GradU({0, 1}, Eigen::all) =
                        cellDiBjGaussBatchElemVR.m(ig)({1, 2}, Eigen::seq(1, Eigen::last)) *
                        uRec[iCell] * IF_NOT_NOREC; //! 2d specific

                    Eigen::Vector<real, -1> ULxy =
                        cellDiBjGaussBatchElemVR.m(ig).row(0).rightCols(uRec[iCell].rows()) *
                        uRec[iCell] * IF_NOT_NOREC;

                    ULxy = CompressRecPart(u[iCell], ULxy); //! do not forget the mean value

                    finc.resizeLike(sourceV);
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

                    finc *= vfv->cellGaussJacobiDets[iCell][ig]; // don't forget this
                    if (finc.hasNaN() || (!finc.allFinite()))
                    {
                        std::cout << finc.transpose() << std::endl;
                        std::cout << ULxy.transpose() << std::endl;
                        std::cout << GradU << std::endl;
                        assert(false);
                    }
                });

            rhs[iCell] += sourceV(Eigen::seq(0, cnvars - 1)) / fv->volumeLocal[iCell];
            jacobianCellSourceDiag[iCell] = sourceV(Eigen::seq(cnvars, 2 * cnvars - 1)) / fv->volumeLocal[iCell];
        }
        InsertCheck(u.dist->getMPI(), "EvaluateRHS -1");
    }

    void EulerEvaluator::LUSGSADMatrixInit(std::vector<real> &dTau, real dt, real alphaDiag, ArrayDOFV &u,
                                           int jacobianCode,
                                           real t)
    {
        assert(false);                                                      // TODO: to support model expanding
        for (index iCell = 0; iCell < mesh->cell2nodeLocal.size(); iCell++) // includes ghost
        {
            if (iCell < mesh->cell2nodeLocal.dist->size())
                jacobianCell[iCell] = Eigen::Matrix<real, 5, 5>::Identity() *
                                      (fv->volumeLocal[iCell] / dTau[iCell] +
                                       fv->volumeLocal[iCell] / dt);
            else
                jacobianCell[iCell].setConstant(UnInitReal);
        }
        for (index iFace = 0; iFace < mesh->face2nodeLocal.size(); iFace++)
        {
            auto &faceRecAtr = vfv->faceRecAtrLocal[iFace][0];
            auto &faceAtr = mesh->faceAtrLocal[iFace][0];
            auto f2c = mesh->face2cellLocal[iFace];

            Eigen::Matrix<real, 5, 1> UL, UR, ULxy, URxy;
            ULxy = u[f2c[0]](Eigen::seq(0, 4));

            Elem::tPoint unitNorm = vfv->faceNormCenter[iFace].normalized();
            Elem::tJacobi normBase = Elem::NormBuildLocalBaseV(unitNorm);
            Elem::tPoint centerRL;
            real volumeInverse;

            if (f2c[1] != FACE_2_VOL_EMPTY)
            {
                UR = URxy = u[f2c[1]](Eigen::seq(0, 4));
                UR({1, 2, 3}) = normBase.transpose() * UR({1, 2, 3});

                centerRL = vfv->getCellCenter(f2c[1]) - vfv->getCellCenter(f2c[0]);
                volumeInverse = 0.5 / fv->volumeLocal[f2c[1]] + 0.5 / fv->volumeLocal[f2c[0]];
            }
            else if (true)
            {
                URxy = generateBoundaryValue(
                    ULxy,
                    unitNorm,
                    normBase,
                    vfv->faceCenters[iFace],
                    t,
                    BoundaryType(faceAtr.iPhy));

                centerRL = (vfv->faceCenters[iFace] - vfv->getCellCenter(f2c[0])) * 2;
                volumeInverse = 1.0 / fv->volumeLocal[f2c[0]];
            }
            UL = ULxy, UR = URxy;
            UL({1, 2, 3}) = normBase.transpose() * UL({1, 2, 3});
            UR({1, 2, 3}) = normBase.transpose() * UR({1, 2, 3});

            Eigen::Matrix<real, 5, 1> F;
            Eigen::Matrix<real, 5, 5> dFdUL, dFdUR;

            Gas::RoeFlux_IdealGas_HartenYee_AutoDiffGen(
                UL, UR, settings.idealGasProperty.gamma, F, dFdUL, dFdUR,
                [&]() {});
            if (F.hasNaN() || dFdUL.hasNaN() || dFdUR.hasNaN())
            {
                std::cout << "F \n"
                          << F << std::endl;
                std::cout << "GL \n"
                          << dFdUL << std::endl;
                std::cout << "GR \n"
                          << dFdUR << std::endl;
                std::cout << "UL \n"
                          << UL << std::endl;
                std::cout << "UR \n"
                          << UR << std::endl;
                assert(false);
            }
            F({1, 2, 3}) = normBase * F({1, 2, 3});
            dFdUL.transposeInPlace();
            dFdUR.transposeInPlace(); // -> dF_idUj
            dFdUL({1, 2, 3}, Eigen::all) = normBase * dFdUL({1, 2, 3}, Eigen::all);
            dFdUR({1, 2, 3}, Eigen::all) = normBase * dFdUR({1, 2, 3}, Eigen::all);
            dFdUL(Eigen::all, {1, 2, 3}) *= normBase.transpose();
            dFdUR(Eigen::all, {1, 2, 3}) *= normBase.transpose();

            if (jacobianCode == 2)
            {
                //*** USES vis jacobian
                // Elem::tPoint vGrad = unitNorm / (unitNorm.dot(centerRL));
                // Elem::tPoint vGrad = centerRL / centerRL.squaredNorm();
                Elem::tPoint vGrad = unitNorm * (fv->faceArea[iFace] * volumeInverse);
                Eigen::Matrix<real, 5, 1> UC = (ULxy + URxy) * 0.5;
                Eigen::Matrix<real, 3, 5> gradU = vGrad * (URxy - ULxy).transpose();
                Eigen::Matrix<real, 5, 1> Fvis;
                Eigen::Matrix<real, 5, 5> dFvisDu, dFvisDUL, dFvisDUR;
                Eigen::Matrix<real, 3, 20> dFvisDGu;

                real k = settings.idealGasProperty.CpGas * settings.idealGasProperty.muGas / settings.idealGasProperty.prGas;
                Gas::ViscousFlux_IdealGas_N_AutoDiffGen(UC, gradU, unitNorm, false,
                                                        settings.idealGasProperty.gamma, settings.idealGasProperty.muGas,
                                                        k, settings.idealGasProperty.CpGas,
                                                        Fvis, dFvisDu, dFvisDGu);

                Eigen::Matrix<real, 5, 5> dFvisDuDiff;
                dFvisDuDiff({0, 1, 2, 3, 4}, {0}).setZero();
                dFvisDuDiff({0, 1, 2, 3, 4}, {1, 2, 3, 4}) = (vGrad.transpose() * dFvisDGu).reshaped<Eigen::ColMajor>(5, 4);

                dFvisDUL = (dFvisDu * 0.5 - dFvisDuDiff).transpose();
                dFvisDUR = (dFvisDu * 0.5 + dFvisDuDiff).transpose();
                // std::cout << "lamFaceVis " << lambdaFaceVis[iFace] << " dFvisDuDiff\n"
                //           << dFvisDuDiff << std::endl;
                // assert(false);
                // //* A
                // dFvisDUL(0, 0) = -0.5 * lambdaFaceVis[iFace];
                // dFvisDUR(0, 0) = 0.5 * lambdaFaceVis[iFace];
                //* B
                dFvisDUL -= 0.5 * lambdaFaceVis[iFace] * Eigen::Matrix<real, 5, 5>::Identity();
                dFvisDUR += 0.5 * lambdaFaceVis[iFace] * Eigen::Matrix<real, 5, 5>::Identity();
                // //* C
                // dFvisDUL *= 0.5;
                // dFvisDUR *= 0.5;
                // dFvisDUL -= 0.5 * 0.5 * lambdaFaceVis[iFace] * Eigen::Matrix<real, 5, 5>::Identity();
                // dFvisDUR += 0.5 * 0.5 * lambdaFaceVis[iFace] * Eigen::Matrix<real, 5, 5>::Identity();

                dFdUFace[iFace]({0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}) = dFdUL - dFvisDUL;
                dFdUFace[iFace]({5, 6, 7, 8, 9}, {0, 1, 2, 3, 4}) = dFdUR - dFvisDUR;

                jacobianFace[iFace]({0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}) =
                    fv->faceArea[iFace] *
                    (-dFdUL + dFvisDUL) * alphaDiag; // right: use minus version
                jacobianFace[iFace]({5, 6, 7, 8, 9}, {0, 1, 2, 3, 4}) =
                    fv->faceArea[iFace] *
                    (dFdUR - dFvisDUR) * alphaDiag; // left: uses plus version
                //*** USES vis jacobian
            }
            else if (jacobianCode == 1)
            {
                jacobianFace[iFace]({0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}) =
                    fv->faceArea[iFace] *
                    (-dFdUL - 0.5 * lambdaFaceVis[iFace] * Eigen::Matrix<real, 5, 5>::Identity()) * alphaDiag; // right: use minus version
                jacobianFace[iFace]({5, 6, 7, 8, 9}, {0, 1, 2, 3, 4}) =
                    fv->faceArea[iFace] *
                    (dFdUR - 0.5 * lambdaFaceVis[iFace] * Eigen::Matrix<real, 5, 5>::Identity()) * alphaDiag; // left: uses plus version
            }

            jacobianCell[f2c[0]] -= jacobianFace[iFace]({0, 1, 2, 3, 4}, {0, 1, 2, 3, 4});
            if (f2c[1] != FACE_2_VOL_EMPTY)
            {
                jacobianCell[f2c[1]] -= jacobianFace[iFace]({5, 6, 7, 8, 9}, {0, 1, 2, 3, 4});
            }
            else if (faceAtr.iPhy == BoundaryType::Farfield ||
                     faceAtr.iPhy == BoundaryType::Special_DMRFar)
            {
                // jacobianCell[f2c[0]];
                // nothing
            }
            else if (faceAtr.iPhy == BoundaryType::Wall_Euler)
            {
                Eigen::Matrix<real, 5, 5> jacobianRL = -jacobianFace[iFace]({5, 6, 7, 8, 9}, {0, 1, 2, 3, 4});
                jacobianRL(Eigen::all, {1, 2, 3}) *= normBase;
                jacobianRL(Eigen::all, {1}) *= -1;
                jacobianRL(Eigen::all, {1, 2, 3}) *= normBase.transpose();
                jacobianCell[f2c[0]] -= jacobianRL;
            }
            else if (faceAtr.iPhy == BoundaryType::Wall_NoSlip)
            {
                Eigen::Matrix<real, 5, 5> jacobianRL = -jacobianFace[iFace]({5, 6, 7, 8, 9}, {0, 1, 2, 3, 4});
                jacobianRL(Eigen::all, {1, 2, 3}) *= -1;
                jacobianCell[f2c[0]] -= jacobianRL;
            }
            else if (faceAtr.iPhy == BoundaryType::Wall)
            {
                std::cout << "Wall is not a proper bc" << std::endl;
                assert(false);
            }
            else
            {
                assert(false);
            }
        }
        for (index iCell = 0; iCell < mesh->cell2nodeLocal.size(); iCell++) // includes ghost
        {
            if (iCell < mesh->cell2nodeLocal.dist->size())
            {
                jacobianCellInv[iCell] = jacobianCell[iCell].fullPivLu().inverse();

                if (jacobianCell[iCell].hasNaN() || jacobianCellInv[iCell].hasNaN() ||
                    (!(jacobianCell[iCell].allFinite() && jacobianCellInv[iCell].allFinite())))
                {
                    std::cout << "JCInv\n"
                              << jacobianCellInv[iCell] << std::endl;
                    std::cout << "JC\n"
                              << jacobianCell[iCell] << std::endl;
                    assert(false);
                }
            }
            else
                jacobianCellInv[iCell].setConstant(UnInitReal);
        }
    }

    void EulerEvaluator::LUSGSADMatrixVec(ArrayDOFV &u, ArrayDOFV &uInc, ArrayDOFV &AuInc)
    {
        assert(false); // TODO: to support model expanding
        for (index iScan = 0; iScan < mesh->cell2nodeLocal.dist->size(); iScan++)
        {
            index iCell = iScan;
            // iCell = (*vfv->SOR_iScan2iCell)[iCell];//TODO: add rb-sor

            auto &c2f = mesh->cell2faceLocal[iCell];
            Eigen::Vector<real, 5> uIncNewBuf;
            uIncNewBuf.setZero(); // norhs

            if (uInc[iCell].hasNaN())
            {
                std::cout << uInc[iCell] << std::endl;
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
                    Eigen::Matrix<real, 5, 5> jacobianOther = iCellAtFace
                                                                  ? jacobianFace[iFace]({0, 1, 2, 3, 4}, {0, 1, 2, 3, 4})
                                                                  : jacobianFace[iFace]({5, 6, 7, 8, 9}, {0, 1, 2, 3, 4});
                    uIncNewBuf += jacobianOther * uInc[iCellOther](Eigen::seq(0, 4));
                }
            }
            AuInc[iCell](Eigen::seq(0, 4)) = jacobianCell[iCell] * uInc[iCell](Eigen::seq(0, 4)) + uIncNewBuf;
        }
    }

    /**
     * @brief waring! rhs here is with volume, different from UpdateLUSGSADForward
     **/
    void EulerEvaluator::UpdateLUSGSADForward(ArrayDOFV &rhs, ArrayDOFV &u, ArrayDOFV &uInc, ArrayDOFV &uIncNew)
    {
        assert(false); // TODO: to support model expanding
        index nCellDist = mesh->cell2nodeLocal.dist->size();
        for (index iScan = 0; iScan < nCellDist; iScan++)
        {
            index iCell = iScan;
            iCell = (*vfv->SOR_iScan2iCell)[iCell]; // TODO: add rb-sor

            auto &c2f = mesh->cell2faceLocal[iCell];
            Eigen::Vector<real, 5> uIncNewBuf;
            uIncNewBuf = rhs[iCell](Eigen::seq(0, 4));

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
                        Eigen::Matrix<real, 5, 5> jacobianOther = iCellAtFace
                                                                      ? jacobianFace[iFace]({0, 1, 2, 3, 4}, {0, 1, 2, 3, 4})
                                                                      : jacobianFace[iFace]({5, 6, 7, 8, 9}, {0, 1, 2, 3, 4});
                        uIncNewBuf -= jacobianOther * uInc[iCellOther](Eigen::seq(0, 4));
                    }
                }
            }

            uIncNew[iCell](Eigen::seq(0, 4)) = jacobianCellInv[iCell] * uIncNewBuf;

            // fix rho increment
            // if (u[iCell](0) + uIncNew[iCell](0) < u[iCell](0) * 1e-5)
            //     uIncNew[iCell](0) = -u[iCell](0) * (1 - 1e-5);
            uIncNew[iCell](Eigen::seq(0, 4)) = CompressRecPart(u[iCell](Eigen::seq(0, 4)), uIncNew[iCell](Eigen::seq(0, 4))) - u[iCell](Eigen::seq(0, 4));
        }
    }

    void EulerEvaluator::UpdateLUSGSADBackward(ArrayDOFV &rhs, ArrayDOFV &u, ArrayDOFV &uInc, ArrayDOFV &uIncNew)
    {
        assert(false); // TODO: to support model expanding
        index nCellDist = mesh->cell2nodeLocal.dist->size();
        for (index iScan = nCellDist - 1; iScan >= 0; iScan--)
        {
            index iCell = iScan;
            iCell = (*vfv->SOR_iScan2iCell)[iCell]; // TODO: add rb-sor

            auto &c2f = mesh->cell2faceLocal[iCell];
            Eigen::Vector<real, 5> uIncNewBuf;
            // uIncNewBuf = rhs[iCell];
            uIncNewBuf.setZero(); // Back

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
                    if (iScanOther > iScan) // Back
                    {
                        Eigen::Matrix<real, 5, 5> jacobianOther = iCellAtFace
                                                                      ? jacobianFace[iFace]({0, 1, 2, 3, 4}, {0, 1, 2, 3, 4})
                                                                      : jacobianFace[iFace]({5, 6, 7, 8, 9}, {0, 1, 2, 3, 4});
                        uIncNewBuf -= jacobianOther * uInc[iCellOther](Eigen::seq(0, 4));
                    }
                }
            }

            uIncNew[iCell](Eigen::seq(0, 4)) += jacobianCellInv[iCell] * uIncNewBuf; // Back

            uIncNew[iCell](Eigen::seq(0, 4)) = CompressRecPart(u[iCell](Eigen::seq(0, 4)), uIncNew[iCell](Eigen::seq(0, 4))) - u[iCell](Eigen::seq(0, 4));
        }
    }

    void EulerEvaluator::LUSGSMatrixInit(std::vector<real> &dTau, real dt, real alphaDiag,
                                         ArrayDOFV &u, ArrayRecV &uRec,
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

            jacobianCell[iCell].setIdentity();

            // LUSGS diag part
            real fpDivisor = 1.0 / dTau[iCell] + 1.0 / dt;
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                fpDivisor += (0.5 * alphaDiag) * fv->faceArea[iFace] * lambdaFace[iFace] / fv->volumeLocal[iCell];
            }
            jacobianCell[iCell] *= fpDivisor; //! all passive vars use same diag for flux part

            // jacobian diag

            jacobianCell[iCell] += alphaDiag * jacobianCellSourceDiag[iCell].asDiagonal();


            //! assuming diagonal here!
            jacobianCellInv[iCell] = jacobianCell[iCell].diagonal().array().inverse().matrix().asDiagonal();
            // jacobianCellInv[iCell] = jacobianCell[iCell].partialPivLu().inverse();

            // std::cout << "jacobian Diag\n"
            //           << jacobianCell[iCell] << std::endl;
        }
    }

    void EulerEvaluator::LUSGSMatrixVec(real alphaDiag, ArrayDOFV &u, ArrayDOFV &uInc, ArrayDOFV &AuInc)
    {
        InsertCheck(u.dist->getMPI(), "LUSGSMatrixVec 1");
        int cnvars = nVars;
        for (index iScan = 0; iScan < mesh->cell2nodeLocal.dist->size(); iScan++)
        {
            index iCell = iScan;
            // iCell = (*vfv->SOR_iScan2iCell)[iCell];//TODO: add rb-sor

            auto &c2f = mesh->cell2faceLocal[iCell];
            Eigen::Vector<real, -1> uIncNewBuf(cnvars);
            uIncNewBuf.setZero(); // norhs

            if (uInc[iCell].hasNaN())
            {
                std::cout << uInc[iCell] << std::endl;
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
                        Eigen::Vector<real, -1> fInc;
                        {
                            Elem::tPoint unitNorm = vfv->faceNormCenter[iFace].normalized() *
                                                    (iCellAtFace ? -1 : 1); // faces out

                            // fInc = fluxJacobian0_Right(
                            //            u[iCellOther],
                            //            unitNorm,
                            //            BoundaryType::Inner) *
                            //        uInc[iCellOther]; //! always inner here
                            fInc = fluxJacobian0_Right_Times_du(
                                u[iCellOther],
                                unitNorm,
                                BoundaryType::Inner, uInc[iCellOther]); //! always inner here
                        }

                        uIncNewBuf -= (0.5 * alphaDiag) * fv->faceArea[iFace] / fv->volumeLocal[iCell] *
                                      (fInc - lambdaFace[iFace] * uInc[iCellOther]);
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
            AuInc[iCell] = jacobianCell[iCell] * uInc[iCell] - uIncNewBuf;
            if (AuInc[iCell].hasNaN())
            {
                std::cout << AuInc[iCell].transpose() << std::endl
                          << uInc[iCell].transpose() << std::endl
                          << u[iCell].transpose() << std::endl
                          << jacobianCell[iCell] << std::endl
                          << iCell << std::endl;
                assert(!AuInc[iCell].hasNaN());
            }
        }
        InsertCheck(u.dist->getMPI(), "LUSGSMatrixVec -1");
    }

    void EulerEvaluator::UpdateLUSGSForward(real alphaDiag,
                                            ArrayDOFV &rhs, ArrayDOFV &u, ArrayDOFV &uInc, ArrayDOFV &uIncNew)
    {
        InsertCheck(u.dist->getMPI(), "UpdateLUSGSForward 1");
        int cnvars = nVars;
        index nCellDist = mesh->cell2nodeLocal.dist->size();
        for (index iScan = 0; iScan < nCellDist; iScan++)
        {
            index iCell = iScan;
            iCell = (*vfv->SOR_iScan2iCell)[iCell]; // TODO: add rb-sor

            auto &c2f = mesh->cell2faceLocal[iCell];
            Eigen::Vector<real, -1> uIncNewBuf(nVars);
            uIncNewBuf = rhs[iCell];

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
                        Eigen::Vector<real, -1> fInc;

                        {
                            Elem::tPoint unitNorm = vfv->faceNormCenter[iFace].normalized() *
                                                    (iCellAtFace ? -1 : 1); // faces out

                            // fInc = fluxJacobian0_Right(
                            //            u[iCellOther],
                            //            unitNorm,
                            //            BoundaryType::Inner) *
                            //        uInc[iCellOther]; //! always inner here
                            fInc = fluxJacobian0_Right_Times_du(
                                u[iCellOther],
                                unitNorm,
                                BoundaryType::Inner, uInc[iCellOther]); //! always inner here
                        }

                        uIncNewBuf -= (0.5 * alphaDiag) * fv->faceArea[iFace] / fv->volumeLocal[iCell] *
                                      (fInc - lambdaFace[iFace] * uInc[iCellOther]);
                        if (uIncNewBuf.hasNaN() || (!uIncNewBuf.allFinite()))
                        {
                            std::cout << rhs[iCell].transpose() << std::endl
                                      << fInc.transpose() << std::endl
                                      << uInc[iCellOther].transpose() << std::endl;
                            assert(!(uIncNewBuf.hasNaN() || (!uIncNewBuf.allFinite())));
                        }
                    }
                }
            }
            uIncNew[iCell] = jacobianCellInv[iCell] * uIncNewBuf;
            if (uIncNew[iCell].hasNaN())
            {
                std::cout << uIncNew[iCell].transpose() << std::endl
                          << jacobianCellInv[iCell] << std::endl
                          << iCell << std::endl;
                assert(!uIncNew[iCell].hasNaN());
            }

            // fix rho increment
            // if (u[iCell](0) + uIncNew[iCell](0) < u[iCell](0) * 1e-5)
            //     uIncNew[iCell](0) = -u[iCell](0) * (1 - 1e-5);
            uIncNew[iCell] = CompressInc(u[iCell], uIncNew[iCell], rhs[iCell]);
        }
        InsertCheck(u.dist->getMPI(), "UpdateLUSGSForward -1");
    }

    void EulerEvaluator::UpdateLUSGSBackward(real alphaDiag,
                                             ArrayDOFV &rhs, ArrayDOFV &u, ArrayDOFV &uInc, ArrayDOFV &uIncNew)
    {
        InsertCheck(u.dist->getMPI(), "UpdateLUSGSBackward 1");
        int cnvars = nVars;
        index nCellDist = mesh->cell2nodeLocal.dist->size();
        for (index iScan = nCellDist - 1; iScan >= 0; iScan--)
        {
            index iCell = iScan;
            iCell = (*vfv->SOR_iScan2iCell)[iCell];

            auto &c2f = mesh->cell2faceLocal[iCell];
            Eigen::Vector<real, -1> uIncNewBuf(cnvars);
            uIncNewBuf.setZero(); // backward

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
                        Eigen::Vector<real, -1> fInc;

                        {
                            Elem::tPoint unitNorm = vfv->faceNormCenter[iFace].normalized() *
                                                    (iCellAtFace ? -1 : 1); // faces out

                            // fInc = fluxJacobian0_Right(
                            //            u[iCellOther],
                            //            unitNorm,
                            //            BoundaryType::Inner) *
                            //        uInc[iCellOther]; //! always inner here
                            fInc = fluxJacobian0_Right_Times_du(
                                u[iCellOther],
                                unitNorm,
                                BoundaryType::Inner, uInc[iCellOther]); //! always inner here
                        }

                        uIncNewBuf -= (0.5 * alphaDiag) * fv->faceArea[iFace] / fv->volumeLocal[iCell] *
                                      (fInc - lambdaFace[iFace] * uInc[iCellOther]);
                    }
                }
            }
            uIncNew[iCell] += jacobianCellInv[iCell] * uIncNewBuf; // backward

            // fix rho increment
            // if (u[iCell](0) + uIncNew[iCell](0) < u[iCell](0) * 1e-5)
            //     uIncNew[iCell](0) = -u[iCell](0) * (1 - 1e-5);
            uIncNew[iCell] = CompressInc(u[iCell], uIncNew[iCell], rhs[iCell]);
        }
        InsertCheck(u.dist->getMPI(), "UpdateLUSGSBackward -1");
    }

    void EulerEvaluator::UpdateSGS(real alphaDiag,
                                   ArrayDOFV &rhs, ArrayDOFV &u, ArrayDOFV &uInc, ArrayDOFV &uIncNew, bool ifForward)
    {
        int cnvars = nVars;
        for (index iScan = 0; iScan < mesh->cell2nodeLocal.dist->size(); iScan++)
        {
            index iCell;
            if (ifForward)
                iCell = iScan;
            else
                iCell = mesh->cell2nodeLocal.dist->size() - 1 - iScan;
            auto &c2f = mesh->cell2faceLocal[iCell];
            Eigen::Vector<real, -1> uIncNewBuf;
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
                        Eigen::Vector<real, -1> fInc;

                        {
                            Elem::tPoint unitNorm = vfv->faceNormCenter[iFace].normalized() *
                                                    (iCellAtFace ? -1 : 1); // faces out

                            // fInc = fluxJacobian0_Right(
                            //            u[iCellOther],
                            //            unitNorm,
                            //            BoundaryType::Inner) *
                            //        uInc[iCellOther]; //! always inner here
                            fInc = fluxJacobian0_Right_Times_du(
                                u[iCellOther],
                                unitNorm,
                                BoundaryType::Inner, uInc[iCellOther]); //! always inner here
                        }

                        uIncNewBuf -= (0.5 * alphaDiag) * fv->faceArea[iFace] / fv->volumeLocal[iCell] *
                                      (fInc - lambdaFace[iFace] * uInc[iCellOther]);
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

    void EulerEvaluator::FixUMaxFilter(ArrayDOFV &u)
    {
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
            u[iCell]({1, 2, 3}) *= uScale;
            real e = u[iCell](4) - 0.5 * u[iCell]({1, 2, 3}).squaredNorm() / (u[iCell](0) + verySmallReal);
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
                    real EkOther = 0.5 * u[iCellOther]({1, 2, 3}).squaredNorm() / (u[iCellOther](0) + verySmallReal);
                    eMU += u[iCellOther](4) - EkOther;
                    eML += 1;
                }
            }
            eT = scaleEInternalCutOff * eMU / eML;
            eT = eTC; //! using global
            eTSqr = eT * eT;

            real Ek = 0.5 * u[iCell]({1, 2, 3}).squaredNorm() / (u[iCell](0) + verySmallReal);
            real e = u[iCell](4) - Ek;
            if (e <= 0)
                e = eT * 0.5;
            else if (e <= eT)
                e = (e * e + eTSqr) / (2 * eT);
            // eNew + Ek = e + Eknew
            real EkNew = Ek - e + (u[iCell](4) - Ek);
            if (EkNew > 0)
            {
                real uScale = sqrt(EkNew / Ek);
                u[iCell]({1, 2, 3}) *= uScale;
            }
            else
            {
                u[iCell]({1, 2, 3}) *= 0;
                u[iCell](4) -= EkNew;
            }
            // u[iCell](4) =Ek + e;
        }
    }

    void EulerEvaluator::EvaluateResidual(Eigen::Vector<real, -1> &res, ArrayDOFV &rhs, index P)
    {
        if (P < 3)
        {
            Eigen::Vector<real, -1> resc;
            resc.resizeLike(rhs[0]);
            resc.setZero();

            for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
                if (rhs[iCell].hasNaN() || (!rhs[iCell].allFinite()))
                {
                    std::cout << rhs[iCell] << std::endl;
                    assert(false);
                }
                resc += rhs[iCell].array().abs().pow(P).matrix();
            }
            MPI_Allreduce(resc.data(), res.data(), res.size(), DNDS_MPI_REAL, MPI_SUM, rhs.dist->getMPI().comm);
            res = res.array().pow(1.0 / P).matrix();
            // std::cout << res << std::endl;
        }
        else
        {
            Eigen::Vector<real, -1> resc;
            resc.resizeLike(rhs[0]);
            resc.setZero();
            for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
                resc = resc.array().max(rhs[iCell].array().abs()).matrix();
            MPI_Allreduce(resc.data(), res.data(), res.size(), DNDS_MPI_REAL, MPI_MAX, rhs.dist->getMPI().comm);
        }
    }
}
