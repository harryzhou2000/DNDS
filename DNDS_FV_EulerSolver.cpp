#include "DNDS_FV_EulerSolver.hpp"

#include "DNDS_GasGen.hpp"
#include "DNDS_HardEigen.h"

namespace DNDS
{

    Eigen::Vector<real, 5> EulerEvaluator::CompressRecPart(
        const Eigen::Vector<real, 5> &umean,
        const Eigen::Vector<real, 5> &uRecInc)
    {

        // if (umean(0) + uRecInc(0) < 0)
        // {
        //     std::cout << umean.transpose() << std::endl
        //               << uRecInc.transpose() << std::endl;
        //     assert(false);
        // }
        // return umean + uRecInc; // ! no compress shortcut
        // return umean; // ! 0th order shortcut

        // // * Compress Method
        // real compressT = 0.00001;
        // real eFixRatio = 0.00001;
        // Eigen::Vector<real, 5> ret;

        // real compress = 1.0;
        // if ((umean(0) + uRecInc(0)) < umean(0) * compressT)
        //     compress *= umean(0) * (1 - compressT) / uRecInc(0);

        // ret = umean + uRecInc * compress;

        // real Ek = ret({1, 2, 3}).squaredNorm() * 0.5 / (verySmallReal + ret(0));
        // real eT = eFixRatio * Ek;
        // real e = ret(4) - Ek;
        // if (e < 0)
        //     e = eT * 0.5;
        // else if (e < eT)
        //     e = (e * e + eT * eT) / (2 * eT);
        // ret(4) = e + Ek;
        // // * Compress Method

        Eigen::Vector<real, 5> ret = umean + uRecInc;
        real eK = ret({1, 2, 3}).squaredNorm() * 0.5 / (verySmallReal + std::abs(ret(0)));
        real e = ret(4) - eK;
        if (e <= 0 || ret(0) <= 0)
            ret = umean;

        return ret;
    }

    void EulerEvaluator::EvaluateDt(std::vector<real> &dt,
                                    ArrayLocal<VecStaticBatch<5>> &u,
                                    // ArrayLocal<SemiVarMatrix<5>> &uRec,
                                    real CFL, real &dtMinall, real MaxDt,
                                    bool UseLocaldt)
    {
        for (auto &i : lambdaCell)
            i = 0.0;

        for (index iFace = 0; iFace < mesh->face2nodeLocal.size(); iFace++)
        {
            auto f2c = mesh->face2cellLocal[iFace];
            auto faceDiBjCenterBatchElemVR = (*vfv->faceDiBjCenterBatch)[iFace];
            Elem::tPoint unitNorm = vfv->faceNormCenter[iFace].normalized();

            index iCellL = f2c[0];
            Eigen::Vector<real, 5> uMean = u[iCellL].p();
            real pL, asqrL, HL, pR, asqrR, HR;
            Gas::tVec vL = u[iCellL].p()({1, 2, 3}) / u[iCellL].p()(0);
            Gas::tVec vR = vL;
            Gas::IdealGasThermal(u[iCellL].p()(4), u[iCellL].p()(0), vL.squaredNorm(),
                                 settings.idealGasProperty.gamma,
                                 pL, asqrL, HL);
            pR = pL, HR = HL, asqrR = asqrL;
            if (f2c[1] != FACE_2_VOL_EMPTY)
            {
                uMean = (uMean + u[f2c[1]].p()) * 0.5;
                vR = u[f2c[1]].p()({1, 2, 3}) / u[f2c[1]].p()(0);
                Gas::IdealGasThermal(u[f2c[1]].p()(4), u[f2c[1]].p()(0), vR.squaredNorm(),
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

            real muf = settings.idealGasProperty.muGas;
            real lamVis = muf / uMean(0) *
                          std::max(4. / 3., settings.idealGasProperty.gamma / settings.idealGasProperty.prGas);

            // Elem::tPoint gradL{0, 0, 0},gradR{0, 0, 0};
            // gradL({0, 1}) = faceDiBjCenterBatchElemVR.m(0)({1, 2}, Eigen::all).rightCols(faceDiBjCenterBatchElemVR.m(0).cols() - 1) * uRec[iCellL].m();
            // if (f2c[1] != FACE_2_VOL_EMPTY) // can't be non local
            //     gradR({0, 1}) = faceDiBjCenterBatchElemVR.m(1)({1, 2}, Eigen::all).rightCols(faceDiBjCenterBatchElemVR.m(1).cols() - 1) * uRec[f2c[1]].m();
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
    void EulerEvaluator::EvaluateRHS(ArrayDOF<5u> &rhs, ArrayDOF<5u> &u,
                                     ArrayLocal<SemiVarMatrix<5u>> &uRec, real t)
    {
        for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
        {
            rhs[iCell].setZero();
        }

        for (index iFace = 0; iFace < mesh->face2nodeLocal.size(); iFace++)
        {
            auto &faceRecAtr = vfv->faceRecAtrLocal[iFace][0];
            auto &faceAtr = mesh->faceAtrLocal[iFace][0];
            auto f2c = mesh->face2cellLocal[iFace];
            Elem::ElementManager eFace(faceAtr.type, faceRecAtr.intScheme);
            Eigen::Vector<real, 5> flux;
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

                    Eigen::Vector<real, 5> UL =
                        faceDiBjGaussBatchElemVR.m(ig * 2 + 0).row(0).rightCols(uRec[f2c[0]].m().rows()) *
                        uRec[f2c[0]].m() * IF_NOT_NOREC;
                    // UL += u[f2c[0]]; //! do not forget the mean value
                    UL = CompressRecPart(u[f2c[0]], UL);
                    Eigen::Vector<real, 5> ULxy = UL;
                    UL({1, 2, 3}) = normBase.transpose() * UL({1, 2, 3});
                    Eigen::Vector<real, 5> UR, URxy;

                    Eigen::Matrix<real, 3, 5> GradULxy, GradURxy;
                    GradULxy.setZero(), GradURxy.setZero();
                    GradULxy({0, 1}, {0, 1, 2, 3, 4}) =
                        faceDiBjGaussBatchElemVR.m(ig * 2 + 0)({1, 2}, Eigen::seq(1, uRec[f2c[0]].m().rows() + 1 - 1)) *
                        uRec[f2c[0]].m() * IF_NOT_NOREC; // ! 2d here

                    real minVol = fv->volumeLocal[f2c[0]];

                    if (f2c[1] != FACE_2_VOL_EMPTY)
                    {
                        UR =
                            faceDiBjGaussBatchElemVR.m(ig * 2 + 1).row(0).rightCols(uRec[f2c[1]].m().rows()) *
                            uRec[f2c[1]].m() * IF_NOT_NOREC;
                        // UR += u[f2c[1]];
                        UR = CompressRecPart(u[f2c[1]], UR);
                        URxy = UR;
                        UR({1, 2, 3}) = normBase.transpose() * UR({1, 2, 3});

                        GradURxy({0, 1}, {0, 1, 2, 3, 4}) =
                            faceDiBjGaussBatchElemVR.m(ig * 2 + 1)({1, 2}, Eigen::seq(1, uRec[f2c[1]].m().rows() + 1 - 1)) *
                            uRec[f2c[1]].m() * IF_NOT_NOREC; // ! 2d here

                        minVol = std::min(minVol, fv->volumeLocal[f2c[1]]);
                    }
                    else if (faceAtr.iPhy == BoundaryType::Farfield ||
                             faceAtr.iPhy == BoundaryType::Special_DMRFar)
                    {
                        if (faceAtr.iPhy == BoundaryType::Farfield)
                            UR = settings.farFieldStaticValue;
                        else if (faceAtr.iPhy == BoundaryType::Special_DMRFar)
                        {
                            Elem::tPoint pPhysics = coords * DiNj(0, Eigen::all).transpose();
                            pPhysics = vfv->faceCenters[iFace]; //! using center!
                            real uShock = 10;

                            if (((pPhysics(0) - uShock / std::sin(pi / 3) * t - 1. / 6.) -
                                 pPhysics(1) / std::tan(pi / 3)) > 0)
                                UR = {1.4, 0, 0, 0, 2.5};
                            else
                                UR = {8, 57.157676649772960, -33, 0, 5.635e2};
                        }
                        else
                            assert(false);

                        URxy = UR;
                        UR({1, 2, 3}) = normBase.transpose() * UR({1, 2, 3});

                        GradURxy = GradULxy;
                    }
                    else if (faceAtr.iPhy == BoundaryType::Wall_Euler)
                    {
                        UR = UL;
                        UR(1) *= -1;
                        URxy = UR;
                        URxy({1, 2, 3}) = normBase * URxy({1, 2, 3});

                        GradURxy = GradULxy;
                    }
                    else if (faceAtr.iPhy == BoundaryType::Wall_NoSlip)
                    {
                        UR = UL;
                        UR({1, 2, 3}) *= -1;
                        URxy = UR;
                        URxy({1, 2, 3}) = normBase * URxy({1, 2, 3});

                        GradURxy = GradULxy;
                        assert(false);
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

                    real distGRP = minVol / fv->faceArea[iFace] * 2;
                    // real distGRP = (vfv->cellBaries[f2c[0]] -
                    //                 (f2c[1] != FACE_2_VOL_EMPTY
                    //                      ? vfv->cellBaries[f2c[1]]
                    //                      : 2 * vfv->faceCenters[iFace] - vfv->cellBaries[f2c[0]]))
                    //                    .norm();
                    Eigen::Vector<real, 5> UMeanXy = 0.5 * (ULxy + URxy);
                    Eigen::Matrix<real, 3, 5> GradUMeanXy = (GradURxy + GradULxy) * 0.5 +
                                                            (1.0 / distGRP) *
                                                                (unitNorm * (URxy - ULxy).transpose());
                    Eigen::Matrix<real, 3, 5> VisFlux;

                    real k = settings.idealGasProperty.CpGas * settings.idealGasProperty.muGas / settings.idealGasProperty.prGas;
                    Gas::ViscousFlux_IdealGas(UMeanXy, GradUMeanXy, unitNorm, faceAtr.iPhy == BoundaryType::Wall_NoSlip,
                                              settings.idealGasProperty.gamma,
                                              settings.idealGasProperty.muGas,
                                              k,
                                              settings.idealGasProperty.CpGas,
                                              VisFlux);

                    // Eigen::Vector<real, 5> F;
                    Gas::HLLEFlux_IdealGas_HartenYee(
                        UL, UR, settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace],
                        [&]()
                        {
                            std::cout << "face at" << vfv->faceCenters[iFace].transpose() << '\n';
                            std::cout << "UL" << UL.transpose() << '\n';
                            std::cout << "UR" << UR.transpose() << std::endl;
                        });
                    finc({1, 2, 3}) = normBase * finc({1, 2, 3});
                    finc -= VisFlux.transpose() * unitNorm;
                    finc *= -vfv->faceNorms[iFace][ig].norm(); // don't forget this
                });

            rhs[f2c[0]] += flux / fv->volumeLocal[f2c[0]];
            if (f2c[1] != FACE_2_VOL_EMPTY)
                rhs[f2c[1]] -= flux / fv->volumeLocal[f2c[1]];
        }
        // for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
        // {
        //     rhs[iCell] = rhs[iCell].array().min(1e10).max(-1e10).matrix();
        // }
    }

    void EulerEvaluator::LUSGSADMatrixInit(std::vector<real> &dTau, real dt, real alphaDiag, ArrayDOF<5u> &u,
                                           int jacobianCode,
                                           real t)
    {
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
            UL = ULxy = u[f2c[0]];

            Elem::tPoint unitNorm = vfv->faceNormCenter[iFace].normalized();
            Elem::tJacobi normBase = Elem::NormBuildLocalBaseV(unitNorm);
            Elem::tPoint centerRL;
            real volumeInverse;

            UL({1, 2, 3}) = normBase.transpose() * UL({1, 2, 3});

            if (f2c[1] != FACE_2_VOL_EMPTY)
            {
                UR = URxy = u[f2c[1]];
                UR({1, 2, 3}) = normBase.transpose() * UR({1, 2, 3});

                centerRL = vfv->getCellCenter(f2c[1]) - vfv->getCellCenter(f2c[0]);
                volumeInverse = 0.5 / fv->volumeLocal[f2c[1]] + 0.5 / fv->volumeLocal[f2c[0]];
            }
            else if (faceAtr.iPhy == BoundaryType::Farfield ||
                     faceAtr.iPhy == BoundaryType::Special_DMRFar)
            {
                if (faceAtr.iPhy == BoundaryType::Farfield)
                    UR = URxy = settings.farFieldStaticValue;
                else if (faceAtr.iPhy == BoundaryType::Special_DMRFar)
                {
                    Elem::tPoint pPhysics = vfv->faceCenters[iFace]; //! using center!
                    real uShock = 10;

                    if (((pPhysics(0) - uShock / std::sin(pi / 3) * t - 1. / 6.) -
                         pPhysics(1) / std::tan(pi / 3)) > 0)
                        UR = URxy = {1.4, 0, 0, 0, 2.5};
                    else
                        UR = URxy = {8, 57.157676649772960, -33, 0, 5.635e2};
                }
                else
                    assert(false);

                UR({1, 2, 3}) = normBase.transpose() * UR({1, 2, 3});

                centerRL = (vfv->faceCenters[iFace] - vfv->getCellCenter(f2c[0])) * 2;
                volumeInverse = 1.0 / fv->volumeLocal[f2c[0]];
            }
            else if (faceAtr.iPhy == BoundaryType::Wall_Euler)
            {
                UR = UL;
                UR(1) *= -1;
                URxy = UR;
                URxy({1, 2, 3}) = normBase * URxy({1, 2, 3});

                centerRL = (vfv->faceCenters[iFace] - vfv->getCellCenter(f2c[0])) * 2;
                volumeInverse = 1.0 / fv->volumeLocal[f2c[0]];
            }
            else if (faceAtr.iPhy == BoundaryType::Wall_NoSlip)
            {
                UR = UL;
                UR({1, 2, 3}) *= -1;
                URxy = UR;
                URxy({1, 2, 3}) = normBase * URxy({1, 2, 3});

                centerRL = (vfv->faceCenters[iFace] - vfv->getCellCenter(f2c[0])) * 2;
                volumeInverse = 1.0 / fv->volumeLocal[f2c[0]];
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

    void EulerEvaluator::LUSGSADMatrixVec(ArrayDOF<5u> &u, ArrayDOF<5u> &uInc, ArrayDOF<5u> &AuInc)
    {
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
                    uIncNewBuf += jacobianOther * uInc[iCellOther];
                }
            }
            AuInc[iCell] = jacobianCell[iCell] * uInc[iCell] + uIncNewBuf;
        }
    }

    /**
     * @brief waring! rhs here is with volume, different from UpdateLUSGSADForward
     **/
    void EulerEvaluator::UpdateLUSGSADForward(ArrayDOF<5u> &rhs, ArrayDOF<5u> &u, ArrayDOF<5u> &uInc, ArrayDOF<5u> &uIncNew)
    {
        index nCellDist = mesh->cell2nodeLocal.dist->size();
        for (index iScan = 0; iScan < nCellDist; iScan++)
        {
            index iCell = iScan;
            iCell = (*vfv->SOR_iScan2iCell)[iCell]; // TODO: add rb-sor

            auto &c2f = mesh->cell2faceLocal[iCell];
            Eigen::Vector<real, 5> uIncNewBuf;
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
                        Eigen::Matrix<real, 5, 5> jacobianOther = iCellAtFace
                                                                      ? jacobianFace[iFace]({0, 1, 2, 3, 4}, {0, 1, 2, 3, 4})
                                                                      : jacobianFace[iFace]({5, 6, 7, 8, 9}, {0, 1, 2, 3, 4});
                        uIncNewBuf -= jacobianOther * uInc[iCellOther];
                    }
                }
            }

            uIncNew[iCell] = jacobianCellInv[iCell] * uIncNewBuf;

            // fix rho increment
            // if (u[iCell](0) + uIncNew[iCell](0) < u[iCell](0) * 1e-5)
            //     uIncNew[iCell](0) = -u[iCell](0) * (1 - 1e-5);
            uIncNew[iCell] = CompressRecPart(u[iCell], uIncNew[iCell]) - u[iCell];
        }
    }

    void EulerEvaluator::UpdateLUSGSADBackward(ArrayDOF<5u> &rhs, ArrayDOF<5u> &u, ArrayDOF<5u> &uInc, ArrayDOF<5u> &uIncNew)
    {
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
                        uIncNewBuf -= jacobianOther * uInc[iCellOther];
                    }
                }
            }

            uIncNew[iCell] += jacobianCellInv[iCell] * uIncNewBuf; // Back

            uIncNew[iCell] = CompressRecPart(u[iCell], uIncNew[iCell]) - u[iCell];
        }
    }

    void EulerEvaluator::LUSGSMatrixVec(std::vector<real> &dTau, real dt, real alphaDiag,
                                        ArrayDOF<5u> &u, ArrayDOF<5u> &uInc, ArrayDOF<5u> &AuInc)
    {
        for (index iScan = 0; iScan < mesh->cell2nodeLocal.dist->size(); iScan++)
        {
            index iCell = iScan;
            // iCell = (*vfv->SOR_iScan2iCell)[iCell];//TODO: add rb-sor

            auto &c2f = mesh->cell2faceLocal[iCell];
            Eigen::Vector<real, 5> uIncNewBuf;
            uIncNewBuf.setZero(); // norhs

            real fpDivisor = fv->volumeLocal[iCell] / dTau[iCell] + fv->volumeLocal[iCell] / dt;
            if (isnan(fpDivisor))
            {
                std::cout << fpDivisor << std::endl
                          << fv->volumeLocal[iCell] << std::endl
                          << dTau[iCell] << std::endl
                          << dt << std::endl;
                assert(false);
            }
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
                fpDivisor += (0.5 * alphaDiag) * fv->faceArea[iFace] * lambdaFace[iFace];
                if (iCellOther != FACE_2_VOL_EMPTY)
                {

                    if (true)
                    {
                        Eigen::Vector<real, 5> umeanOther = u[iCellOther];
                        Eigen::Vector<real, 5> umeanOtherInc = uInc[iCellOther];
                        Eigen::Vector<real, 5> umeanOtherN = umeanOther + umeanOtherInc;
                        Eigen::Vector<real, 5> fInc;
                        do
                        {
                            Elem::tPoint unitNorm = vfv->faceNormCenter[iFace].normalized() *
                                                    (iCellAtFace ? -1 : 1); // faces out
                            Elem::tJacobi normBase = Elem::NormBuildLocalBaseV(unitNorm);
                            real p, pN, asqr, asqrN, H, HN;
                            // if (!(umeanOther(0) > 0 && umeanOtherN(0) > 0))
                            //     std::cout << umeanOther.transpose() << std::endl
                            //               << umeanOtherN.transpose() << std::endl
                            //               << uInc[iCellOther].transpose() << std::endl
                            //               << "i " << iCell << "\t" << iCellOther << std::endl;
                            // assert(umeanOther(0) > 0 && umeanOtherN(0) > 0);
                            Elem::tPoint velo = umeanOther({1, 2, 3}) / umeanOther(0);
                            Elem::tPoint veloN = umeanOtherN({1, 2, 3}) / umeanOtherN(0);
                            real vsqr = velo.squaredNorm();
                            real vsqrN = veloN.squaredNorm();

                            Gas::IdealGasThermal(umeanOther(4), umeanOther(0), vsqr,
                                                 settings.idealGasProperty.gamma, p, asqr, H);
                            Gas::IdealGasThermal(umeanOtherN(4), umeanOtherN(0), vsqrN,
                                                 settings.idealGasProperty.gamma, pN, asqrN, HN);
                            Eigen::Vector<real, 5> f, fN;

                            // linear version
                            Gas::tVec dVelo;
                            real dp;
                            Gas::IdealGasUIncrement(umeanOther, umeanOtherInc, velo, settings.idealGasProperty.gamma, dVelo, dp);
                            Gas::GasInviscidFluxFacialIncrement(umeanOther, umeanOtherInc, unitNorm, velo, dVelo, dp, p, fInc);
                            break;
                            // abort();

                            // get to norm coord
                            umeanOther({1, 2, 3}) = normBase.transpose() * umeanOther({1, 2, 3});
                            umeanOtherN({1, 2, 3}) = normBase.transpose() * umeanOtherN({1, 2, 3});
                            velo = normBase.transpose() * velo;
                            veloN = normBase.transpose() * veloN;
                            Gas::GasInviscidFlux(umeanOther, velo, p, f);
                            Gas::GasInviscidFlux(umeanOtherN, veloN, pN, fN);
                            fInc = fN - f;
                            fInc({1, 2, 3}) = normBase * fInc({1, 2, 3}); // to xy
                            if (fInc.hasNaN() || (!fInc.allFinite()))
                            {
                                std::cout << p << "\t" << pN << std::endl
                                          << umeanOther.transpose() << std::endl
                                          << umeanOtherN.transpose() << std::endl
                                          << umeanOtherInc.transpose() << std::endl
                                          << vsqr << "\t" << vsqrN << std::endl;
                                assert(!(fInc.hasNaN() || (!fInc.allFinite())));
                            }
                        } while (false);

                        uIncNewBuf -= (0.5 * alphaDiag) * fv->faceArea[iFace] *
                                      (fInc - lambdaFace[iFace] * umeanOtherInc);
                        if (uIncNewBuf.hasNaN() || (!uIncNewBuf.allFinite()))
                        {
                            std::cout
                                << fInc.transpose() << std::endl
                                << umeanOtherInc.transpose() << std::endl;
                            assert(!(uIncNewBuf.hasNaN() || (!uIncNewBuf.allFinite())));
                        }
                    }
                }
            }
            // uIncNewBuf /= fpDivisor;
            // uIncNew[iCell] = uIncNewBuf;
            AuInc[iCell] = uInc[iCell] * fpDivisor - uIncNewBuf;
            if (AuInc[iCell].hasNaN())
            {
                std::cout << AuInc[iCell].transpose() << std::endl
                          << uInc[iCell].transpose() << std::endl
                          << u[iCell].transpose() << std::endl
                          << fpDivisor << std::endl
                          << iCell << std::endl;
                assert(!AuInc[iCell].hasNaN());
            }
        }
    }

    void EulerEvaluator::UpdateLUSGSForward(std::vector<real> &dTau, real dt, real alphaDiag,
                                            ArrayDOF<5u> &rhs, ArrayDOF<5u> &u, ArrayDOF<5u> &uInc, ArrayDOF<5u> &uIncNew)
    {
        index nCellDist = mesh->cell2nodeLocal.dist->size();
        for (index iScan = 0; iScan < nCellDist; iScan++)
        {
            index iCell = iScan;
            iCell = (*vfv->SOR_iScan2iCell)[iCell]; // TODO: add rb-sor

            auto &c2f = mesh->cell2faceLocal[iCell];
            Eigen::Vector<real, 5> uIncNewBuf;
            uIncNewBuf = fv->volumeLocal[iCell] * rhs[iCell];

            real fpDivisor = fv->volumeLocal[iCell] / dTau[iCell] + fv->volumeLocal[iCell] / dt;

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto &f2c = (*mesh->face2cellPair)[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                fpDivisor += (0.5 * alphaDiag) * fv->faceArea[iFace] * lambdaFace[iFace];
                if (iCellOther != FACE_2_VOL_EMPTY)
                {

                    index iScanOther = iCellOther < nCellDist
                                           ? (*vfv->SOR_iCell2iScan)[iCellOther]
                                           : iScan + 1;
                    if (iScanOther < iScan)
                    {
                        Eigen::Vector<real, 5> umeanOther = u[iCellOther];
                        Eigen::Vector<real, 5> umeanOtherInc = uInc[iCellOther];
                        Eigen::Vector<real, 5> umeanOtherN = umeanOther + umeanOtherInc;
                        Eigen::Vector<real, 5> fInc;
                        do
                        {
                            Elem::tPoint unitNorm = vfv->faceNormCenter[iFace].normalized() *
                                                    (iCellAtFace ? -1 : 1); // faces out
                            Elem::tJacobi normBase = Elem::NormBuildLocalBaseV(unitNorm);
                            real p, pN, asqr, asqrN, H, HN;
                            // if (!(umeanOther(0) > 0 && umeanOtherN(0) > 0))
                            //     std::cout << umeanOther.transpose() << std::endl
                            //               << umeanOtherN.transpose() << std::endl
                            //               << uInc[iCellOther].transpose() << std::endl
                            //               << "i " << iCell << "\t" << iCellOther << std::endl;
                            // assert(umeanOther(0) > 0 && umeanOtherN(0) > 0);
                            Elem::tPoint velo = umeanOther({1, 2, 3}) / umeanOther(0);
                            Elem::tPoint veloN = umeanOtherN({1, 2, 3}) / umeanOtherN(0);
                            real vsqr = velo.squaredNorm();
                            real vsqrN = veloN.squaredNorm();

                            Gas::IdealGasThermal(umeanOther(4), umeanOther(0), vsqr,
                                                 settings.idealGasProperty.gamma, p, asqr, H);
                            Gas::IdealGasThermal(umeanOtherN(4), umeanOtherN(0), vsqrN,
                                                 settings.idealGasProperty.gamma, pN, asqrN, HN);
                            Eigen::Vector<real, 5> f, fN;

                            // linear version
                            Gas::tVec dVelo;
                            real dp;
                            Gas::IdealGasUIncrement(umeanOther, umeanOtherInc, velo, settings.idealGasProperty.gamma, dVelo, dp);
                            Gas::GasInviscidFluxFacialIncrement(umeanOther, umeanOtherInc, unitNorm, velo, dVelo, dp, p, fInc);
                            break;
                            // abort();

                            // get to norm coord
                            umeanOther({1, 2, 3}) = normBase.transpose() * umeanOther({1, 2, 3});
                            umeanOtherN({1, 2, 3}) = normBase.transpose() * umeanOtherN({1, 2, 3});
                            velo = normBase.transpose() * velo;
                            veloN = normBase.transpose() * veloN;
                            Gas::GasInviscidFlux(umeanOther, velo, p, f);
                            Gas::GasInviscidFlux(umeanOtherN, veloN, pN, fN);
                            fInc = fN - f;
                            fInc({1, 2, 3}) = normBase * fInc({1, 2, 3}); // to xy
                            if (fInc.hasNaN() || (!fInc.allFinite()))
                            {
                                std::cout << p << "\t" << pN << std::endl
                                          << umeanOther.transpose() << std::endl
                                          << umeanOtherN.transpose() << std::endl
                                          << umeanOtherInc.transpose() << std::endl
                                          << vsqr << "\t" << vsqrN << std::endl;
                                assert(!(fInc.hasNaN() || (!fInc.allFinite())));
                            }
                        } while (false);

                        uIncNewBuf -= (0.5 * alphaDiag) * fv->faceArea[iFace] *
                                      (fInc - lambdaFace[iFace] * umeanOtherInc);
                        if (uIncNewBuf.hasNaN() || (!uIncNewBuf.allFinite()))
                        {
                            std::cout << rhs[iCell].transpose() << std::endl
                                      << fInc.transpose() << std::endl
                                      << umeanOtherInc.transpose() << std::endl;
                            assert(!(uIncNewBuf.hasNaN() || (!uIncNewBuf.allFinite())));
                        }
                    }
                }
            }
            uIncNewBuf /= fpDivisor;
            uIncNew[iCell] = uIncNewBuf;
            if (uIncNew[iCell].hasNaN())
            {
                std::cout << uIncNew[iCell].transpose() << std::endl
                          << fpDivisor << std::endl
                          << iCell << std::endl;
                assert(!uIncNew[iCell].hasNaN());
            }

            // fix rho increment
            // if (u[iCell](0) + uIncNew[iCell](0) < u[iCell](0) * 1e-5)
            //     uIncNew[iCell](0) = -u[iCell](0) * (1 - 1e-5);
            uIncNew[iCell] = CompressRecPart(u[iCell], uIncNew[iCell]) - u[iCell];
        }
    }

    void EulerEvaluator::UpdateLUSGSBackward(std::vector<real> &dTau, real dt, real alphaDiag,
                                             ArrayDOF<5u> &rhs, ArrayDOF<5u> &u, ArrayDOF<5u> &uInc, ArrayDOF<5u> &uIncNew)
    {
        index nCellDist = mesh->cell2nodeLocal.dist->size();
        for (index iScan = nCellDist - 1; iScan >= 0; iScan--)
        {
            index iCell = iScan;
            iCell = (*vfv->SOR_iScan2iCell)[iCell];

            auto &c2f = mesh->cell2faceLocal[iCell];
            Eigen::Vector<real, 5> uIncNewBuf;
            uIncNewBuf.setZero(); // backward

            real fpDivisor = fv->volumeLocal[iCell] / dTau[iCell] + fv->volumeLocal[iCell] / dt;

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto &f2c = (*mesh->face2cellPair)[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                fpDivisor += (0.5 * alphaDiag) * fv->faceArea[iFace] * lambdaFace[iFace];
                if (iCellOther != FACE_2_VOL_EMPTY)
                {

                    index iScanOther = iCellOther < nCellDist
                                           ? (*vfv->SOR_iCell2iScan)[iCellOther]
                                           : iScan + 1;
                    if (iScanOther > iScan) // backward
                    {
                        Eigen::Vector<real, 5> umeanOther = u[iCellOther];
                        Eigen::Vector<real, 5> umeanOtherInc = uInc[iCellOther];
                        Eigen::Vector<real, 5> umeanOtherN = umeanOther + umeanOtherInc;
                        Eigen::Vector<real, 5> fInc;
                        do
                        {
                            Elem::tPoint unitNorm = vfv->faceNormCenter[iFace].normalized() *
                                                    (iCellAtFace ? -1 : 1); // faces out
                            Elem::tJacobi normBase = Elem::NormBuildLocalBaseV(unitNorm);
                            real p, pN, asqr, asqrN, H, HN;
                            // if (!(umeanOther(0) > 0 && umeanOtherN(0) > 0))
                            //     std::cout << umeanOther.transpose() << std::endl
                            //               << umeanOtherN.transpose() << std::endl
                            //               << uInc[iCellOther].transpose() << std::endl
                            //               << "i " << iCell << "\t" << iCellOther << std::endl;
                            // assert(umeanOther(0) > 0 && umeanOtherN(0) > 0);
                            Elem::tPoint velo = umeanOther({1, 2, 3}) / umeanOther(0);
                            Elem::tPoint veloN = umeanOtherN({1, 2, 3}) / umeanOtherN(0);
                            real vsqr = velo.squaredNorm();
                            real vsqrN = veloN.squaredNorm();

                            Gas::IdealGasThermal(umeanOther(4), umeanOther(0), vsqr,
                                                 settings.idealGasProperty.gamma, p, asqr, H);
                            Gas::IdealGasThermal(umeanOtherN(4), umeanOtherN(0), vsqrN,
                                                 settings.idealGasProperty.gamma, pN, asqrN, HN);
                            Eigen::Vector<real, 5> f, fN;

                            // linear version
                            Gas::tVec dVelo;
                            real dp;
                            Gas::IdealGasUIncrement(umeanOther, umeanOtherInc, velo, settings.idealGasProperty.gamma, dVelo, dp);
                            Gas::GasInviscidFluxFacialIncrement(umeanOther, umeanOtherInc, unitNorm, velo, dVelo, dp, p, fInc);
                            break;

                            // get to norm coord
                            umeanOther({1, 2, 3}) = normBase.transpose() * umeanOther({1, 2, 3});
                            umeanOtherN({1, 2, 3}) = normBase.transpose() * umeanOtherN({1, 2, 3});
                            velo = normBase.transpose() * velo;
                            veloN = normBase.transpose() * veloN;
                            Gas::GasInviscidFlux(umeanOther, velo, p, f);
                            Gas::GasInviscidFlux(umeanOtherN, veloN, pN, fN);
                            fInc = fN - f;
                            fInc({1, 2, 3}) = normBase * fInc({1, 2, 3}); // to xy
                            if (fInc.hasNaN() || (!fInc.allFinite()))
                            {
                                std::cout << p << "\t" << pN << std::endl
                                          << umeanOther.transpose() << std::endl
                                          << umeanOtherN.transpose() << std::endl
                                          << umeanOtherInc.transpose() << std::endl
                                          << vsqr << "\t" << vsqrN << std::endl;
                                assert(!(fInc.hasNaN() || (!fInc.allFinite())));
                            }
                        } while (false);

                        uIncNewBuf -= (0.5 * alphaDiag) * fv->faceArea[iFace] *
                                      (fInc - lambdaFace[iFace] * umeanOtherInc);
                    }
                }
            }
            uIncNewBuf /= fpDivisor;
            uIncNew[iCell] += uIncNewBuf; // backward

            // fix rho increment
            // if (u[iCell](0) + uIncNew[iCell](0) < u[iCell](0) * 1e-5)
            //     uIncNew[iCell](0) = -u[iCell](0) * (1 - 1e-5);
            uIncNew[iCell] = CompressRecPart(u[iCell], uIncNew[iCell]) - u[iCell];
        }
    }

    void EulerEvaluator::UpdateSGS(std::vector<real> &dTau, real dt, real alphaDiag,
                                   ArrayDOF<5u> &rhs, ArrayDOF<5u> &u, ArrayDOF<5u> &uInc, ArrayDOF<5u> &uIncNew, bool ifForward)
    {
        for (index iScan = 0; iScan < mesh->cell2nodeLocal.dist->size(); iScan++)
        {
            index iCell;
            if (ifForward)
                iCell = iScan;
            else
                iCell = mesh->cell2nodeLocal.dist->size() - 1 - iScan;
            auto &c2f = mesh->cell2faceLocal[iCell];
            Eigen::Vector<real, 5> uIncNewBuf;
            // uIncNewBuf.setZero(); // backward
            uIncNewBuf = fv->volumeLocal[iCell] * rhs[iCell]; // full

            real fpDivisor = fv->volumeLocal[iCell] / dTau[iCell] + fv->volumeLocal[iCell] / dt;

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto &f2c = (*mesh->face2cellPair)[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                fpDivisor += (0.5 * alphaDiag) * fv->faceArea[iFace] * lambdaFace[iFace];
                if (iCellOther != FACE_2_VOL_EMPTY)
                {
                    // if (true) // full
                    if ((ifForward && iCellOther < iCell) || ((!ifForward) && iCellOther > iCell))
                    {
                        Eigen::Vector<real, 5> umeanOther = u[iCellOther];
                        Eigen::Vector<real, 5> umeanOtherInc = uInc[iCellOther];
                        Eigen::Vector<real, 5> umeanOtherN = umeanOther + umeanOtherInc;
                        Eigen::Vector<real, 5> fInc;
                        do
                        {
                            Elem::tPoint unitNorm = vfv->faceNormCenter[iFace].normalized() *
                                                    (iCellAtFace ? -1 : 1); // faces out
                            Elem::tJacobi normBase = Elem::NormBuildLocalBaseV(unitNorm);
                            real p, pN, asqr, asqrN, H, HN;
                            // if (!(umeanOther(0) > 0 && umeanOtherN(0) > 0))
                            //     std::cout << umeanOther.transpose() << std::endl
                            //               << umeanOtherN.transpose() << std::endl
                            //               << uInc[iCellOther].transpose() << std::endl
                            //               << "i " << iCell << "\t" << iCellOther << std::endl;
                            // assert(umeanOther(0) > 0 && umeanOtherN(0) > 0);
                            Elem::tPoint velo = umeanOther({1, 2, 3}) / umeanOther(0);
                            Elem::tPoint veloN = umeanOtherN({1, 2, 3}) / umeanOtherN(0);
                            real vsqr = velo.squaredNorm();
                            real vsqrN = veloN.squaredNorm();

                            Gas::IdealGasThermal(umeanOther(4), umeanOther(0), vsqr,
                                                 settings.idealGasProperty.gamma, p, asqr, H);
                            Gas::IdealGasThermal(umeanOtherN(4), umeanOtherN(0), vsqrN,
                                                 settings.idealGasProperty.gamma, pN, asqrN, HN);
                            Eigen::Vector<real, 5> f, fN;

                            // linear version
                            Gas::tVec dVelo;
                            real dp;
                            Gas::IdealGasUIncrement(umeanOther, umeanOtherInc, velo, settings.idealGasProperty.gamma, dVelo, dp);
                            Gas::GasInviscidFluxFacialIncrement(umeanOther, umeanOtherInc, unitNorm, velo, dVelo, dp, p, fInc);
                            break;

                            // get to norm coord
                            umeanOther({1, 2, 3}) = normBase.transpose() * umeanOther({1, 2, 3});
                            umeanOtherN({1, 2, 3}) = normBase.transpose() * umeanOtherN({1, 2, 3});
                            velo = normBase.transpose() * velo;
                            veloN = normBase.transpose() * veloN;
                            Gas::GasInviscidFlux(umeanOther, velo, p, f);
                            Gas::GasInviscidFlux(umeanOtherN, veloN, pN, fN);
                            fInc = fN - f;
                            fInc({1, 2, 3}) = normBase * fInc({1, 2, 3}); // to xy
                            if (fInc.hasNaN() || (!fInc.allFinite()))
                            {
                                std::cout << p << "\t" << pN << std::endl
                                          << umeanOther.transpose() << std::endl
                                          << umeanOtherN.transpose() << std::endl
                                          << umeanOtherInc.transpose() << std::endl
                                          << vsqr << "\t" << vsqrN << std::endl;
                                assert(!(fInc.hasNaN() || (!fInc.allFinite())));
                            }
                            // std::cout << normBase << std::endl
                            //            << normBase * normBase.transpose() << std::endl;
                            // std::abort();
                        } while (false);

                        uIncNewBuf -= (0.5 * alphaDiag) * fv->faceArea[iFace] *
                                      (fInc - lambdaFace[iFace] * umeanOtherInc);
                    }
                }
            }
            uIncNewBuf /= fpDivisor;
            real relax = 1;
            uIncNew[iCell] = uIncNewBuf * relax + uInc[iCell] * (1 - relax); // full

            // fix rho increment
            if (u[iCell](0) + uIncNew[iCell](0) < u[iCell](0) * 1e-5)
                uIncNew[iCell](0) = -u[iCell](0) * (1 - 1e-5);
        }
    }

    void EulerEvaluator::FixUMaxFilter(ArrayDOF<5u> &u)
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

    void EulerEvaluator::EvaluateResidual(Eigen::Vector<real, 5> &res, ArrayDOF<5u> &rhs, index P)
    {
        if (P < 3)
        {
            Eigen::Vector<real, 5> resc;
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
            Eigen::Vector<real, 5> resc;
            resc.setZero();
            for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
                resc = resc.array().max(rhs[iCell].array().abs()).matrix();
            MPI_Allreduce(resc.data(), res.data(), res.size(), DNDS_MPI_REAL, MPI_MAX, rhs.dist->getMPI().comm);
        }
    }
}
