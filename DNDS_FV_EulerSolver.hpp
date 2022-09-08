#pragma once
#include "DNDS_Gas.hpp"
#include "DNDS_Mesh.hpp"
#include "DNDS_FV_VR.hpp"
#include "DNDS_ODE.hpp"
#include "DNDS_Scripting.hpp"
#include <iomanip>

namespace DNDS
{
    class EulerEvaluator
    {
    public:
        CompactFacedMeshSerialRW *mesh = nullptr;
        ImplicitFiniteVolume2D *fv = nullptr;
        VRFiniteVolume2D *vfv = nullptr;
        int kAv = 0;

        std::vector<real> lambdaCell;
        std::vector<real> lambdaFace;

        struct Setting
        {
            struct IdealGasProperty
            {
                real gamma = 1.4;
                real Rgas = 289;
                real muGas = 1;
                real prGas = 0.7;
                real CpGas = Rgas * gamma / (gamma - 1);
            } idealGasProperty;
            real visScale = 1;
            real visScaleIn = 1;
            real isiScale = 1;
            real isiScaleIn = 1;
            real isiCutDown = 0.5;
            real ekCutDown = 0.5;

            Eigen::Vector<real, 5> farFieldStaticValue = Eigen::Vector<real, 5>{1, 0, 0, 0, 2.5};

        } settings;

        EulerEvaluator(CompactFacedMeshSerialRW *Nmesh, ImplicitFiniteVolume2D *Nfv, VRFiniteVolume2D *Nvfv)
            : mesh(Nmesh), fv(Nfv), vfv(Nvfv), kAv(Nvfv->P_ORDER + 1)
        {
            lambdaCell.resize(mesh->cell2nodeLocal.size()); // but only dist part are used, ghost part to not judge for it in facial iter
            lambdaFace.resize(mesh->face2nodeLocal.size());
        }

        static Eigen::Vector<real, 5> CompressRecPart(const Eigen::Vector<real, 5> &umean, const Eigen::Vector<real, 5> &uRecInc)
        {
            real compressT = 0.00001;
            real eFixRatio = 0.00001;
            Eigen::Vector<real, 5> ret;

            real compress = 1.0;
            if ((umean(0) + uRecInc(0)) < umean(0) * compressT)
                compress *= umean(0) * (1 - compressT) / uRecInc(0);

            ret = umean + uRecInc * compress;

            real Ek = ret({1, 2, 3}).squaredNorm() * 0.5 / (verySmallReal + ret(0));
            real eT = eFixRatio * Ek;
            real e = ret(4) - Ek;
            if (e < 0)
                e = eT * 0.5;
            else if (e < eT)
                e = (e * e + eT * eT) / (2 * eT);
            ret(4) = e + Ek;

            return ret;
        }

        void EvaluateDt(std::vector<real> &dt,
                        ArrayLocal<VecStaticBatch<5>> &u,
                        // ArrayLocal<SemiVarMatrix<5>> &uRec,
                        real CFL, real &dtMinall, real MaxDt = 1,
                        bool UseLocaldt = false)
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
                if (f2c[1] != FACE_2_VOL_EMPTY)
                    uMean = (uMean + u[f2c[1]].p()) * 0.5;
                assert(uMean(0) > 0);
                auto veloMean = (uMean({1, 2, 3}).array() / uMean(0)).matrix();
                real veloNMean = veloMean.dot(unitNorm);

                real ekFixRatio = 0.001;
                Eigen::Vector3d velo = uMean({1, 2, 3}) / uMean(0);
                real vsqr = velo.squaredNorm();
                real Ek = vsqr * 0.5 * uMean(0);
                real Efix = Ek * ekFixRatio;
                real e = uMean(4) - Ek;
                if (e < 0)
                    e = 0.5 * Efix;
                else if (e < Efix)
                    e = (e * e + Efix * Efix) / (2 * Efix);
                uMean(4) = Ek + e;

                real pMean, asqrMean, HMean;
                Gas::IdealGasThermal(uMean(4), uMean(0), veloMean.squaredNorm(),
                                     settings.idealGasProperty.gamma,
                                     pMean, asqrMean, HMean);
                assert(asqrMean >= 0);
                real lambdaConvection = std::abs(veloNMean) + std::sqrt(asqrMean);

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
            }
            real dtMin = veryLargeReal;
            for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
                // std::cout << fv->volumeLocal[iCell] << " " << (lambdaCell[iCell]) << " " << CFL << std::endl;
                // exit(0);
                dt[iCell] = std::min(CFL * fv->volumeLocal[iCell] / (lambdaCell[iCell] + 1e-10), MaxDt);
                dtMin = std::min(dtMin, dt[iCell]);
            }

            MPI_Allreduce(&dtMin, &dtMinall, 1, DNDS_MPI_REAL, MPI_MIN, u.dist->getMPI().comm);
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
        void EvaluateRHS(ArrayDOF<5u> &rhs, ArrayDOF<5u> &u, ArrayLocal<SemiVarMatrix<5u>> &uRec)
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
                real dist;
                auto faceDiBjGaussBatchElemVR = (*vfv->faceDiBjGaussBatch)[iFace];

                if (f2c[1] != FACE_2_VOL_EMPTY)
                    dist = (vfv->cellCenters[f2c[0]] - vfv->cellCenters[f2c[1]]).norm();
                else
                    dist = (vfv->faceCenters[iFace] - vfv->cellCenters[f2c[0]]).norm() * 1;

                eFace.Integration(
                    flux,
                    [&](decltype(flux) &finc, int ig, Elem::tPoint &p, Elem::tDiFj &DiNj)
                    {
                        int nDiff = vfv->faceWeights->operator[](iFace).size();
                        Elem::tPoint unitNorm = vfv->faceNorms[iFace][ig].normalized();
                        Elem::tJacobi normBase = Elem::NormBuildLocalBaseV(unitNorm);

                        Eigen::Vector<real, 5> UL =
                            faceDiBjGaussBatchElemVR.m(ig * 2 + 0).row(0).rightCols(uRec[f2c[0]].m().rows()) *
                            uRec[f2c[0]].m();
                        // UL += u[f2c[0]]; //! do not forget the mean value
                        UL = CompressRecPart(u[f2c[0]], UL);
                        Eigen::Vector<real, 5> ULxy = UL;
                        UL({1, 2, 3}) = normBase.transpose() * UL({1, 2, 3});
                        Eigen::Vector<real, 5> UR, URxy;

                        Eigen::Matrix<real, 3, 5> GradULxy, GradURxy;
                        GradULxy.setZero(), GradURxy.setZero();
                        GradULxy({0, 1}, {0, 1, 2, 3, 4}) =
                            faceDiBjGaussBatchElemVR.m(ig * 2 + 0)({1, 2}, Eigen::seq(1, uRec[f2c[0]].m().rows() + 1 - 1)) *
                            uRec[f2c[0]].m(); // ! 2d here

                        real minVol = fv->volumeLocal[f2c[0]];

                        if (f2c[1] != FACE_2_VOL_EMPTY)
                        {
                            UR =
                                faceDiBjGaussBatchElemVR.m(ig * 2 + 1).row(0).rightCols(uRec[f2c[1]].m().rows()) *
                                uRec[f2c[1]].m();
                            // UR += u[f2c[1]];
                            UR = CompressRecPart(u[f2c[1]], UR);
                            URxy = UR;
                            UR({1, 2, 3}) = normBase.transpose() * UR({1, 2, 3});

                            GradURxy({0, 1}, {0, 1, 2, 3, 4}) =
                                faceDiBjGaussBatchElemVR.m(ig * 2 + 1)({1, 2}, Eigen::seq(1, uRec[f2c[1]].m().rows() + 1 - 1)) *
                                uRec[f2c[1]].m(); // ! 2d here

                            minVol = std::min(minVol, fv->volumeLocal[f2c[1]]);
                        }
                        else if (faceAtr.iPhy == BoundaryType::Farfield)
                        {
                            UR = settings.farFieldStaticValue;
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
                        Gas::RoeFlux_IdealGas_HartenYee(
                            UL, UR, settings.idealGasProperty.gamma, finc,
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
            for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
                rhs[iCell] = rhs[iCell].array().min(1e10).max(-1e10).matrix();
            }
        }

        /**
         * @brief to use LUSGS, use LUSGSForward(..., uInc, uInc); uInc.pull; LUSGSBackward(..., uInc, uInc);
         * the underlying logic is that for index, ghost > dist, so the forward uses no ghost,
         * and ghost should be pulled before using backward;
         * to use Jacobian instead of LUSGS, use LUSGSForward(..., uInc, uIncNew); LUSGSBackward(..., uInc, uIncNew); uIncNew.pull; uInc = uIncNew;
         * \param uIncNew overwritten;
         *
         */
        void UpdateLUSGSForward(std::vector<real> &dTau, real dt, real alphaDiag,
                                ArrayDOF<5u> &rhs, ArrayDOF<5u> &u, ArrayDOF<5u> &uInc, ArrayDOF<5u> &uIncNew)
        {
            for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
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
                    if (iCellOther != FACE_2_VOL_EMPTY)
                    {
                        fpDivisor += (0.5 * alphaDiag) * fv->faceArea[iFace] * lambdaFace[iFace];
                        if (iCellOther < iCell)
                        {
                            auto umeanOther = u[iCellOther];
                            auto umeanOtherInc = uInc[iCellOther];
                            Eigen::Vector<real, 5> umeanOtherN = umeanOther + umeanOtherInc;
                            Eigen::Vector<real, 5> fInc;
                            {
                                real p, pN, asqr, asqrN, H, HN;
                                if (!(umeanOther(0) > 0 && umeanOtherN(0) > 0))
                                    std::cout << umeanOther.transpose() << std::endl
                                              << umeanOtherN.transpose() << std::endl
                                              << uInc[iCellOther].transpose() << std::endl
                                              << "i " << iCell << "\t" << iCellOther << std::endl;
                                assert(umeanOther(0) > 0 && umeanOtherN(0) > 0);
                                auto velo = umeanOther({1, 2, 3}) / umeanOther(0);
                                auto veloN = umeanOtherN({1, 2, 3}) / umeanOtherN(0);
                                real vsqr = velo.squaredNorm();
                                real vsqrN = veloN.squaredNorm();

                                Gas::IdealGasThermal(umeanOther(4), umeanOther(0), vsqr,
                                                     settings.idealGasProperty.gamma, p, asqr, H);
                                Gas::IdealGasThermal(umeanOtherN(4), umeanOtherN(0), vsqrN,
                                                     settings.idealGasProperty.gamma, pN, asqrN, HN);
                                Eigen::Vector<real, 5> f, fN;
                                Gas::GasInviscidFlux(umeanOther, velo, p, f);
                                Gas::GasInviscidFlux(umeanOtherN, veloN, pN, fN);
                                fInc = fN - f;
                                if (fInc.hasNaN() || (!fInc.allFinite()))
                                {
                                    std::cout << p << "\t" << pN << std::endl
                                              << umeanOther.transpose() << std::endl
                                              << umeanOtherN.transpose() << std::endl
                                              << umeanOtherInc.transpose() << std::endl
                                              << vsqr << "\t" << vsqrN << std::endl;
                                    assert(!(fInc.hasNaN() || (!fInc.allFinite())));
                                }
                            }

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
                if (u[iCell](0) + uIncNew[iCell](0) < u[iCell](0) * 1e-5)
                    uIncNew[iCell](0) = -u[iCell](0) * (1 - 1e-5);
            }
        }

        /**
         * @brief
         * \param uIncNew overwritten;
         *
         */
        void UpdateLUSGSBackward(std::vector<real> &dTau, real dt, real alphaDiag,
                                 ArrayDOF<5u> &rhs, ArrayDOF<5u> &u, ArrayDOF<5u> &uInc, ArrayDOF<5u> &uIncNew)
        {
            for (index iScan = 0; iScan < mesh->cell2nodeLocal.dist->size(); iScan++)
            {
                index iCell = mesh->cell2nodeLocal.dist->size() - 1 - iScan;
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
                    if (iCellOther != FACE_2_VOL_EMPTY)
                    {
                        fpDivisor += (0.5 * alphaDiag) * fv->faceArea[iFace] * lambdaFace[iFace];
                        if (iCellOther > iCell) // backward
                        {
                            auto umeanOther = u[iCellOther];
                            auto umeanOtherInc = uInc[iCellOther];
                            Eigen::Vector<real, 5> umeanOtherN = umeanOther + umeanOtherInc;
                            Eigen::Vector<real, 5> fInc;
                            {
                                real p, pN, asqr, asqrN, H, HN;
                                assert(umeanOther(0) > 0 && umeanOtherN(0) > 0);
                                auto velo = umeanOther({1, 2, 3}) / umeanOther(0);
                                auto veloN = umeanOtherN({1, 2, 3}) / umeanOtherN(0);
                                real vsqr = velo.squaredNorm();
                                real vsqrN = veloN.squaredNorm();

                                Gas::IdealGasThermal(umeanOther(4), umeanOther(0), vsqr,
                                                     settings.idealGasProperty.gamma, p, asqr, H);
                                Gas::IdealGasThermal(umeanOtherN(4), umeanOtherN(0), vsqrN,
                                                     settings.idealGasProperty.gamma, pN, asqrN, HN);
                                Eigen::Vector<real, 5> f, fN;
                                Gas::GasInviscidFlux(umeanOther, velo, p, f);
                                Gas::GasInviscidFlux(umeanOtherN, veloN, pN, fN);
                                fInc = fN - f;
                            }

                            uIncNewBuf -= (0.5 * alphaDiag) * fv->faceArea[iFace] *
                                          (fInc - lambdaFace[iFace] * umeanOtherInc);
                        }
                    }
                }
                uIncNewBuf /= fpDivisor;
                uIncNew[iCell] += uIncNewBuf; // backward

                // fix rho increment
                if (u[iCell](0) + uIncNew[iCell](0) < u[iCell](0) * 1e-5)
                    uIncNew[iCell](0) = -u[iCell](0) * (1 - 1e-5);
            }
        }

        void UpdateSGS(std::vector<real> &dTau, real dt, real alphaDiag,
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
                    if (iCellOther != FACE_2_VOL_EMPTY)
                    {
                        fpDivisor += (0.5 * alphaDiag) * fv->faceArea[iFace] * lambdaFace[iFace];
                        if (true) // full
                        {
                            auto umeanOther = u[iCellOther];
                            auto umeanOtherInc = uInc[iCellOther];
                            Eigen::Vector<real, 5> umeanOtherN = umeanOther + umeanOtherInc;
                            Eigen::Vector<real, 5> fInc;
                            {
                                real p, pN, asqr, asqrN, H, HN;
                                assert(umeanOther(0) > 0 && umeanOtherN(0) > 0);
                                auto velo = umeanOther({1, 2, 3}) / umeanOther(0);
                                auto veloN = umeanOtherN({1, 2, 3}) / umeanOtherN(0);
                                real vsqr = velo.squaredNorm();
                                real vsqrN = veloN.squaredNorm();

                                Gas::IdealGasThermal(umeanOther(4), umeanOther(0), vsqr,
                                                     settings.idealGasProperty.gamma, p, asqr, H);
                                Gas::IdealGasThermal(umeanOtherN(4), umeanOtherN(0), vsqrN,
                                                     settings.idealGasProperty.gamma, pN, asqrN, HN);
                                Eigen::Vector<real, 5> f, fN;
                                Gas::GasInviscidFlux(umeanOther, velo, p, f);
                                Gas::GasInviscidFlux(umeanOtherN, veloN, pN, fN);
                                fInc = fN - f;
                            }

                            uIncNewBuf -= (0.5 * alphaDiag) * fv->faceArea[iFace] *
                                          (fInc - lambdaFace[iFace] * umeanOtherInc);
                        }
                    }
                }
                uIncNewBuf /= fpDivisor;
                uIncNew[iCell] = uIncNewBuf; // full

                // fix rho increment
                if (u[iCell](0) + uIncNew[iCell](0) < u[iCell](0) * 1e-5)
                    uIncNew[iCell](0) = -u[iCell](0) * (1 - 1e-5);
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

        void EvaluateResidual(Eigen::Vector<real, 5> &res, ArrayDOF<5u> &rhs, index P = 1)
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

        void FixUMaxFilter(ArrayDOF<5u> &u)
        {
            // TODO: make spacial filter jacobian
            real scaleRhoCutoff = 0.01;
            real scaleEInternalCutOff = 0.01;
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
    };

    class EulerSolver
    {
        MPIInfo mpi;
        std::shared_ptr<CompactFacedMeshSerialRW> mesh;
        std::shared_ptr<ImplicitFiniteVolume2D> fv;
        std::shared_ptr<VRFiniteVolume2D> vfv;

        ArrayDOF<5u> u, uPoisson, uInc;
        ArrayLocal<SemiVarMatrix<5u>> uRec, uRecNew;

        static const int nOUTS = 8;
        // rho u v w p T M ifUseLimiter
        std::shared_ptr<Array<VecStaticBatch<nOUTS>>> outDist;
        std::shared_ptr<Array<VecStaticBatch<nOUTS>>> outSerial;

        ArrayLocal<SemiVarMatrix<5u>> uF0, uF1;
        // std::vector<uint32_t> ifUseLimiter;
        std::vector<real> ifUseLimiter;

    public:
        EulerSolver(const MPIInfo &nmpi) : mpi(nmpi)
        {
        }

        struct Configuration
        {
            int recOrder = 2;
            int nInternalRecStep = 1;
            int nTimeStep = 1000;
            int nConsoleCheck = 10;
            int nConsoleCheckInternal = 1;
            int nDataOut = 10000;
            int nDataOutC = 50;
            int nDataOutInternal = 1;
            int nDataOutCInternal = 1;
            int nTimeStepInternal = 1000;
            real tDataOut = veryLargeReal;
            real tEnd = veryLargeReal;

            real CFL = 0.5;
            real dtImplicit = 1e100;

            real meshRotZ = 0;
            std::string mName = "data/mesh/NACA0012_WIDE_H3.msh";
            std::string outPltName = "data/out/debugData_";
            std::string outLogName = "data/out/debugData_";
            real err_dMax = 0.1;

            real res_base = 0;

            VRFiniteVolume2D::Setting vfvSetting;

            int nDropVisScale;
            real vDropVisScale;
            EulerEvaluator::Setting eulerSetting;

            int curvilinearOneStep = 500;
            int curvilinearRepeatInterval = 500;
            int curvilinearRepeatNum = 10;

            int curvilinearRestartNstep = 100;
            real curvilinearRange = 0.1;

            bool useLocalDt = true;
            int nForceLocalStartStep = -1;
            int nCFLRampStart = 1000;
            int nCFLRampLength = 10000;
            real CFLRampEnd = 10;
        } config;

        void ConfigureFromJson(const std::string &jsonName)
        {
            rapidjson::Document doc;
            JSON::ReadFile(jsonName, doc);
            JSON::ParamParser root(mpi);

            root.AddInt("nInternalRecStep", &config.nInternalRecStep);
            root.AddInt("recOrder", &config.recOrder);
            root.AddInt("nTimeStep", &config.nTimeStep);
            root.AddInt("nTimeStepInternal", &config.nTimeStepInternal);
            root.AddInt("nConsoleCheck", &config.nConsoleCheck);
            root.AddInt("nConsoleCheckInternal", &config.nConsoleCheckInternal);
            root.AddInt("nDataOutC", &config.nDataOutC);
            root.AddInt("nDataOut", &config.nDataOut);
            root.AddInt("nDataOutCInternal", &config.nDataOutCInternal);
            root.AddInt("nDataOutInternal", &config.nDataOutInternal);
            root.AddDNDS_Real("tDataOut", &config.tDataOut);
            root.AddDNDS_Real("tEnd", &config.tEnd);
            root.AddDNDS_Real("CFL", &config.CFL);
            root.AddDNDS_Real("dtImplicit", &config.dtImplicit);
            root.AddDNDS_Real("meshRotZ", &config.meshRotZ);
            root.Addstd_String("meshFile", &config.mName);
            root.Addstd_String("outLogName", &config.outLogName);
            root.Addstd_String("outPltName", &config.outPltName);
            root.AddDNDS_Real("err_dMax", &config.err_dMax);
            root.AddDNDS_Real("res_base", &config.res_base);
            root.AddBool("useLocalDt", &config.useLocalDt);
            root.AddInt("nForceLocalStartStep", &config.nForceLocalStartStep);

            root.AddInt("nCFLRampStart", &config.nCFLRampStart);
            root.AddInt("nCFLRampLength", &config.nCFLRampLength);
            root.AddDNDS_Real("CFLRampEnd", &config.CFLRampEnd);

            JSON::ParamParser vfvParser(mpi);
            root.AddObject("vfvSetting", &vfvParser);
            {
                vfvParser.AddBool("SOR_Instead", &config.vfvSetting.SOR_Instead);
                vfvParser.AddBool("SOR_InverseScanning", &config.vfvSetting.SOR_InverseScanning);
                vfvParser.AddBool("SOR_RedBlack", &config.vfvSetting.SOR_RedBlack);
                vfvParser.AddDNDS_Real("JacobiRelax", &config.vfvSetting.JacobiRelax);
                vfvParser.AddDNDS_Real("tangWeight", &config.vfvSetting.tangWeight);
                vfvParser.AddBool("anistropicLengths", &config.vfvSetting.anistropicLengths);
                vfvParser.AddDNDS_Real("scaleMLargerPortion", &config.vfvSetting.scaleMLargerPortion);
                vfvParser.AddDNDS_Real("farWeight", &config.vfvSetting.farWeight);
                vfvParser.AddDNDS_Real("wallWeight", &config.vfvSetting.wallWeight);
                vfvParser.AddInt("curvilinearOrder", &config.vfvSetting.curvilinearOrder);
                vfvParser.AddDNDS_Real("WBAP_SmoothIndicatorScale", &config.vfvSetting.WBAP_SmoothIndicatorScale);
            }

            root.AddInt("nDropVisScale", &config.nDropVisScale);
            root.AddDNDS_Real("vDropVisScale", &config.vDropVisScale);

            JSON::ParamParser eulerParser(mpi);
            root.AddObject("eulerSetting", &eulerParser);
            {
                eulerParser.AddDNDS_Real("visScale", &config.eulerSetting.visScale);
                eulerParser.AddDNDS_Real("visScaleIn", &config.eulerSetting.visScaleIn);
                eulerParser.AddDNDS_Real("ekCutDown", &config.eulerSetting.ekCutDown);
                eulerParser.AddDNDS_Real("isiScale", &config.eulerSetting.isiScale);
                eulerParser.AddDNDS_Real("isiScaleIn", &config.eulerSetting.isiScaleIn);
                eulerParser.AddDNDS_Real("isiCutDown", &config.eulerSetting.isiCutDown);
                eulerParser.AddDNDS_Real("visScale", &config.eulerSetting.visScale);
            }
            JSON::ParamParser eulerGasParser(mpi);
            {
                eulerParser.AddObject("idealGasProperty", &eulerGasParser);
                {
                    eulerGasParser.AddDNDS_Real("gamma", &config.eulerSetting.idealGasProperty.gamma);
                    eulerGasParser.AddDNDS_Real("Rgas", &config.eulerSetting.idealGasProperty.Rgas);
                    eulerGasParser.AddDNDS_Real("muGas", &config.eulerSetting.idealGasProperty.muGas);
                }
            }
            Eigen::VectorXd eulerSetting_farFieldStaticValueBuf;
            {
                eulerParser.AddEigen_RealVec("farFieldStaticValue", &eulerSetting_farFieldStaticValueBuf);
            }
            root.AddInt("curvilinearOneStep", &config.curvilinearOneStep);
            root.AddInt("curvilinearRestartNstep", &config.curvilinearRestartNstep);
            root.AddInt("curvilinearRepeatInterval", &config.curvilinearRepeatInterval);
            root.AddInt("curvilinearRepeatNum", &config.curvilinearRepeatNum);
            root.AddDNDS_Real("curvilinearRange", &config.curvilinearRange);

            root.Parse(doc.GetObject(), 0);
            assert(eulerSetting_farFieldStaticValueBuf.size() == 5);
            config.eulerSetting.farFieldStaticValue = eulerSetting_farFieldStaticValueBuf;
            if (mpi.rank == 0)
                log() << "JSON: Parse Done" << std::endl;

            if (doc["vfvSetting"].IsObject())
            {
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

            fv->BuildMean(u);
            fv->BuildMean(uPoisson);
            fv->BuildMean(uInc);
            vfv->BuildRec(uRec);
            vfv->BuildRecFacial(uF0);

            uRecNew.Copy(uRec);
            uF1.Copy(uF0);
            uF1.InitPersistentPullClean();

            // vfv->BuildRecFacial(uF1);//! why copy is bad ???
            // vfv->BuildRec(uRecNew);

            u.setConstant(config.eulerSetting.farFieldStaticValue);
            uPoisson.setConstant(0.0);

            outDist = std::make_shared<decltype(outDist)::element_type>(
                decltype(outDist)::element_type::tContext(mesh->cell2faceLocal.dist->size()), mpi);
            outSerial = std::make_shared<decltype(outDist)::element_type>(outDist.get());
            outSerial->BorrowGGIndexing(*mesh->cell2node);
            outSerial->createMPITypes();
            outSerial->initPersistentPull();

            // //Box
            // for (index iCell = 0; iCell < u.dist->size(); iCell++)
            // {
            //     auto pos = vfv->cellBaries[iCell];
            //     if (pos(0) < (0.75 + 1e-5) && pos(0) > (0.25 - 1e-5) && pos(1) < (0.75 + 1e-5) && pos(1) > (0.25 - 1e-5))
            //     {
            //         u[iCell] = Eigen::Vector<real, 5>{1, 0, 0, 0, 20};
            //     }
            // }

            ifUseLimiter.resize(u.size());
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
            EulerEvaluator eval(mesh.get(), fv.get(), vfv.get());
            std::ofstream logErr(config.outLogName + ".log");
            eval.settings = config.eulerSetting;
            // std::cout << uF0.dist->commStat.hasPersistentPullReqs << std::endl;
            // exit(0);
            // uRec.InitPersistentPullClean();
            // u.InitPersistentPullClean();
            // uInc.InitPersistentPullClean();
            // uF0.InitPersistentPullClean();
            // u.StartPersistentPullClean();
            double tstart = MPI_Wtime();
            double trec{0}, tcomm{0}, trhs{0}, tLim{0};
            int stepCount = 0;
            Eigen::Vector<real, 5> resBaseC;
            resBaseC.setConstant(config.res_base);

            // Doing Poisson Init:

            int curvilinearNum = 0;
            int curvilinearStepper = 0;

            real tsimu = 0.0;
            real nextTout = 0.0;
            int nextStepOut = config.nDataOut;
            int nextStepOutC = config.nDataOutC;
            PerformanceTimer::Instance().clearAllTimer();
            real CFLNow = config.CFL;
            for (int step = 1; step <= config.nTimeStep; step++)
            {

                if (step == config.nForceLocalStartStep)
                    config.useLocalDt = true;
                if (step == config.nDropVisScale)
                    eval.settings.visScale *= config.vDropVisScale;
                bool ifOutT = false;
                real curDtMin;
                curvilinearStepper++;
                ode.Step(
                    u,
                    [&](ArrayDOF<5u> &crhs, ArrayDOF<5u> &cx)
                    {
                        eval.FixUMaxFilter(u);
                        u.StartPersistentPullClean();
                        u.WaitPersistentPullClean();

                        for (int iRec = 0; iRec < config.nInternalRecStep; iRec++)
                        {
                            double tstartA = MPI_Wtime();
                            vfv->ReconstructionJacobiStep(cx, uRec, uRecNew);
                            trec += MPI_Wtime() - tstartA;

                            uRec.StartPersistentPullClean();
                            uRec.WaitPersistentPullClean();
                        }
                        double tstartH = MPI_Wtime();

                        vfv->ReconstructionWBAPLimitFacial(
                            cx, uRec, uRec, uF0, uF1, ifUseLimiter,
                            [&](const auto &UL, const auto &UR, const auto &n) -> auto{
                                Eigen::Vector<real, 5> UC = (UL + UR) * 0.5;
                                auto normBase = Elem::NormBuildLocalBaseV(n);
                                UC({1, 2, 3}) = normBase.transpose() * UC({1, 2, 3});

                                real ekFixRatio = 0.001;
                                Eigen::Vector3d velo = UC({1, 2, 3}) / UC(0);
                                real vsqr = velo.squaredNorm();
                                real Ek = vsqr * 0.5 * UC(0);
                                real Efix = Ek * ekFixRatio;
                                real e = UC(4) - Ek;
                                if (e < 0)
                                    e = 0.5 * Efix;
                                else if (e < Efix)
                                    e = (e * e + Efix * Efix) / (2 * Efix);
                                UC(4) = Ek + e;

                                // return Gas::IdealGas_EulerGasLeftEigenVector(UC, eval.settings.idealGasProperty.gamma);
                                return Eigen::Matrix<real, 5, 5>::Identity();
                            },
                            [&](const auto &UL, const auto &UR, const auto &n) -> auto{
                                Eigen::Vector<real, 5> UC = (UL + UR) * 0.5;
                                auto normBase = Elem::NormBuildLocalBaseV(n);
                                UC({1, 2, 3}) = normBase.transpose() * UC({1, 2, 3});

                                real ekFixRatio = 0.001;
                                Eigen::Vector3d velo = UC({1, 2, 3}) / UC(0);
                                real vsqr = velo.squaredNorm();
                                real Ek = vsqr * 0.5 * UC(0);
                                real Efix = Ek * ekFixRatio;
                                real e = UC(4) - Ek;
                                if (e < 0)
                                    e = 0.5 * Efix;
                                else if (e < Efix)
                                    e = (e * e + Efix * Efix) / (2 * Efix);
                                UC(4) = Ek + e;

                                // return Gas::IdealGas_EulerGasRightEigenVector(UC, eval.settings.idealGasProperty.gamma);
                                return Eigen::Matrix<real, 5, 5>::Identity();
                            });
                        tLim += MPI_Wtime() - tstartH;

                        uRec.StartPersistentPullClean(); //! this also need to update!
                        uRec.WaitPersistentPullClean();

                        double tstartE = MPI_Wtime();
                        eval.EvaluateRHS(crhs, cx, uRec);
                        trhs += MPI_Wtime() - tstartE;
                    },
                    [&](std::vector<real> &dt)
                    {
                        eval.FixUMaxFilter(u);
                        u.StartPersistentPullClean(); //! this also need to update!
                        u.WaitPersistentPullClean();
                        uRec.StartPersistentPullClean();
                        uRec.WaitPersistentPullClean();

                        eval.EvaluateDt(dt, u, CFLNow, curDtMin, 1e100, config.useLocalDt);
                        if (curDtMin + tsimu > nextTout)
                            curDtMin = nextTout - tsimu, ifOutT = true;
                        if (!config.useLocalDt)
                            for (auto &dti : dt)
                                dti = curDtMin;
                    });
                // std::cout << "A\n"
                //           << std::setprecision(15)
                //           << u[12279].transpose() << "\n"
                //           << u[12280].transpose() << std::endl;
                tsimu += curDtMin;
                if (ifOutT)
                    tsimu = nextTout;
                Eigen::Vector<real, 5> res;
                eval.EvaluateResidual(res, ode.rhsbuf[0]);
                if (stepCount == 0 && resBaseC.norm() == 0)
                    resBaseC = res;

                if (step % config.nConsoleCheck == 0)
                {
                    double telapsed = MPI_Wtime() - tstart;
                    if (mpi.rank == 0)
                    {
                        tcomm = PerformanceTimer::Instance().getTimer(PerformanceTimer::Comm);
                        auto fmt = log().flags();
                        log() << std::setprecision(15) << std::scientific
                              << "=== Step [" << step << "]   "
                              << "res \033[91m[" << (res.array() / resBaseC.array()).transpose() << "]\033[39m   "
                              << "t,dt(min) \033[92m[" << tsimu << ", " << curDtMin << "]\033[39m   "
                              << std::setprecision(3) << std::fixed
                              << "Time [" << telapsed << "]   recTime [" << trec << "]   rhsTime [" << trhs << "]   commTime [" << tcomm << "]  limTime [" << tLim << "]  " << std::endl;
                        log().setf(fmt);
                        logErr << step << "\t" << std::setprecision(9) << std::scientific
                               << (res.array() / resBaseC.array()).transpose() << " "
                               << tsimu << " " << curDtMin << std::endl;
                    }
                    tstart = MPI_Wtime();
                    trec = tcomm = trhs = tLim = 0.;
                    PerformanceTimer::Instance().clearAllTimer();
                }
                if (step == nextStepOut)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + std::to_string(step) + ".plt", ode);
                    nextStepOut += config.nDataOut;
                }
                if (step == nextStepOutC)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + "C" + ".plt", ode);
                    nextStepOutC += config.nDataOutC;
                }
                if (ifOutT)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + "t_" + std::to_string(nextTout) + ".plt", ode);
                    nextTout += config.tDataOut;
                    if (nextTout > config.tEnd)
                        nextTout = config.tEnd;
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

                if (tsimu == config.tEnd)
                    break;
            }

            // u.WaitPersistentPullClean();
            logErr.close();
        }

        void RunImplicitEuler()
        {

            ODE::ImplicitEulerDualTimeStep<decltype(u)> ode(
                u.dist->size(),
                [&](decltype(u) &data)
                {
                    data.resize(u.dist->size(), u.dist->getMPI());
                    data.CreateGhostCopyComm(mesh->cell2faceLocal);
                });
            EulerEvaluator eval(mesh.get(), fv.get(), vfv.get());
            std::ofstream logErr(config.outLogName + ".log");
            eval.settings = config.eulerSetting;
            // std::cout << uF0.dist->commStat.hasPersistentPullReqs << std::endl;
            // exit(0);
            // uRec.InitPersistentPullClean();
            // u.InitPersistentPullClean();
            // uInc.InitPersistentPullClean();
            // uF0.InitPersistentPullClean();
            // u.StartPersistentPullClean();
            double tstart = MPI_Wtime();
            double trec{0}, tcomm{0}, trhs{0}, tLim{0};
            int stepCount = 0;
            Eigen::Vector<real, 5> resBaseC;
            Eigen::Vector<real, 5> resBaseCInternal;
            resBaseC.setConstant(config.res_base);

            // Doing Poisson Init:

            int curvilinearNum = 0;
            int curvilinearStepper = 0;

            real tsimu = 0.0;
            real nextTout = config.tDataOut;
            int nextStepOut = config.nDataOut;
            int nextStepOutC = config.nDataOutC;
            PerformanceTimer::Instance().clearAllTimer();
            real CFLNow = config.CFL;
            for (int step = 1; step <= config.nTimeStep; step++)
            {

                bool ifOutT = false;
                real curDtMin;
                real curDtImplicit = config.dtImplicit;
                if (tsimu + curDtImplicit > nextTout)
                {
                    ifOutT = true;
                    curDtImplicit = (nextTout - tsimu);
                }
                CFLNow = config.CFL;
                ode.Step(
                    u, uInc,
                    [&](ArrayDOF<5u> &crhs, ArrayDOF<5u> &cx)
                    {
                        eval.FixUMaxFilter(cx);
                        cx.StartPersistentPullClean();
                        cx.WaitPersistentPullClean();

                        for (int iRec = 0; iRec < config.nInternalRecStep; iRec++)
                        {
                            double tstartA = MPI_Wtime();
                            vfv->ReconstructionJacobiStep(cx, uRec, uRecNew);
                            trec += MPI_Wtime() - tstartA;

                            uRec.StartPersistentPullClean();
                            uRec.WaitPersistentPullClean();
                        }
                        double tstartH = MPI_Wtime();

                        vfv->ReconstructionWBAPLimitFacial(
                            cx, uRec, uRec, uF0, uF1, ifUseLimiter,
                            [&](const auto &UL, const auto &UR, const auto &n) -> auto{
                                Eigen::Vector<real, 5> UC = (UL + UR) * 0.5;
                                auto normBase = Elem::NormBuildLocalBaseV(n);
                                UC({1, 2, 3}) = normBase.transpose() * UC({1, 2, 3});

                                real ekFixRatio = 0.001;
                                Eigen::Vector3d velo = UC({1, 2, 3}) / UC(0);
                                real vsqr = velo.squaredNorm();
                                real Ek = vsqr * 0.5 * UC(0);
                                real Efix = Ek * ekFixRatio;
                                real e = UC(4) - Ek;
                                if (e < 0)
                                    e = 0.5 * Efix;
                                else if (e < Efix)
                                    e = (e * e + Efix * Efix) / (2 * Efix);
                                UC(4) = Ek + e;

                                // return Gas::IdealGas_EulerGasLeftEigenVector(UC, eval.settings.idealGasProperty.gamma);
                                return Eigen::Matrix<real, 5, 5>::Identity();
                            },
                            [&](const auto &UL, const auto &UR, const auto &n) -> auto{
                                Eigen::Vector<real, 5> UC = (UL + UR) * 0.5;
                                auto normBase = Elem::NormBuildLocalBaseV(n);
                                UC({1, 2, 3}) = normBase.transpose() * UC({1, 2, 3});

                                real ekFixRatio = 0.001;
                                Eigen::Vector3d velo = UC({1, 2, 3}) / UC(0);
                                real vsqr = velo.squaredNorm();
                                real Ek = vsqr * 0.5 * UC(0);
                                real Efix = Ek * ekFixRatio;
                                real e = UC(4) - Ek;
                                if (e < 0)
                                    e = 0.5 * Efix;
                                else if (e < Efix)
                                    e = (e * e + Efix * Efix) / (2 * Efix);
                                UC(4) = Ek + e;

                                // return Gas::IdealGas_EulerGasRightEigenVector(UC, eval.settings.idealGasProperty.gamma);
                                return Eigen::Matrix<real, 5, 5>::Identity();
                            });
                        tLim += MPI_Wtime() - tstartH;

                        uRec.StartPersistentPullClean(); //! this also need to update!
                        uRec.WaitPersistentPullClean();

                        double tstartE = MPI_Wtime();
                        eval.EvaluateRHS(crhs, cx, uRec);
                        trhs += MPI_Wtime() - tstartE;
                    },
                    [&](std::vector<real> &dTau)
                    {
                        eval.FixUMaxFilter(u);
                        u.StartPersistentPullClean(); //! this also need to update!
                        u.WaitPersistentPullClean();
                        uRec.StartPersistentPullClean();
                        uRec.WaitPersistentPullClean();

                        eval.EvaluateDt(dTau, u, CFLNow, curDtMin, 1e100, true);
                    },
                    [&](ArrayDOF<5u> &cx, ArrayDOF<5u> &crhs, std::vector<real> &dTau,
                        real dt, real alphaDiag, ArrayDOF<5u> &cxInc)
                    {
                        eval.UpdateLUSGSForward(dTau, dt, alphaDiag, crhs, cx, cxInc, cxInc);
                        cxInc.StartPersistentPullClean();
                        cxInc.WaitPersistentPullClean();
                        eval.UpdateLUSGSBackward(dTau, dt, alphaDiag, crhs, cx, cxInc, cxInc);
                        for (int iIter = 1; iIter <= 0; iIter++)
                        {
                            cxInc.StartPersistentPullClean();
                            cxInc.WaitPersistentPullClean();
                            eval.UpdateSGS(dTau, dt, alphaDiag, crhs, cx, cxInc, cxInc, true);
                            cxInc.StartPersistentPullClean();
                            cxInc.WaitPersistentPullClean();
                            eval.UpdateSGS(dTau, dt, alphaDiag, crhs, cx, cxInc, cxInc, false);
                        }
                    },
                    config.nTimeStepInternal,
                    [&](int iter, ArrayDOF<5u> &cxinc) -> bool
                    {
                        Eigen::Vector<real, 5> res;
                        eval.EvaluateResidual(res, cxinc);
                        if (iter == 1)
                            resBaseCInternal = res;
                        Eigen::Vector<real, 5> resRel = (res.array() / resBaseCInternal.array()).matrix();
                        if (iter % config.nConsoleCheckInternal == 0)
                        {
                            double telapsed = MPI_Wtime() - tstart;
                            if (mpi.rank == 0)
                            {
                                tcomm = PerformanceTimer::Instance().getTimer(PerformanceTimer::Comm);
                                auto fmt = log().flags();
                                log() << std::setprecision(3) << std::scientific
                                      << "\t Internal === Step [" << iter << "]   "
                                      << "res \033[91m[" << resRel.transpose() << "]\033[39m   "
                                      << "t,dTaumin,CFL \033[92m[" << tsimu << ", " << curDtMin << ", " << CFLNow << "]\033[39m   "
                                      << std::setprecision(3) << std::fixed
                                      << "Time [" << telapsed << "]   recTime [" << trec << "]   rhsTime [" << trhs << "]   commTime [" << tcomm << "]  limTime [" << tLim << "]  " << std::endl;
                                log().setf(fmt);
                                logErr << step << "\t" << iter << "\t" << std::setprecision(9) << std::scientific
                                       << resRel << " "
                                       << tsimu << " " << curDtMin << std::endl;
                            }
                            tstart = MPI_Wtime();
                            trec = tcomm = trhs = tLim = 0.;
                            PerformanceTimer::Instance().clearAllTimer();
                        }

                        if (iter % config.nDataOutInternal == 0)
                        {
                            eval.FixUMaxFilter(u);
                            PrintData(config.outPltName + std::to_string(step) + "_" + std::to_string(iter) + ".plt", ode);
                            nextStepOut += config.nDataOut;
                        }
                        if (iter % config.nDataOutCInternal == 0)
                        {
                            eval.FixUMaxFilter(u);
                            PrintData(config.outPltName + "C" + ".plt", ode);
                            nextStepOutC += config.nDataOutC;
                        }
                        if (iter >= config.nCFLRampStart && iter <= config.nCFLRampLength + config.nCFLRampStart)
                        {
                            real inter = real(iter - config.nCFLRampStart) / config.nCFLRampLength;
                            real logCFL = std::log(config.CFL) + (std::log(config.CFLRampEnd / config.CFL) * inter);
                            CFLNow = std::exp(logCFL);
                        }

                        return resRel.maxCoeff() < 1e-20;
                    },
                    curDtImplicit + verySmallReal);

                tsimu += curDtImplicit;
                if (ifOutT)
                    tsimu = nextTout;
                Eigen::Vector<real, 5> res;
                eval.EvaluateResidual(res, ode.rhsbuf[0]);
                if (stepCount == 0 && resBaseC.norm() == 0)
                    resBaseC = res;

                if (step % config.nConsoleCheck == 0)
                {
                    double telapsed = MPI_Wtime() - tstart;
                    if (mpi.rank == 0)
                    {
                        tcomm = PerformanceTimer::Instance().getTimer(PerformanceTimer::Comm);
                        auto fmt = log().flags();
                        log() << std::setprecision(3) << std::scientific
                              << "=== Step [" << step << "]   "
                              << "res \033[91m[" << (res.array() / resBaseC.array()).transpose() << "]\033[39m   "
                              << "t,dt(min) \033[92m[" << tsimu << ", " << curDtMin << "]\033[39m   "
                              << std::setprecision(3) << std::fixed
                              << "Time [" << telapsed << "]   recTime [" << trec << "]   rhsTime [" << trhs << "]   commTime [" << tcomm << "]  limTime [" << tLim << "]  " << std::endl;
                        log().setf(fmt);
                        logErr << step << "\t" << std::setprecision(9) << std::scientific
                               << (res.array() / resBaseC.array()).transpose() << " "
                               << tsimu << " " << curDtMin << std::endl;
                    }
                    tstart = MPI_Wtime();
                    trec = tcomm = trhs = tLim = 0.;
                    PerformanceTimer::Instance().clearAllTimer();
                }
                if (step == nextStepOut)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + std::to_string(step) + ".plt", ode);
                    nextStepOut += config.nDataOut;
                }
                if (step == nextStepOutC)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + "C" + ".plt", ode);
                    nextStepOutC += config.nDataOutC;
                }
                if (ifOutT)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + "t_" + std::to_string(nextTout) + ".plt", ode);
                    nextTout += config.tDataOut;
                    if (nextTout > config.tEnd)
                        nextTout = config.tEnd;
                }

                stepCount++;

                if (tsimu >= config.tEnd)
                    break;
            }

            // u.WaitPersistentPullClean();
            logErr.close();
        }

        template <typename tODE>
        void PrintData(const std::string &fname, tODE &ode)
        {

            for (int iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
                Eigen::Vector<real, 5> recu =
                    vfv->cellDiBjCenterBatch->operator[](iCell).m(0)({0}, Eigen::all).rightCols(uRec[iCell].m().rows()) *
                    uRec[iCell].m();
                // recu += u[iCell];
                // assert(recu(0) > 0);
                recu = EulerEvaluator::CompressRecPart(u[iCell], recu);
                Gas::tVec velo = (recu({1, 2, 3}).array() / recu(0)).matrix();
                real vsqr = velo.squaredNorm();
                real asqr, p, H;
                Gas::IdealGasThermal(recu(4), recu(0), vsqr, config.eulerSetting.idealGasProperty.gamma, p, asqr, H);
                assert(asqr > 0);
                real M = std::sqrt(vsqr / asqr);
                real T = p / recu(0) / config.eulerSetting.idealGasProperty.Rgas;

                (*outDist)[iCell][0] = recu(0);
                (*outDist)[iCell][1] = velo(0);
                (*outDist)[iCell][2] = velo(1);
                (*outDist)[iCell][3] = velo(2);
                (*outDist)[iCell][4] = p;
                (*outDist)[iCell][5] = T;
                (*outDist)[iCell][6] = M;
                // (*outDist)[iCell][7] = (bool)(ifUseLimiter[iCell] & 0x0000000FU);
                (*outDist)[iCell][7] = ifUseLimiter[iCell] / config.vfvSetting.WBAP_SmoothIndicatorScale;
            }
            outSerial->startPersistentPull();
            outSerial->waitPersistentPull();
            const static std::vector<std::string> names{
                "R", "U", "V", "W", "P", "T", "M", "ifUseLimiter"};
            mesh->PrintSerialPartPltBinaryDataArray(
                fname, 0, nOUTS, //! oprank = 0
                [&](int idata)
                { return names[idata]; },
                [&](int idata, index iv)
                {
                    return (*outSerial)[iv][idata];
                },
                0);
        }
    };

}