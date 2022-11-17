#pragma once
#include "DNDS_Gas.hpp"
#include "DNDS_Mesh.hpp"
#include "DNDS_FV_VR.hpp"
#include "DNDS_ODE.hpp"
#include "DNDS_Scripting.hpp"
#include "DNDS_Linear.hpp"
#include <iomanip>
#include <functional>

namespace DNDS
{
    class EulerEvaluator
    {
        int nVars = 5;
    public:
        CompactFacedMeshSerialRW *mesh = nullptr;
        ImplicitFiniteVolume2D *fv = nullptr;
        VRFiniteVolume2D *vfv = nullptr;
        int kAv = 0;

        

        std::vector<real> lambdaCell;
        std::vector<real> lambdaFace;
        std::vector<real> lambdaFaceVis;
        std::vector<real> deltaLambdaFace;
        std::vector<Eigen::Matrix<real, 10, 5>> dFdUFace;
        std::vector<Eigen::Matrix<real, 10, 5>> jacobianFace;
        std::vector<Eigen::Matrix<real, 5, 5>> jacobianCell;
        std::vector<Eigen::Matrix<real, 5, 5>> jacobianCellInv;

        std::vector<std::vector<real>> dWall;

        // ArrayVDOF<25> dRdUrec;
        // ArrayVDOF<25> dRdb;

        struct Setting
        {
            enum RiemannSolverType
            {
                Roe = 1,
                HLLC = 2,
                HLLEP = 3
            } rsType = Roe;
            struct IdealGasProperty
            {
                real gamma = 1.4;
                real Rgas = 289;
                real muGas = 1;
                real prGas = 0.7;
                real CpGas = Rgas * gamma / (gamma - 1);
            } idealGasProperty;

            int nTimeFilterPass = 0;

            real visScale = 1;
            real visScaleIn = 1;
            real isiScale = 1;
            real isiScaleIn = 1;
            real isiCutDown = 0.5;
            real ekCutDown = 0.5;

            Eigen::Vector<real, 5> farFieldStaticValue = Eigen::Vector<real, 5>{1, 0, 0, 0, 2.5};

            struct BoxInitializer
            {
                real x0, x1, y0, y1, z0, z1;
                Eigen::Vector<real, 5> v;
            };
            std::vector<BoxInitializer> boxInitializers;

            struct PlaneInitializer
            {
                real a, b, c, h;
                Eigen::Vector<real, 5> v;
            };
            std::vector<PlaneInitializer> planeInitializers;

        } settings;

        EulerEvaluator(CompactFacedMeshSerialRW *Nmesh, ImplicitFiniteVolume2D *Nfv, VRFiniteVolume2D *Nvfv)
            : mesh(Nmesh), fv(Nfv), vfv(Nvfv), kAv(Nvfv->P_ORDER + 1)
        {
            lambdaCell.resize(mesh->cell2nodeLocal.size()); // but only dist part are used, ghost part to not judge for it in facial iter
            lambdaFace.resize(mesh->face2nodeLocal.size());
            lambdaFaceVis.resize(lambdaFace.size());
            deltaLambdaFace.resize(lambdaFace.size());

            dFdUFace.resize(lambdaFace.size());
            jacobianFace.resize(lambdaFace.size());
            jacobianCell.resize(lambdaCell.size());
            jacobianCellInv.resize(lambdaCell.size());

            // vfv->BuildRec(dRdUrec);
            // vfv->BuildRec(dRdb);

            //! wall dist code, to be imporved!!!
            real maxD = 0.1;
            dWall.resize(mesh->cell2nodeLocal.size());

            MPIInfo mpi = mesh->mpi;
            const int NSampleLine = 5;

            index nBCPoint = 0;
            for (index iFace = 0; iFace < mesh->face2nodeLocal.dist->size(); iFace++)
                if (mesh->faceAtrLocal[iFace][0].iPhy == BoundaryType::Wall ||
                    mesh->faceAtrLocal[iFace][0].iPhy == BoundaryType::Wall_Euler ||
                    mesh->faceAtrLocal[iFace][0].iPhy == BoundaryType::Wall_NoSlip)
                {
                    Elem::ElementManager eFace(mesh->faceAtrLocal[iFace][0].type, vfv->faceRecAtrLocal[iFace][0].intScheme);
                    nBCPoint += NSampleLine; //! knowing as line //eFace.getNInt()
                }
            Array<VecStaticBatch<6>> BCPointDist(VecStaticBatch<6>::Context(nBCPoint), mpi);
            index iFill = 0;
            for (index iFace = 0; iFace < mesh->face2nodeLocal.dist->size(); iFace++)
                if (mesh->faceAtrLocal[iFace][0].iPhy == BoundaryType::Wall ||
                    mesh->faceAtrLocal[iFace][0].iPhy == BoundaryType::Wall_Euler ||
                    mesh->faceAtrLocal[iFace][0].iPhy == BoundaryType::Wall_NoSlip)
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
            for (index iCell = 0; iCell < mesh->cell2nodeLocal.size(); iCell++)
            {
                auto &cellAttribute = mesh->cellAtrLocal[iCell][0];
                auto &cellRecAttribute = vfv->cellRecAtrLocal[iCell][0];
                auto c2n = mesh->cell2nodeLocal[iCell];
                Eigen::MatrixXd coords;
                mesh->LoadCoords(c2n, coords);

                Elem::ElementManager eCell(cellAttribute.type, cellRecAttribute.intScheme);
                dWall[iCell].resize(eCell.getNInt());
                for (int ig = 0; ig < eCell.getNInt(); ig++)
                {
                    Elem::tPoint p;
                    eCell.GetIntPoint(ig, p);
                    Eigen::MatrixXd DiNj(1, eCell.getNNode());
                    eCell.GetDiNj(p, DiNj);
                    Elem::tPoint pC = coords * DiNj(0, Eigen::all).transpose();

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
                    dWall[iCell][ig] = distMin;

                    minResult = std::min(minResult, dWall[iCell][ig]);
                }
            }
            std::cout << minResult << " MinWallDist \n";
        }

        real muEff(const Eigen::VectorXd &U)
        {
        }

        Eigen::VectorXd fluxFace(
            const Eigen::VectorXd &ULxy,
            const Eigen::VectorXd &URxy,
            const Eigen::MatrixXd &DiffUxy,
            const Elem::tPoint &unitNorm,
            const Elem::tJacobi &normBase,
            BoundaryType btype,
            Setting::RiemannSolverType rsType,
            index iFace, int ig)
        {
            Eigen::VectorXd UR = URxy;
            Eigen::VectorXd UL = ULxy;
            UR({1, 2, 3}) = normBase.transpose() * UR({1, 2, 3});
            UL({1, 2, 3}) = normBase.transpose() * UL({1, 2, 3});
            Eigen::VectorXd UMeanXy = 0.5 * (ULxy + URxy);
            Eigen::Matrix<real, 3, -1> VisFlux;
            VisFlux.resizeLike(DiffUxy);

            real k = settings.idealGasProperty.CpGas * settings.idealGasProperty.muGas / settings.idealGasProperty.prGas;
            Gas::ViscousFlux_IdealGas(UMeanXy, DiffUxy, unitNorm, btype == BoundaryType::Wall_NoSlip,
                                      settings.idealGasProperty.gamma,
                                      settings.idealGasProperty.muGas,
                                      k,
                                      settings.idealGasProperty.CpGas,
                                      VisFlux);

            Eigen::VectorXd finc;
            finc.resizeLike(ULxy);

            if (rsType == Setting::RiemannSolverType::HLLEP)
                Gas::HLLEPFlux_IdealGas(
                    UL, UR, settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace],
                    [&]()
                    {
                        std::cout << "face at" << vfv->faceCenters[iFace].transpose() << '\n';
                        std::cout << "UL" << UL.transpose() << '\n';
                        std::cout << "UR" << UR.transpose() << std::endl;
                    });
            else if (rsType == Setting::RiemannSolverType::HLLC)
                Gas::HLLCFlux_IdealGas_HartenYee(
                    UL, UR, settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace],
                    [&]()
                    {
                        std::cout << "face at" << vfv->faceCenters[iFace].transpose() << '\n';
                        std::cout << "UL" << UL.transpose() << '\n';
                        std::cout << "UR" << UR.transpose() << std::endl;
                    });
            else if (rsType == Setting::RiemannSolverType::Roe)
                Gas::RoeFlux_IdealGas_HartenYee(
                    UL, UR, settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace],
                    [&]()
                    {
                        std::cout << "face at" << vfv->faceCenters[iFace].transpose() << '\n';
                        std::cout << "UL" << UL.transpose() << '\n';
                        std::cout << "UR" << UR.transpose() << std::endl;
                    });
            else
                assert(false);

            finc({1, 2, 3}) = normBase * finc({1, 2, 3});
            finc -= VisFlux.transpose() * unitNorm;

            return -finc;
        }

        Eigen::VectorXd source(
            const Eigen::VectorXd &U,
            const Eigen::MatrixXd &DiffU,
            index iCell, index ig)
        {
        }

        Eigen::MatrixXd fluxJacobian0_Right(
            const Eigen::VectorXd &UR,
            const Elem::tPoint &uNorm,
            BoundaryType btype)
        {
            //! for euler!!
            const Eigen::VectorXd &U = UR;
            const Elem::tPoint &n = uNorm;

            real rhoun = n.dot(U({1, 2, 3}));
            real rhousqr = U({1, 2, 3}).squaredNorm();
            real gamma = settings.idealGasProperty.gamma;
            Eigen::MatrixXd subFdU(5, 5);
            subFdU.setZero();
            subFdU(0, 1) = n(1 - 1);
            subFdU(0, 2) = n(2 - 1);
            subFdU(0, 3) = n(3 - 1);
            subFdU(1, 0) = -1.0 / (U(1 - 1) * U(1 - 1)) * U(2 - 1) * rhoun + (1.0 / (U(1 - 1) * U(1 - 1)) * n(1 - 1) * (gamma - 1.0) * (rhousqr - U(1 - 1) * U(5 - 1) * 2.0)) / 2.0 + (U(5 - 1) * n(1 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(1, 1) = (rhoun + U(2 - 1) * n(1 - 1) * 2.0 - U(2 - 1) * gamma * n(1 - 1)) / U(1 - 1);
            subFdU(1, 2) = (U(2 - 1) * n(2 - 1)) / U(1 - 1) - (U(3 - 1) * n(1 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(1, 3) = (U(2 - 1) * n(3 - 1)) / U(1 - 1) - (U(4 - 1) * n(1 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(1, 4) = n(1 - 1) * (gamma - 1.0);
            subFdU(2, 0) = -1.0 / (U(1 - 1) * U(1 - 1)) * U(3 - 1) * rhoun + (1.0 / (U(1 - 1) * U(1 - 1)) * n(2 - 1) * (gamma - 1.0) * (rhousqr - U(1 - 1) * U(5 - 1) * 2.0)) / 2.0 + (U(5 - 1) * n(2 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(2, 1) = (U(3 - 1) * n(1 - 1)) / U(1 - 1) - (U(2 - 1) * n(2 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(2, 2) = (rhoun + U(3 - 1) * n(2 - 1) * 2.0 - U(3 - 1) * gamma * n(2 - 1)) / U(1 - 1);
            subFdU(2, 3) = (U(3 - 1) * n(3 - 1)) / U(1 - 1) - (U(4 - 1) * n(2 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(2, 4) = n(2 - 1) * (gamma - 1.0);
            subFdU(3, 0) = -1.0 / (U(1 - 1) * U(1 - 1)) * U(4 - 1) * rhoun + (1.0 / (U(1 - 1) * U(1 - 1)) * n(3 - 1) * (gamma - 1.0) * (rhousqr - U(1 - 1) * U(5 - 1) * 2.0)) / 2.0 + (U(5 - 1) * n(3 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(3, 1) = (U(4 - 1) * n(1 - 1)) / U(1 - 1) - (U(2 - 1) * n(3 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(3, 2) = (U(4 - 1) * n(2 - 1)) / U(1 - 1) - (U(3 - 1) * n(3 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(3, 3) = (rhoun + U(4 - 1) * n(3 - 1) * 2.0 - U(4 - 1) * gamma * n(3 - 1)) / U(1 - 1);
            subFdU(3, 4) = n(3 - 1) * (gamma - 1.0);
            subFdU(4, 0) = 1.0 / (U(1 - 1) * U(1 - 1) * U(1 - 1)) * rhoun * (-rhousqr + (U(2 - 1) * U(2 - 1)) * gamma + (U(3 - 1) * U(3 - 1)) * gamma + (U(4 - 1) * U(4 - 1)) * gamma - U(1 - 1) * U(5 - 1) * gamma);
            subFdU(4, 1) = 1.0 / (U(1 - 1) * U(1 - 1)) * n(1 - 1) * (-rhousqr + (U(2 - 1) * U(2 - 1)) * gamma + (U(3 - 1) * U(3 - 1)) * gamma + (U(4 - 1) * U(4 - 1)) * gamma - U(1 - 1) * U(5 - 1) * gamma * 2.0) * (-1.0 / 2.0) - 1.0 / (U(1 - 1) * U(1 - 1)) * U(2 - 1) * rhoun * (gamma - 1.0);
            subFdU(4, 2) = 1.0 / (U(1 - 1) * U(1 - 1)) * n(2 - 1) * (-rhousqr + (U(2 - 1) * U(2 - 1)) * gamma + (U(3 - 1) * U(3 - 1)) * gamma + (U(4 - 1) * U(4 - 1)) * gamma - U(1 - 1) * U(5 - 1) * gamma * 2.0) * (-1.0 / 2.0) - 1.0 / (U(1 - 1) * U(1 - 1)) * U(3 - 1) * rhoun * (gamma - 1.0);
            subFdU(4, 3) = 1.0 / (U(1 - 1) * U(1 - 1)) * n(3 - 1) * (-rhousqr + (U(2 - 1) * U(2 - 1)) * gamma + (U(3 - 1) * U(3 - 1)) * gamma + (U(4 - 1) * U(4 - 1)) * gamma - U(1 - 1) * U(5 - 1) * gamma * 2.0) * (-1.0 / 2.0) - 1.0 / (U(1 - 1) * U(1 - 1)) * U(4 - 1) * rhoun * (gamma - 1.0);
            subFdU(4, 4) = (gamma * rhoun) / U(1 - 1);
            return subFdU;
        }

        Eigen::VectorXd generateBoundaryValue(
            const Eigen::VectorXd &ULxy,
            const Elem::tPoint &uNorm,
            const Elem::tJacobi &normBase,
            const Elem::tPoint &pPhysics,
            real t,
            BoundaryType btype)
        {
            Eigen::VectorXd URxy;

            if (btype == BoundaryType::Farfield ||
                btype == BoundaryType::Special_DMRFar)
            {
                if (btype == BoundaryType::Farfield)
                    URxy = settings.farFieldStaticValue;
                else if (btype == BoundaryType::Special_DMRFar)
                {
                    URxy = settings.farFieldStaticValue;
                    real uShock = 10;
                    if (((pPhysics(0) - uShock / std::sin(pi / 3) * t - 1. / 6.) -
                         pPhysics(1) / std::tan(pi / 3)) > 0)
                        URxy({0, 1, 2, 3, 4}) = Eigen::Vector<real, 5>{1.4, 0, 0, 0, 2.5};
                    else
                        URxy({0, 1, 2, 3, 4}) = Eigen::Vector<real, 5>{8, 57.157676649772960, -33, 0, 5.635e2};
                }
                else
                    assert(false);
            }
            else if (btype == BoundaryType::Wall_Euler)
            {
                URxy = ULxy;
                URxy({1, 2, 3}) -= URxy({1, 2, 3}).dot(uNorm) * uNorm;
            }
            else if (btype == BoundaryType::Wall_NoSlip)
            {
                URxy = ULxy;
                URxy({1, 2, 3}) *= -1;
            }
            else if (btype == BoundaryType::Wall)
            {
                std::cout << "Wall is not a proper bc" << std::endl;
                assert(false);
            }
            else
            {
                assert(false);
            }
            return URxy;
        }

        static Eigen::Vector<real, -1> CompressRecPart(
            const Eigen::Vector<real, -1> &umean,
            const Eigen::Vector<real, -1> &uRecInc)
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


        //! for euler now!
        Eigen::Vector<real, -1> ret = umean + uRecInc;
        real eK = ret({1, 2, 3}).squaredNorm() * 0.5 / (verySmallReal + std::abs(ret(0)));
        real e = ret(4) - eK;
        if (e <= 0 || ret(0) <= 0)
            ret = umean;

        return ret;
    }

        void EvaluateDt(std::vector<real> &dt,
                        ArrayDOFV &u,
                        real CFL, real &dtMinall, real MaxDt = 1,
                        bool UseLocaldt = false);
        /**
         * @brief
         * \param rhs overwritten;
         *
         */
        void EvaluateRHS(ArrayDOFV &rhs, ArrayDOFV &u,
                         ArrayRecV &uRec, real t);

        void LUSGSADMatrixInit(std::vector<real> &dTau, real dt, real alphaDiag, ArrayDOFV &u, int jacobianCode = 1,
                               real t = 0);

        void LUSGSADMatrixVec(ArrayDOFV &u, ArrayDOFV &uInc, ArrayDOFV &AuInc);

        void UpdateLUSGSADForward(ArrayDOFV &rhs, ArrayDOFV &u, ArrayDOFV &uInc, ArrayDOFV &uIncNew);

        void UpdateLUSGSADBackward(ArrayDOFV &rhs, ArrayDOFV &u, ArrayDOFV &uInc, ArrayDOFV &uIncNew);

        void LUSGSMatrixVec(std::vector<real> &dTau, real dt, real alphaDiag,
                            ArrayDOFV &u, ArrayDOFV &uInc, ArrayDOFV &AuInc);

        /**
         * @brief to use LUSGS, use LUSGSForward(..., uInc, uInc); uInc.pull; LUSGSBackward(..., uInc, uInc);
         * the underlying logic is that for index, ghost > dist, so the forward uses no ghost,
         * and ghost should be pulled before using backward;
         * to use Jacobian instead of LUSGS, use LUSGSForward(..., uInc, uIncNew); LUSGSBackward(..., uInc, uIncNew); uIncNew.pull; uInc = uIncNew;
         * \param uIncNew overwritten;
         *
         */
        void UpdateLUSGSForward(std::vector<real> &dTau, real dt, real alphaDiag,
                                ArrayDOFV &rhs, ArrayDOFV &u, ArrayDOFV &uInc, ArrayDOFV &uIncNew);

        /**
         * @brief
         * \param uIncNew overwritten;
         *
         */
        void UpdateLUSGSBackward(std::vector<real> &dTau, real dt, real alphaDiag,
                                 ArrayDOFV &rhs, ArrayDOFV &u, ArrayDOFV &uInc, ArrayDOFV &uIncNew);

        void UpdateSGS(std::vector<real> &dTau, real dt, real alphaDiag,
                       ArrayDOFV &rhs, ArrayDOFV &u, ArrayDOFV &uInc, ArrayDOFV &uIncNew, bool ifForward);

        void FixUMaxFilter(ArrayDOFV &u);

        void EvaluateResidual(Eigen::Vector<real, -1> &res, ArrayDOFV &rhs, index P = 1);
    };
}