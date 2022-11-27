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
    enum EulerModel
    {
        NS = 0,
        NS_SA = 1
    };

    static inline int getNVars(EulerModel model)
    {
        int nVars = -1;
        switch (model)
        {
        case NS:
            nVars = 5;
            break;
        case NS_SA:
            nVars = 6;
            break;
        }
        return nVars;
    }

    class EulerEvaluator
    {
        int nVars = 5;
        EulerModel model = NS;

        bool passiveDiscardSource = false;

    public:
        void setPassiveDiscardSource(bool n) { passiveDiscardSource = n; }

    private:
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
                real Rgas = 1;
                real muGas = 1;
                real prGas = 0.7;
                real CpGas = Rgas * gamma / (gamma - 1);
                real TRef = 273.15;
                real CSutherland = 110.4;
            } idealGasProperty;

            int nTimeFilterPass = 0;

            real visScale = 1;
            real visScaleIn = 1;
            real isiScale = 1;
            real isiScaleIn = 1;
            real isiCutDown = 0.5;
            real ekCutDown = 0.5;

            Eigen::Vector<real, -1> farFieldStaticValue = Eigen::Vector<real, 5>{1, 0, 0, 0, 2.5};

            struct BoxInitializer
            {
                real x0, x1, y0, y1, z0, z1;
                Eigen::Vector<real, -1> v;
            };
            std::vector<BoxInitializer> boxInitializers;

            struct PlaneInitializer
            {
                real a, b, c, h;
                Eigen::Vector<real, -1> v;
            };
            std::vector<PlaneInitializer> planeInitializers;

        } settings;

        EulerEvaluator(CompactFacedMeshSerialRW *Nmesh, ImplicitFiniteVolume2D *Nfv, VRFiniteVolume2D *Nvfv,
                       EulerModel nmodel)
            : model(nmodel), mesh(Nmesh), fv(Nfv), vfv(Nvfv), kAv(Nvfv->P_ORDER + 1)
        {
            nVars = getNVars(model);

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
            const int NSampleLine = 11;

            index nBCPoint = 0;
            for (index iFace = 0; iFace < mesh->face2nodeLocal.dist->size(); iFace++)
                if (
                    mesh->faceAtrLocal[iFace][0].iPhy == BoundaryType::Wall_NoSlip)
                {
                    Elem::ElementManager eFace(mesh->faceAtrLocal[iFace][0].type, vfv->faceRecAtrLocal[iFace][0].intScheme);
                    nBCPoint += NSampleLine; //! knowing as line //eFace.getNInt()
                }
            Array<VecStaticBatch<6>> BCPointDist(VecStaticBatch<6>::Context(nBCPoint), mpi);
            index iFill = 0;
            for (index iFace = 0; iFace < mesh->face2nodeLocal.dist->size(); iFace++)
                if (
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
                dWall[iCell].resize(eCell.getNInt(), std::pow(veryLargeReal, 1. / 6.));
                for (int ig = 0; ig < eCell.getNInt(); ig++)
                {
                    Elem::tPoint p;
                    eCell.GetIntPoint(ig, p);
                    Eigen::MatrixXd DiNj(1, eCell.getNNode());
                    eCell.GetDiNj(p, DiNj);
                    Elem::tPoint pC = coords * DiNj(0, Eigen::all).transpose();

                    index imin = -1;
                    real distMin = std::pow(veryLargeReal, 1. / 6.);
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
                    // if (pC(0) > 0)
                    //     dWall[iCell][ig] = std::min(pC(1), distMin); // ! debugging!

                    minResult = std::min(minResult, dWall[iCell][ig]);
                }
            }
            std::cout << minResult << " MinWallDist \n";
        }

        real muEff(const Eigen::VectorXd &U)
        {
            return 0. / 0.;
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

            real pMean, asqrMean, Hmean;
            real gamma = settings.idealGasProperty.gamma;
            Gas::IdealGasThermal(UMeanXy(4), UMeanXy(0), (UMeanXy({1, 2, 3}) / UMeanXy(0)).squaredNorm(),
                                 gamma, pMean, asqrMean, Hmean);

            // ! refvalue:
            real muRef = settings.idealGasProperty.muGas;

            real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * UMeanXy(0));
            real mufPhy, muf;
            mufPhy = muf = settings.idealGasProperty.muGas *
                           std::pow(T / settings.idealGasProperty.TRef, 1.5) *
                           (settings.idealGasProperty.TRef + settings.idealGasProperty.CSutherland) /
                           (T + settings.idealGasProperty.CSutherland);

            real fnu1 = 0.;
            if (model == NS_SA)
            {
                real cnu1 = 7.1;
                real Chi = UMeanXy(5) * muRef / mufPhy;
                if (Chi < 10)
                    Chi = 0.05 * std::log(1 + std::exp(20 * Chi));
                real Chi3 = std::pow(Chi, 3);
                fnu1 = Chi3 / (Chi3 + std::pow(cnu1, 3));
                muf *= (1 + Chi * fnu1);
            }

            real k = settings.idealGasProperty.CpGas * muf / settings.idealGasProperty.prGas;
            VisFlux.setZero();
            Gas::ViscousFlux_IdealGas(UMeanXy, DiffUxy, unitNorm, btype == BoundaryType::Wall_NoSlip,
                                      settings.idealGasProperty.gamma,
                                      muf,
                                      k,
                                      settings.idealGasProperty.CpGas,
                                      VisFlux);
            if (model == NS_SA)
            {
                real sigma = 2. / 3.;
                real lambdaFaceC = sqrt(std::abs(asqrMean)) + std::abs(UL(1) + UR(1)) * 0.5;
                Eigen::Matrix<real, 3, 1> diffRhoNu = DiffUxy({0, 1, 2}, {5}) * muRef;
                Eigen::Matrix<real, 3, 1> diffRho = DiffUxy({0, 1, 2}, {0});
                Eigen::Matrix<real, 3, 1> diffNu = (diffRhoNu - UMeanXy(5) * muRef / UMeanXy(0) * diffRho) / UMeanXy(0);

                real cn1 = 16;
                real fn = 1;
                if (UMeanXy(5) < 0)
                {
                    real Chi = UMeanXy(5) * muRef / mufPhy;
                    fn = (cn1 + std::pow(Chi, 3)) / (cn1 - std::pow(Chi, 3));
                }
                VisFlux({0, 1, 2}, {5}) = diffNu * (mufPhy + UMeanXy(5) * muRef * fn) / sigma / muRef;
            }

            Eigen::VectorXd finc;
            finc.resizeLike(ULxy);

            // std::cout << "HERE" << std::endl;
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
            // std::cout << "HERE2" << std::endl;

            if (model == NS_SA)
            {
                // real lambdaFaceC = sqrt(std::abs(asqrMean)) + std::abs(UL(1) / UL(0) + UR(1) / UR(0)) * 0.5;
                real lambdaFaceC = std::abs(UL(1) / UL(0) + UR(1) / UR(0)) * 0.5; //! using velo instead of velo + a
                finc(5) = ((UL(1) / UL(0) * UL(5) + UR(1) / UR(0) * UR(5)) -
                           (UR(5) - UL(5)) * lambdaFaceC) *
                          0.5;
            }

            finc({1, 2, 3}) = normBase * finc({1, 2, 3});
            finc -= VisFlux.transpose() * unitNorm;

            if (finc.hasNaN() || (!finc.allFinite()))
            {
                std::cout << finc.transpose() << std::endl;
                std::cout << ULxy.transpose() << std::endl;
                std::cout << URxy.transpose() << std::endl;
                std::cout << DiffUxy << std::endl;
                std::cout << unitNorm << std::endl;
                std::cout << normBase << std::endl;
                std::cout << T << std::endl;
                std::cout << muf << std::endl;
                std::cout << pMean << std::endl;
                assert(false);
            }

            return -finc;
        }

        Eigen::VectorXd source(
            const Eigen::VectorXd &UMeanXy,
            const Eigen::MatrixXd &DiffUxy,
            index iCell, index ig)
        {
            if (model == NS)
            {
            }
            else if (model == NS_SA)
            {
                real d = std::min(dWall[iCell][ig], std::pow(veryLargeReal, 1. / 6.));
                real cb1 = 0.1355;
                real cb2 = 0.622;
                real sigma = 2. / 3.;
                real cnu1 = 7.1;
                real cnu2 = 0.7;
                real cnu3 = 0.9;
                real cw2 = 0.3;
                real cw3 = 2;
                real kappa = 0.41;
                real rlim = 10;
                real cw1 = cb1 / sqr(kappa) + (1 + cb2) / sigma;

                real ct3 = 1.2;
                real ct4 = 0.5;

                real pMean, asqrMean, Hmean;
                real gamma = settings.idealGasProperty.gamma;
                Gas::IdealGasThermal(UMeanXy(4), UMeanXy(0), (UMeanXy({1, 2, 3}) / UMeanXy(0)).squaredNorm(),
                                     gamma, pMean, asqrMean, Hmean);
                // ! refvalue:
                real muRef = settings.idealGasProperty.muGas;

                real nuh = UMeanXy(5) * muRef / UMeanXy(0);

                real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * UMeanXy(0));
                real mufPhy, muf;
                mufPhy = muf = settings.idealGasProperty.muGas *
                               std::pow(T / settings.idealGasProperty.TRef, 1.5) *
                               (settings.idealGasProperty.TRef + settings.idealGasProperty.CSutherland) /
                               (T + settings.idealGasProperty.CSutherland);

                real Chi = std::abs(UMeanXy(5) * muRef / mufPhy);
                real fnu1 = std::pow(Chi, 3) / (std::pow(Chi, 3) + std::pow(cnu1, 3));
                real fnu2 = 1 - Chi / (1 + Chi * fnu1);

                Eigen::Matrix<real, 3, 1> velo = UMeanXy({1, 2, 3}) / UMeanXy(0);
                Eigen::Matrix<real, 3, 1> diffRhoNu = DiffUxy({0, 1, 2}, {5}) * muRef;
                Eigen::Matrix<real, 3, 1> diffRho = DiffUxy({0, 1, 2}, {0});
                Eigen::Matrix<real, 3, 1> diffNu = (diffRhoNu - nuh * diffRho) / UMeanXy(0);
                Eigen::Matrix<real, 3, 3> diffRhoU = DiffUxy({0, 1, 2}, {1, 2, 3});
                Eigen::Matrix<real, 3, 3> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);

                Eigen::Matrix<real, 3, 3> Omega = 0.5 * (diffU.transpose() - diffU);
                real S = Omega.norm() * std::sqrt(2);
                real Sbar = nuh / (sqr(kappa) * sqr(d)) * fnu2;

                real Sh;
                if (Sbar < -cnu2 * S)
                    Sh = S + S * (sqr(cnu2) * S + cnu3 * Sbar) / ((cnu3 - 2 * cnu2) * S - Sbar);
                else
                    Sh = S + Sbar;

                real r = std::min(nuh / (Sh * sqr(kappa * d) + verySmallReal), rlim);
                real g = r + cw2 * (std::pow(r, 6) - r);
                real fw = g * std::pow((1 + std::pow(cw3, 6)) / (std::pow(g, 6) + std::pow(cw3, 6)), 1. / 6.);

                real ft2 = ct3 * std::exp(-ct4 * sqr(Chi));
                real D = (cw1 * fw - cb1 / sqr(kappa) * ft2) * sqr(nuh / d); //! modified >>
                real P = cb1 * (1 - ft2) * Sh * nuh;                         //! modified >>
                // real D = cw1 * fw * sqr(nuh / d);
                // real P = cb1 * Sh * nuh;
                real fn = 1;
                if (UMeanXy(5) < 0)
                {
                    real cn1 = 16;
                    real Chi = UMeanXy(5) * muRef / mufPhy;
                    fn = (cn1 + std::pow(Chi, 3)) / (cn1 - std::pow(Chi, 3));
                    P = cb1 * (1 - ct3) * S * nuh;
                    D = -cw1 * sqr(nuh / d);
                }

                Eigen::VectorXd ret;
                ret.resizeLike(UMeanXy);
                ret.setZero();

                if (passiveDiscardSource)
                    P = D = 0;
                ret(5) = UMeanXy(0) * (P - D + diffNu.squaredNorm() * cb2 / sigma) / muRef -
                         (UMeanXy(5) * fn * muRef + mufPhy) / (UMeanXy(0) * sigma) * diffRho.dot(diffNu) / muRef;

                if (ret.hasNaN())
                {
                    std::cout << P << std::endl;
                    std::cout << D << std::endl;
                    std::cout << UMeanXy(0) << std::endl;
                    std::cout << Sh << std::endl;
                    std::cout << nuh << std::endl;
                    std::cout << g << std::endl;
                    std::cout << r << std::endl;
                    std::cout << S << std::endl;
                    std::cout << d << std::endl;
                    std::cout << fnu2 << std::endl;
                    std::cout << mufPhy << std::endl;
                    assert(false);
                }
                // if (passiveDiscardSource)
                //     ret(Eigen::seq(5, Eigen::last)).setZero();
                return ret;
            }
            else
            {
                assert(false);
            }

            return Eigen::VectorXd::Zero(UMeanXy.size());
        }

        // zeroth means needs not derivative
        Eigen::VectorXd sourceJacobianDiag_Zeroth(
            const Eigen::VectorXd &UMeanXy,
            const Eigen::MatrixXd &DiffUxy,
            index iCell, index ig)
        {
            if (model == NS)
            {
            }
            else if (model == NS_SA)
            {
                real d = std::min(dWall[iCell][ig], std::pow(veryLargeReal, 1. / 6.));
                real cb1 = 0.1355;
                real cb2 = 0.622;
                real sigma = 2. / 3.;
                real cnu1 = 7.1;
                real cnu2 = 0.7;
                real cnu3 = 0.9;
                real cw2 = 0.3;
                real cw3 = 2;
                real kappa = 0.41;
                real rlim = 10;
                real cw1 = cb1 / sqr(kappa) + (1 + cb2) / sigma;

                real ct3 = 1.2;
                real ct4 = 0.5;

                real pMean, asqrMean, Hmean;
                real gamma = settings.idealGasProperty.gamma;
                Gas::IdealGasThermal(UMeanXy(4), UMeanXy(0), (UMeanXy({1, 2, 3}) / UMeanXy(0)).squaredNorm(),
                                     gamma, pMean, asqrMean, Hmean);
                // ! refvalue:
                real muRef = settings.idealGasProperty.muGas;

                real nuh = UMeanXy(5) * muRef / UMeanXy(0);

                real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * UMeanXy(0));
                real mufPhy, muf;
                mufPhy = muf = settings.idealGasProperty.muGas *
                               std::pow(T / settings.idealGasProperty.TRef, 1.5) *
                               (settings.idealGasProperty.TRef + settings.idealGasProperty.CSutherland) /
                               (T + settings.idealGasProperty.CSutherland);

                real Chi = std::abs(UMeanXy(5) * muRef / mufPhy);
                real fnu1 = std::pow(Chi, 3) / (std::pow(Chi, 3) + std::pow(cnu1, 3));
                real fnu2 = 1 - Chi / (1 + Chi * fnu1);

                Eigen::Matrix<real, 3, 1> velo = UMeanXy({1, 2, 3}) / UMeanXy(0);
                Eigen::Matrix<real, 3, 1> diffRhoNu = DiffUxy({0, 1, 2}, {5}) * muRef;
                Eigen::Matrix<real, 3, 1> diffRho = DiffUxy({0, 1, 2}, {0});
                Eigen::Matrix<real, 3, 1> diffNu = (diffRhoNu - nuh * diffRho) / UMeanXy(0);
                Eigen::Matrix<real, 3, 3> diffRhoU = DiffUxy({0, 1, 2}, {1, 2, 3});
                Eigen::Matrix<real, 3, 3> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);

                Eigen::Matrix<real, 3, 3> Omega = 0.5 * (diffU.transpose() - diffU);
                real S = Omega.norm() * std::sqrt(2);
                real Sbar = nuh / (sqr(kappa) * sqr(d)) * fnu2;

                real Sh;
                if (Sbar < -cnu2 * S)
                    Sh = S + S * (sqr(cnu2) * S + cnu3 * Sbar) / ((cnu3 - 2 * cnu2) * S - Sbar);
                else
                    Sh = S + Sbar;

                real r = std::min(nuh / (Sh * sqr(kappa * d) + verySmallReal), rlim);
                real g = r + cw2 * (std::pow(r, 6) - r);
                real fw = g * std::pow((1 + std::pow(cw3, 6)) / (std::pow(g, 6) + std::pow(cw3, 6)), 1. / 6.);

                real ft2 = ct3 * std::exp(-ct4 * sqr(Chi));
                real D = (cw1 * fw - cb1 / sqr(kappa) * ft2) * sqr(nuh / d); //! modified >>
                real P = cb1 * (1 - ft2) * Sh * nuh;                         //! modified >>
                // real D = cw1 * fw * sqr(nuh / d);
                // real P = cb1 * Sh * nuh;
                real fn = 1;
                if (UMeanXy(5) < 0)
                {
                    real cn1 = 16;
                    real Chi = UMeanXy(5) * muRef / mufPhy;
                    fn = (cn1 + std::pow(Chi, 3)) / (cn1 - std::pow(Chi, 3));
                    P = cb1 * (1 - ct3) * S * nuh;
                    D = -cw1 * sqr(nuh / d);
                }

                Eigen::VectorXd ret;
                ret.resizeLike(UMeanXy);
                ret.setZero();

                if (passiveDiscardSource)
                    P = D = 0;
                ret(5) = UMeanXy(0) * (-D) / muRef / UMeanXy(5);

                if (ret.hasNaN())
                {
                    std::cout << P << std::endl;
                    std::cout << D << std::endl;
                    std::cout << UMeanXy(0) << std::endl;
                    std::cout << Sh << std::endl;
                    std::cout << nuh << std::endl;
                    std::cout << g << std::endl;
                    std::cout << r << std::endl;
                    std::cout << S << std::endl;
                    std::cout << d << std::endl;
                    std::cout << fnu2 << std::endl;
                    std::cout << mufPhy << std::endl;
                    assert(false);
                }
                // if (passiveDiscardSource)
                //     ret(Eigen::seq(5, Eigen::last)).setZero();
                return ret;
            }
            else
            {
                assert(false);
            }

            return Eigen::VectorXd::Zero(UMeanXy.size());
        }

        Eigen::MatrixXd fluxJacobian0_Right(
            const Eigen::VectorXd &UR,
            const Elem::tPoint &uNorm,
            BoundaryType btype)
        {
            const Eigen::VectorXd &U = UR;
            const Elem::tPoint &n = uNorm;

            real rhoun = n.dot(U({1, 2, 3}));
            real rhousqr = U({1, 2, 3}).squaredNorm();
            real gamma = settings.idealGasProperty.gamma;
            Eigen::MatrixXd subFdU(nVars, nVars);
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

            real un = rhoun / U(0);

            if (model == NS_SA)
            {
                subFdU(5, 5) = un;
                subFdU(5, 0) = -un * U(5) / U(0);
                subFdU(5, 1) = n(0) * U(5) / U(0);
                subFdU(5, 2) = n(1) * U(5) / U(0);
                subFdU(5, 3) = n(2) * U(5) / U(0);
            }
            return subFdU;
        }

        Eigen::VectorXd fluxJacobian0_Right_Times_du(
            const Eigen::VectorXd &U,
            const Elem::tPoint &n,
            BoundaryType btype,
            const Eigen::VectorXd &dU)
        {
            real gamma = settings.idealGasProperty.gamma;
            Elem::tPoint velo = U({1, 2, 3}) / U(0);
            real p, H, asqr;
            Gas::IdealGasThermal(U(4), U(0), velo.squaredNorm(), gamma, p, asqr, H);
            Elem::tPoint dVelo;
            real dp;
            Gas::IdealGasUIncrement(U, dU, velo, gamma, dVelo, dp);
            Eigen::VectorXd dF(U.size());
            Gas::GasInviscidFluxFacialIncrement(U, dU,
                                                n,
                                                velo, dVelo,
                                                dp, p,
                                                dF);
            if (model == NS_SA)
            {
                dF(5) = dU(5) * n.dot(velo) + U(5) * n.dot(dVelo);
            }
            return dF;
        }

        Eigen::VectorXd generateBoundaryValue(
            const Eigen::VectorXd &ULxy,
            const Elem::tPoint &uNorm,
            const Elem::tJacobi &normBase,
            const Elem::tPoint &pPhysics,
            real t,
            BoundaryType btype)
        {
            assert(ULxy(0) > 0);
            Eigen::VectorXd URxy;

            if (btype == BoundaryType::Farfield ||
                btype == BoundaryType::Special_DMRFar)
            {
                if (btype == BoundaryType::Farfield)
                {
                    const Eigen::VectorXd &far = settings.farFieldStaticValue;

                    real un = ULxy({1, 2, 3}).dot(uNorm) / ULxy(0);
                    real vsqr = (ULxy({1, 2, 3}) / ULxy(0)).squaredNorm();
                    real gamma = settings.idealGasProperty.gamma;
                    real asqr, H, p;
                    Gas::IdealGasThermal(ULxy(4), ULxy(0), vsqr, gamma, p, asqr, H);

                    assert(asqr >= 0);
                    real a = std::sqrt(asqr);

                    if (un - a > 0) // full outflow
                    {
                        URxy = ULxy;
                    }
                    else if (un > 0) //  1 sonic outflow, 1 sonic inflow, other outflow (subsonic out)
                    {
                        Eigen::VectorXd farPrimitive, ULxyPrimitive;
                        Gas::IdealGasThermalConservative2Primitive(far, farPrimitive, gamma);
                        Gas::IdealGasThermalConservative2Primitive(ULxy, ULxyPrimitive, gamma);
                        ULxyPrimitive(4) = farPrimitive(4); // using far pressure
                        Gas::IdealGasThermalPrimitive2Conservative(ULxyPrimitive, URxy, gamma);
                    }
                    else if (un + a > 0) //  1 sonic outflow, 1 sonic inflow, other inflow (subsonic in)
                    {
                        Eigen::VectorXd farPrimitive, ULxyPrimitive;
                        Gas::IdealGasThermalConservative2Primitive(far, farPrimitive, gamma);
                        Gas::IdealGasThermalConservative2Primitive(ULxy, ULxyPrimitive, gamma);
                        // farPrimitive(0) = ULxyPrimitive(0); // using inner density
                        farPrimitive(4) = ULxyPrimitive(4); // using inner pressure
                        Gas::IdealGasThermalPrimitive2Conservative(farPrimitive, URxy, gamma);
                    }
                    else // full inflow
                    {
                        URxy = settings.farFieldStaticValue;
                    }
                    // URxy = settings.farFieldStaticValue; //!override
                }
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
                if (model == NS_SA)
                {
                    URxy(5) *= -1;
                }
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

        inline Eigen::Vector<real, -1> CompressRecPart(
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
            if (model == NS_SA && ret(5) < 0)
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

        void EulerEvaluator::LUSGSMatrixInit(std::vector<real> &dTau, real dt, real alphaDiag,
                                             ArrayDOFV &u, ArrayRecV &uRec,
                                             int jacobianCode,
                                             real t);

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