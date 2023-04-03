#pragma once
#include "DNDS_Gas.hpp"
#include "DNDS_Mesh.hpp"
#include "DNDS_FV_VR.hpp"
#include "DNDS_Scripting.hpp"
#include "DNDS_Linear.hpp"
#include <iomanip>
#include <functional>

// #define DNDS_FV_EULEREVALUATOR_SOURCE_TERM_ZERO
// #define DNDS_FV_EULEREVALUATOR_IGNORE_SOURCE_TERM
// #define DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM

#ifdef DNDS_FV_EULEREVALUATOR_IGNORE_SOURCE_TERM // term dependency
// #define DNDS_FV_EULEREVALUATOR_USE_SCALAR_JACOBIAN
#endif

namespace DNDS
{
    enum EulerModel
    {
        NS = 0,
        NS_SA = 1,
        NS_2D = 2
    };

    constexpr static inline int getNVars_Fixed(const EulerModel model)
    {
        if (model == NS)
            return 5;
        else if (model == NS_SA)
            return 6;
        else if (model == NS_2D)
            return 4;
        return Eigen::Dynamic;
    }

    constexpr static inline int getDim_Fixed(const EulerModel model)
    {
        if (model == NS)
            return 3;
        else if (model == NS_SA)
            return 3;
        else if (model == NS_2D)
            return 2;
        return Eigen::Dynamic;
    }

    // constexpr static inline bool ifFixedNvars(EulerModel model)
    // {
    //     return (
    //         model == NS ||
    //         model == NS_SA);
    // } // use +/- is ok

    constexpr static inline int getNVars(EulerModel model)
    {
        int nVars = getNVars_Fixed(model);
        if (nVars < 0)
        { // *** handle variable nVars
        }
        return nVars;
    }

    template <int nvars_Fixed, int mul>
    constexpr static inline int nvarsFixedMultipy()
    {
        return nvars_Fixed != Eigen::Dynamic ? nvars_Fixed * mul : Eigen::Dynamic;
    }

    template <EulerModel model>
    class EulerEvaluator
    {
    public:
        static const int nVars_Fixed = getNVars_Fixed(model);
        static const int dim = getDim_Fixed(model);
        static const auto I4 = dim + 1;

#define DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS                            \
    static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>); \
    static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);     \
    static const auto Seq01234 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>);

        typedef Eigen::Vector<real, dim> TVec;
        typedef Eigen::Matrix<real, dim, dim> TMat;
        typedef Eigen::Vector<real, nVars_Fixed> TU;
        typedef Eigen::Matrix<real, nVars_Fixed, nVars_Fixed> TJacobianU;
        typedef Eigen::Matrix<real, dim, nVars_Fixed> TDiffU;
        typedef Eigen::Matrix<real, nVars_Fixed, dim> TDIffUTransposed;

    public:
        static const int gdim = 2; //* geometry dim

    private:
        int nVars = 5;

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

        // todo: improve to contiguous

        std::vector<Eigen::Vector<real, nVars_Fixed>> jacobianCellSourceDiag;
        std::vector<Eigen::Matrix<real, nvarsFixedMultipy<nVars_Fixed, 2>(), nVars_Fixed>> jacobianFace;
        std::vector<Eigen::Matrix<real, nVars_Fixed, nVars_Fixed>> jacobianCell;
        std::vector<Eigen::Matrix<real, nVars_Fixed, nVars_Fixed>> jacobianCellInv;
        std::vector<real> jacobianCell_Scalar;
        std::vector<real> jacobianCellInv_Scalar;

        // std::vector<Eigen::Vector<real, nVars_Fixed>> jacobianCellSourceDiag_Fixed;
        // std::vector<Eigen::Matrix<real, nVars_Fixed, nVars_Fixed>> jacobianFace_Fixed;
        // std::vector<Eigen::Matrix<real, nVars_Fixed, nVars_Fixed>> jacobianCell_Fixed;
        // std::vector<Eigen::Matrix<real, nVars_Fixed, nVars_Fixed>> jacobianCellInv_Fixed;

        std::vector<std::vector<real>> dWall;

        // ArrayVDOF<25> dRdUrec;
        // ArrayVDOF<25> dRdb;

        Eigen::Vector<real, -1> fluxWallSum;

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

            int specialBuiltinInitializer = 0;

            Eigen::Vector<real, 3> constMassForce = Eigen::Vector<real, 3>{0, 0, 0};

        } settings;

        EulerEvaluator(CompactFacedMeshSerialRW *Nmesh, ImplicitFiniteVolume2D *Nfv, VRFiniteVolume2D *Nvfv)
            : mesh(Nmesh), fv(Nfv), vfv(Nvfv), kAv(Nvfv->P_ORDER + 1)
        {
            nVars = getNVars(model);

            lambdaCell.resize(mesh->cell2nodeLocal.size()); // but only dist part are used, ghost part to not judge for it in facial iter
            lambdaFace.resize(mesh->face2nodeLocal.size());
            lambdaFaceVis.resize(lambdaFace.size());
            deltaLambdaFace.resize(lambdaFace.size());

            dFdUFace.resize(lambdaFace.size());

            jacobianFace.resize(lambdaFace.size(), typename decltype(jacobianFace)::value_type(nVars * 2, nVars));
            jacobianCell.resize(lambdaCell.size(), typename decltype(jacobianCell)::value_type(nVars, nVars));
            jacobianCellInv.resize(lambdaCell.size(), typename decltype(jacobianCellInv)::value_type(nVars, nVars));
            using jacobianCellSourceDiagElemType = typename decltype(jacobianCellSourceDiag)::value_type;
            jacobianCellSourceDiag.resize(lambdaCell.size(), jacobianCellSourceDiagElemType::Zero(nVars)); // zeroed
            jacobianCell_Scalar.resize(lambdaCell.size());
            jacobianCellInv_Scalar.resize(lambdaCell.size());

            // jacobianFace_Fixed.resize(lambdaFace.size());
            // jacobianCell_Fixed.resize(lambdaCell.size());
            // jacobianCellInv_Fixed.resize(lambdaCell.size());
            // jacobianCellSourceDiag_Fixed.resize(lambdaCell.size()); // zeroed
            // for (auto &i : jacobianCellSourceDiag_Fixed)
            //     i.setZero();

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
                        BCPointDist[iFill].p()({3, 4, 5}) = vfv->faceNormCenter[iFace].normalized(); //! ig is not gauss index!
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

        real muEff(const TU &U)
        {
            return 0. / 0.;
        }

        TU fluxFace(
            const TU &ULxy,
            const TU &URxy,
            const TDiffU &DiffUxy,
            const TVec &unitNorm,
            const TMat &normBase,
            BoundaryType btype,
            typename Setting::RiemannSolverType rsType,
            index iFace, int ig)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS

            TU UR = URxy;
            TU UL = ULxy;
            UR(Seq123) = normBase(Seq012, Seq012).transpose() * UR(Seq123);
            UL(Seq123) = normBase(Seq012, Seq012).transpose() * UL(Seq123);
            TU UMeanXy = 0.5 * (ULxy + URxy);

            real pMean, asqrMean, Hmean;
            real gamma = settings.idealGasProperty.gamma;
            Gas::IdealGasThermal(UMeanXy(I4), UMeanXy(0), (UMeanXy(Seq123) / UMeanXy(0)).squaredNorm(),
                                 gamma, pMean, asqrMean, Hmean);

            // ! refvalue:

            real muRef = settings.idealGasProperty.muGas;

            real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * UMeanXy(0));
            real mufPhy, muf;
            mufPhy = muf = settings.idealGasProperty.muGas *
                           std::pow(T / settings.idealGasProperty.TRef, 1.5) *
                           (settings.idealGasProperty.TRef + settings.idealGasProperty.CSutherland) /
                           (T + settings.idealGasProperty.CSutherland);
#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
            real fnu1 = 0.;
            if constexpr (model == NS_SA)
            {
                real cnu1 = 7.1;
                real Chi = UMeanXy(I4 + 1) * muRef / mufPhy;
                if (Chi < 10)
                    Chi = 0.05 * std::log(1 + std::exp(20 * Chi));
                real Chi3 = std::pow(Chi, 3);
                fnu1 = Chi3 / (Chi3 + std::pow(cnu1, 3));
                muf *= (1 + Chi * fnu1);
            }

            real k = settings.idealGasProperty.CpGas * muf / settings.idealGasProperty.prGas;
            TDiffU VisFlux;
            VisFlux.resizeLike(DiffUxy);
            VisFlux.setZero();
            Gas::ViscousFlux_IdealGas<dim>(
                UMeanXy, DiffUxy, unitNorm, btype == BoundaryType::Wall_NoSlip,
                settings.idealGasProperty.gamma,
                muf,
                k,
                settings.idealGasProperty.CpGas,
                VisFlux);
            if constexpr (model == NS_SA)
            {
                real sigma = 2. / 3.;
                real lambdaFaceC = sqrt(std::abs(asqrMean)) + std::abs(UL(1) + UR(1)) * 0.5;
                Eigen::Matrix<real, dim, 1> diffRhoNu = DiffUxy(Seq012, {I4 + 1}) * muRef;
                Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
                Eigen::Matrix<real, dim, 1> diffNu = (diffRhoNu - UMeanXy(I4 + 1) * muRef / UMeanXy(0) * diffRho) / UMeanXy(0);

                real cn1 = 16;
                real fn = 1;
                if (UMeanXy(I4 + 1) < 0)
                {
                    real Chi = UMeanXy(I4 + 1) * muRef / mufPhy;
                    fn = (cn1 + std::pow(Chi, 3)) / (cn1 - std::pow(Chi, 3));
                }
                VisFlux(Seq012, {I4 + 1}) = diffNu * (mufPhy + UMeanXy(I4 + 1) * muRef * fn) / sigma / muRef;
            }
#endif

            TU finc;
            finc.resizeLike(ULxy);

            // std::cout << "HERE" << std::endl;
            if (rsType == Setting::RiemannSolverType::HLLEP)
                Gas::HLLEPFlux_IdealGas<dim>(
                    UL, UR, settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace],
                    [&]()
                    {
                        std::cout << "face at" << vfv->faceCenters[iFace].transpose() << '\n';
                        std::cout << "UL" << UL.transpose() << '\n';
                        std::cout << "UR" << UR.transpose() << std::endl;
                    });
            else if (rsType == Setting::RiemannSolverType::HLLC)
                Gas::HLLCFlux_IdealGas_HartenYee<dim>(
                    UL, UR, settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace],
                    [&]()
                    {
                        std::cout << "face at" << vfv->faceCenters[iFace].transpose() << '\n';
                        std::cout << "UL" << UL.transpose() << '\n';
                        std::cout << "UR" << UR.transpose() << std::endl;
                    });
            else if (rsType == Setting::RiemannSolverType::Roe)
                Gas::RoeFlux_IdealGas_HartenYee<dim>(
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

            if constexpr (model == NS_SA)
            {
                // real lambdaFaceC = sqrt(std::abs(asqrMean)) + std::abs(UL(1) / UL(0) + UR(1) / UR(0)) * 0.5;
                real lambdaFaceC = std::abs(UL(1) / UL(0) + UR(1) / UR(0)) * 0.5; //! using velo instead of velo + a
                finc(I4 + 1) = ((UL(1) / UL(0) * UL(I4 + 1) + UR(1) / UR(0) * UR(I4 + 1)) -
                                (UR(I4 + 1) - UL(I4 + 1)) * lambdaFaceC) *
                               0.5;
            }

            finc(Seq123) = normBase * finc(Seq123);
#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
            finc -= VisFlux.transpose() * unitNorm;
#endif

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

        TU source(
            const TU &UMeanXy,
            const TDiffU &DiffUxy,
            index iCell, index ig)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
#ifdef DNDS_FV_EULEREVALUATOR_SOURCE_TERM_ZERO
            TU ret;
            ret.resizeLike(UMeanXy);
            ret.setZero();
            return ret;
#endif
            if constexpr (model == NS || model == NS_2D)
            {
                TU ret;
                ret.resizeLike(UMeanXy);
                ret.setZero();
                ret(Seq123) = settings.constMassForce(Seq012) * UMeanXy(0);
                return ret;
            }
            else if constexpr (model == NS_SA)
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
                Gas::IdealGasThermal(UMeanXy(I4), UMeanXy(0), (UMeanXy(Seq123) / UMeanXy(0)).squaredNorm(),
                                     gamma, pMean, asqrMean, Hmean);
                // ! refvalue:
                real muRef = settings.idealGasProperty.muGas;

                real nuh = UMeanXy(I4 + 1) * muRef / UMeanXy(0);

                real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * UMeanXy(0));
                real mufPhy, muf;
                mufPhy = muf = settings.idealGasProperty.muGas *
                               std::pow(T / settings.idealGasProperty.TRef, 1.5) *
                               (settings.idealGasProperty.TRef + settings.idealGasProperty.CSutherland) /
                               (T + settings.idealGasProperty.CSutherland);

                real Chi = std::abs(UMeanXy(I4 + 1) * muRef / mufPhy);
                real fnu1 = std::pow(Chi, 3) / (std::pow(Chi, 3) + std::pow(cnu1, 3));
                real fnu2 = 1 - Chi / (1 + Chi * fnu1);

                Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
                Eigen::Matrix<real, dim, 1> diffRhoNu = DiffUxy(Seq012, {I4 + 1}) * muRef;
                Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
                Eigen::Matrix<real, dim, 1> diffNu = (diffRhoNu - nuh * diffRho) / UMeanXy(0);
                Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
                Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);

                Eigen::Matrix<real, dim, dim> Omega = 0.5 * (diffU.transpose() - diffU);
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
                if (UMeanXy(I4 + 1) < 0)
                {
                    real cn1 = 16;
                    real Chi = UMeanXy(I4 + 1) * muRef / mufPhy;
                    fn = (cn1 + std::pow(Chi, 3)) / (cn1 - std::pow(Chi, 3));
                    P = cb1 * (1 - ct3) * S * nuh;
                    D = -cw1 * sqr(nuh / d);
                }

                TU ret;
                ret.resizeLike(UMeanXy);
                ret.setZero();
                ret(Seq123) = settings.constMassForce * UMeanXy(0);

                if (passiveDiscardSource)
                    P = D = 0;
                ret(I4 + 1) = UMeanXy(0) * (P - D + diffNu.squaredNorm() * cb2 / sigma) / muRef -
                              (UMeanXy(I4 + 1) * fn * muRef + mufPhy) / (UMeanXy(0) * sigma) * diffRho.dot(diffNu) / muRef;

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
        }

        // zeroth means needs not derivative
        TU sourceJacobianDiag(
            const TU &UMeanXy,
            const TDiffU &DiffUxy,
            index iCell, index ig)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
#ifdef DNDS_FV_EULEREVALUATOR_SOURCE_TERM_ZERO
            TU ret;
            ret.resizeLike(UMeanXy);
            ret.setZero();
            return ret;
#endif
            if constexpr (model == NS || model == NS_2D)
            {
            }
            else if constexpr (model == NS_SA)
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
                Gas::IdealGasThermal(UMeanXy(I4), UMeanXy(0), (UMeanXy(Seq123) / UMeanXy(0)).squaredNorm(),
                                     gamma, pMean, asqrMean, Hmean);
                // ! refvalue:
                real muRef = settings.idealGasProperty.muGas;

                real nuh = UMeanXy(I4 + 1) * muRef / UMeanXy(0);

                real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * UMeanXy(0));
                real mufPhy, muf;
                mufPhy = muf = settings.idealGasProperty.muGas *
                               std::pow(T / settings.idealGasProperty.TRef, 1.5) *
                               (settings.idealGasProperty.TRef + settings.idealGasProperty.CSutherland) /
                               (T + settings.idealGasProperty.CSutherland);

                real Chi = std::abs(UMeanXy(I4 + 1) * muRef / mufPhy);
                real fnu1 = std::pow(Chi, 3) / (std::pow(Chi, 3) + std::pow(cnu1, 3));
                real fnu2 = 1 - Chi / (1 + Chi * fnu1);

                Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
                Eigen::Matrix<real, dim, 1> diffRhoNu = DiffUxy(Seq012, {I4 + 1}) * muRef;
                Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
                Eigen::Matrix<real, dim, 1> diffNu = (diffRhoNu - nuh * diffRho) / UMeanXy(0);
                Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
                Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);

                Eigen::Matrix<real, dim, dim> Omega = 0.5 * (diffU.transpose() - diffU);
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
                if (UMeanXy(I4 + 1) < 0)
                {
                    real cn1 = 16;
                    real Chi = UMeanXy(I4 + 1) * muRef / mufPhy;
                    fn = (cn1 + std::pow(Chi, 3)) / (cn1 - std::pow(Chi, 3));
                    P = cb1 * (1 - ct3) * S * nuh;
                    D = -cw1 * sqr(nuh / d);
                }

                TU ret;
                ret.resizeLike(UMeanXy);
                ret.setZero();

                if (passiveDiscardSource)
                    P = D = 0;
                ret(I4 + 1) = -std::min(UMeanXy(0) * (P * 0 - D * 1) / muRef / (UMeanXy(I4 + 1) + verySmallReal), -verySmallReal);

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
                    std::cout << UMeanXy.transpose() << std::endl;
                    std::cout << pMean << std::endl;
                    std::cout << ret.transpose() << std::endl;

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

            TU Ret;
            Ret.resizeLike(UMeanXy);
            Ret.setZero();
            return Ret;
        }

        /**
         * @brief inviscid flux approx jacobian (flux term not reconstructed / no riemann)
         *
         */
        auto fluxJacobian0_Right(
            TU &UR,
            const TVec &uNorm,
            BoundaryType btype)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            assert(dim == 3); // only for 3D!!!!!!!!
            const TU &U = UR;
            const TVec &n = uNorm;

            real rhoun = n.dot(U({1, 2, 3}));
            real rhousqr = U({1, 2, 3}).squaredNorm();
            real gamma = settings.idealGasProperty.gamma;
            TJacobianU subFdU;
            subFdU.resize(nVars, nVars);

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

            if constexpr (model == NS_SA)
            {
                subFdU(5, 5) = un;
                subFdU(5, 0) = -un * U(5) / U(0);
                subFdU(5, 1) = n(0) * U(5) / U(0);
                subFdU(5, 2) = n(1) * U(5) / U(0);
                subFdU(5, 3) = n(2) * U(5) / U(0);
            }
            return subFdU;
        }

        /**
         * @brief inviscid flux approx jacobian (flux term not reconstructed / no riemann)
         *
         */
        TU fluxJacobian0_Right_Times_du(
            const TU &U,
            const TVec &n,
            BoundaryType btype,
            const TU &dU)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            real gamma = settings.idealGasProperty.gamma;
            TVec velo = U(Seq123) / U(0);
            real p, H, asqr;
            Gas::IdealGasThermal(U(I4), U(0), velo.squaredNorm(), gamma, p, asqr, H);
            TVec dVelo;
            real dp;
            Gas::IdealGasUIncrement<dim>(U, dU, velo, gamma, dVelo, dp);
            TU dF(U.size());
            Gas::GasInviscidFluxFacialIncrement<dim>(
                U, dU,
                n,
                velo, dVelo,
                dp, p,
                dF);
            if (model == NS_SA)
            {
                dF(I4 + 1) = dU(I4 + 1) * n.dot(velo) + U(I4 + 1) * n.dot(dVelo);
            }
            return dF;
        }

        TU generateBoundaryValue(
            const TU &ULxy,
            const TVec &uNorm,
            const TMat &normBase,
            const TVec &pPhysics,
            real t,
            BoundaryType btype)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            assert(ULxy(0) > 0);
            TU URxy;

            if (btype == BoundaryType::Farfield ||
                btype == BoundaryType::Special_DMRFar ||
                btype == BoundaryType::Special_RTFar)
            {
                if (btype == BoundaryType::Farfield)
                {
                    const TU &far = settings.farFieldStaticValue;

                    real un = ULxy(Seq123).dot(uNorm) / ULxy(0);
                    real vsqr = (ULxy(Seq123) / ULxy(0)).squaredNorm();
                    real gamma = settings.idealGasProperty.gamma;
                    real asqr, H, p;
                    Gas::IdealGasThermal(ULxy(I4), ULxy(0), vsqr, gamma, p, asqr, H);

                    assert(asqr >= 0);
                    real a = std::sqrt(asqr);

                    if (un - a > 0) // full outflow
                    {
                        URxy = ULxy;
                    }
                    else if (un > 0) //  1 sonic outflow, 1 sonic inflow, other outflow (subsonic out)
                    {
                        TU farPrimitive, ULxyPrimitive;
                        Gas::IdealGasThermalConservative2Primitive<dim>(far, farPrimitive, gamma);
                        Gas::IdealGasThermalConservative2Primitive<dim>(ULxy, ULxyPrimitive, gamma);
                        ULxyPrimitive(I4) = farPrimitive(I4); // using far pressure
                        Gas::IdealGasThermalPrimitive2Conservative<dim>(ULxyPrimitive, URxy, gamma);
                    }
                    else if (un + a > 0) //  1 sonic outflow, 1 sonic inflow, other inflow (subsonic in)
                    {
                        TU farPrimitive, ULxyPrimitive;
                        Gas::IdealGasThermalConservative2Primitive<dim>(far, farPrimitive, gamma);
                        Gas::IdealGasThermalConservative2Primitive<dim>(ULxy, ULxyPrimitive, gamma);
                        // farPrimitive(0) = ULxyPrimitive(0); // using inner density
                        farPrimitive(I4) = ULxyPrimitive(I4); // using inner pressure
                        Gas::IdealGasThermalPrimitive2Conservative<dim>(farPrimitive, URxy, gamma);
                    }
                    else // full inflow
                    {
                        URxy = settings.farFieldStaticValue;
                    }
                    // URxy = settings.farFieldStaticValue; //!override
                }
                else if (btype == BoundaryType::Special_DMRFar)
                {
                    assert(dim > 1);
                    URxy = settings.farFieldStaticValue;
                    real uShock = 10;
                    if constexpr (dim == 3) //* manual static dispatch
                    {
                        if (((pPhysics(0) - uShock / std::sin(pi / 3) * t - 1. / 6.) -
                             pPhysics(1) / std::tan(pi / 3)) > 0)
                            URxy({0, 1, 2, 3, 4}) = Eigen::Vector<real, 5>{1.4, 0, 0, 0, 2.5};
                        else
                            URxy({0, 1, 2, 3, 4}) = Eigen::Vector<real, 5>{8, 57.157676649772960, -33, 0, 5.635e2};
                    }
                    else
                    {
                        if (((pPhysics(0) - uShock / std::sin(pi / 3) * t - 1. / 6.) -
                             pPhysics(1) / std::tan(pi / 3)) > 0)
                            URxy({0, 1, 2, 3}) = Eigen::Vector<real, 4>{1.4, 0, 0, 2.5};
                        else
                            URxy({0, 1, 2, 3}) = Eigen::Vector<real, 4>{8, 57.157676649772960, -33, 5.635e2};
                    }
                }
                else if (btype == BoundaryType::Special_RTFar)
                {
                    assert(dim > 1);
                    Eigen::VectorXd far = settings.farFieldStaticValue;
                    real gamma = settings.idealGasProperty.gamma;
                    real un = ULxy(Seq123).dot(uNorm) / ULxy(0);
                    real vsqr = (ULxy(Seq123) / ULxy(0)).squaredNorm();
                    real asqr, H, p;
                    Gas::IdealGasThermal(ULxy(I4), ULxy(0), vsqr, gamma, p, asqr, H);

                    assert(asqr >= 0);
                    real a = std::sqrt(asqr);
                    real v = -0.025 * a * cos(pPhysics(0) * 8 * pi);

                    if (pPhysics(1) < 0.5)
                    {

                        real rho = 2;
                        real p = 1;
                        far(0) = rho;
                        far(1) = 0;
                        far(2) = rho * v;
                        far(I4) = 0.5 * rho * sqr(v) + p / (gamma - 1);
                    }
                    else
                    {
                        real rho = 1;
                        real p = 2.5;
                        far(0) = rho;
                        far(1) = 0;
                        far(2) = rho * v;
                        far(I4) = 0.5 * rho * sqr(v) + p / (gamma - 1);
                    }

                    if (un - a > 0) // full outflow
                    {
                        URxy = ULxy;
                    }
                    else if (un > 0) //  1 sonic outflow, 1 sonic inflow, other outflow (subsonic out)
                    {
                        TU farPrimitive, ULxyPrimitive;
                        Gas::IdealGasThermalConservative2Primitive<dim>(far, farPrimitive, gamma);
                        Gas::IdealGasThermalConservative2Primitive<dim>(ULxy, ULxyPrimitive, gamma);
                        ULxyPrimitive(I4) = farPrimitive(I4); // using far pressure
                        Gas::IdealGasThermalPrimitive2Conservative<dim>(ULxyPrimitive, URxy, gamma);
                    }
                    else if (un + a > 0) //  1 sonic outflow, 1 sonic inflow, other inflow (subsonic in)
                    {
                        TU farPrimitive, ULxyPrimitive;
                        Gas::IdealGasThermalConservative2Primitive<dim>(far, farPrimitive, gamma);
                        Gas::IdealGasThermalConservative2Primitive<dim>(ULxy, ULxyPrimitive, gamma);
                        // farPrimitive(0) = ULxyPrimitive(0); // using inner density
                        farPrimitive(I4) = ULxyPrimitive(I4); // using inner pressure
                        Gas::IdealGasThermalPrimitive2Conservative<dim>(farPrimitive, URxy, gamma);
                    }
                    else // full inflow
                    {
                        URxy = settings.farFieldStaticValue;
                    }
                    // URxy = settings.farFieldStaticValue; //!override
                }
                else
                    assert(false);
            }
            else if (btype == BoundaryType::Wall_Euler)
            {
                URxy = ULxy;
                URxy(Seq123) -= URxy(Seq123).dot(uNorm) * uNorm;
            }
            else if (btype == BoundaryType::Wall_NoSlip)
            {
                URxy = ULxy;
                URxy(Seq123) *= -1;
                if (model == NS_SA)
                {
                    URxy(I4 + 1) *= -1;
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

        inline TU CompressRecPart(
            const TU &umean,
            const TU &uRecInc)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
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

            TU ret = umean + uRecInc;
            real eK = ret(Seq123).squaredNorm() * 0.5 / (verySmallReal + std::abs(ret(0)));
            real e = ret(I4) - eK;
            if (e <= 0 || ret(0) <= 0)
                ret = umean;
            if constexpr (model == NS_SA)
                if (ret(I4 + 1) < 0)
                    ret = umean;

            return ret;
        }

        inline TU CompressInc(
            const TU &u,
            const TU &uInc,
            const TU &rhs)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            TU ret = uInc;
            if constexpr (model == NS_SA)
            {
                if (u(I4 + 1) + uInc(I4 + 1) < 0)
                {
                    // std::cout << "Fixing SA inc " << std::endl;

                    assert(u(I4 + 1) >= 0); //! might be bad using gmeres, add this to gmres inc!
                    real declineV = uInc(I4 + 1) / (u(I4 + 1) + verySmallReal);
                    real newu5 = u(I4 + 1) * std::exp(declineV);
                    // ! refvalue:
                    real muRef = settings.idealGasProperty.muGas;
                    newu5 = std::max(1e-12, newu5);
                    ret(I4 + 1) = newu5 - u(I4 + 1);
                }
            }

            return ret;
        }

        void EvaluateDt(std::vector<real> &dt,
                        ArrayDOFV<nVars_Fixed> &u,
                        real CFL, real &dtMinall, real MaxDt = 1,
                        bool UseLocaldt = false);
        /**
         * @brief
         * \param rhs overwritten;
         *
         */
        void EvaluateRHS(ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u,
                         ArrayRecV &uRec, real t);

        void LUSGSADMatrixInit(std::vector<real> &dTau, real dt, real alphaDiag, ArrayDOFV<nVars_Fixed> &u, int jacobianCode = 1,
                               real t = 0);

        void LUSGSADMatrixVec(ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &AuInc);

        void UpdateLUSGSADForward(ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew);

        void UpdateLUSGSADBackward(ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew);

        void LUSGSMatrixInit(std::vector<real> &dTau, real dt, real alphaDiag,
                             ArrayDOFV<nVars_Fixed> &u, ArrayRecV &uRec,
                             int jacobianCode,
                             real t);

        void LUSGSMatrixVec(real alphaDiag,
                            ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &AuInc);

        /**
         * @brief to use LUSGS, use LUSGSForward(..., uInc, uInc); uInc.pull; LUSGSBackward(..., uInc, uInc);
         * the underlying logic is that for index, ghost > dist, so the forward uses no ghost,
         * and ghost should be pulled before using backward;
         * to use Jacobian instead of LUSGS, use LUSGSForward(..., uInc, uIncNew); LUSGSBackward(..., uInc, uIncNew); uIncNew.pull; uInc = uIncNew;
         * \param uIncNew overwritten;
         *
         */
        void UpdateLUSGSForward(real alphaDiag,
                                ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew);

        /**
         * @brief
         * \param uIncNew overwritten;
         *
         */
        void UpdateLUSGSBackward(real alphaDiag,
                                 ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew);

        void UpdateSGS(real alphaDiag,
                       ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew, bool ifForward);

        void FixUMaxFilter(ArrayDOFV<nVars_Fixed> &u);

        void EvaluateResidual(Eigen::Vector<real, -1> &res, ArrayDOFV<nVars_Fixed> &rhs, index P = 1, bool volWise = false);
    };
}