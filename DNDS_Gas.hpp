#pragma once
#include "DNDS_Defines.h"
#include "DNDS_AutoDiff.hpp"
#include "Eigen/Dense"

namespace DNDS
{
    namespace Gas
    {
        typedef Eigen::Vector3d tVec;
        typedef Eigen::Vector2d tVec2;

        /**
         * @brief 3D only warning: ReV should be already initialized
         *
         */
        template <int dim = 3, class TVec, class TeV>
        inline void EulerGasRightEigenVector(const TVec &velo, real Vsqr, real H, real a, TeV &ReV)
        {
            ReV.setZero();
            ReV(0, {0, 1, dim + 1}).setConstant(1);
            ReV(Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>), Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>))
                .diagonal()
                .setConstant(1);

            ReV(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>), {0, 1, dim + 1}).colwise() = velo;
            ReV(1, 0) -= a;
            ReV(1, dim + 1) += a;

            // Last Row
            ReV(dim + 1, 0) = H - velo(0) * a;
            ReV(dim + 1, dim + 1) = H + velo(0) * a;
            ReV(dim + 1, 1) = 0.5 * Vsqr;

            ReV(dim + 1, Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>)) =
                velo(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim - 1>));
        }

        /**
         * @brief 3D only warning: LeV should be already initialized
         *
         */
        template <int dim = 3, class TVec, class TeV>
        inline void EulerGasLeftEigenVector(const TVec &velo, real Vsqr, real H, real a, real gamma, TeV &LeV)
        {
            LeV.setZero();
            real gammaBar = gamma - 1;
            LeV(0, 0) = H + a / gammaBar * (velo(0) - a);
            LeV(0, 1) = -velo(0) - a / gammaBar;
            LeV(0, Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>)) =
                -velo(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim - 1>));
            LeV(0, dim + 1) = 1;

            LeV(1, 0) = -2 * H + 4 / gammaBar * (a * a);
            LeV(1, Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) = velo.transpose() * 2;
            LeV(1, dim + 1) = -2;

            LeV(Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>), 0) =
                velo(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim - 1>)) * (-2 * (a * a) / gammaBar);
            LeV(Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>), Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>))
                .diagonal()
                .setConstant(2 * (a * a) / gammaBar);

            LeV(dim + 1, 0) = H - a / gammaBar * (velo(0) + a);
            LeV(dim + 1, 1) = -velo(0) + a / gammaBar;
            LeV(dim + 1, Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>)) =
                -velo(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim - 1>));
            LeV(dim + 1, dim + 1) = 1;

            LeV *= gammaBar / (2 * a * a);
        }

        inline void IdealGasThermal(
            real E, real rho, real vSqr, real gamma, real &p, real &asqr, real &H)
        {
            p = (gamma - 1) * (E - rho * 0.5 * vSqr);
            asqr = gamma * p / rho;
            H = (E + p) / rho;
        }

        template <int dim = 3, class TCons, class TPrim>
        inline void IdealGasThermalConservative2Primitive(
            const TCons &U, TPrim &prim, real gamma)
        {
            prim = U / U(0);
            real vSqr = (U(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) / U(0)).squaredNorm();
            real rho = U(0);
            real E = U(1 + dim);
            real p = (gamma - 1) * (E - rho * 0.5 * vSqr);
            prim(0) = rho;
            prim(1 + dim) = p;
            assert(rho > 0);
        }

        template <int dim = 3, class TCons, class TPrim>
        inline void IdealGasThermalPrimitive2Conservative(
            const TPrim &prim, TCons &U, real gamma)
        {
            U = prim * prim(0);
            real vSqr = prim(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).squaredNorm();
            real rho = prim(0);
            real p = prim(dim + 1);
            real E = p / (gamma - 1) + rho * 0.5 * vSqr;
            U(0) = rho;
            U(dim + 1) = E;
            assert(rho > 0);
        }

        /**
         * @brief calculates Inviscid Flux for x direction
         *
         */
        template <int dim = 3, typename TU, typename TF, class TVec>
        inline void GasInviscidFlux(const TU &U, const TVec &velo, real p, TF &F)
        {
            F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = U(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) * velo(0); // note that additional flux are unattended!
            F(1) += p;
            F(dim + 1) += velo(0) * p;
        }

        template <int dim = 3, typename TU, class TVec>
        inline void IdealGasUIncrement(const TU &U, const TU &dU, const TVec &velo, real gamma, TVec &dVelo, real &dp)
        {
            dVelo = (dU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) -
                     U(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) / U(0) * dU(0)) /
                    U(0);
            dp = (gamma - 1) * (dU(dim + 1) -
                                0.5 * (dU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).dot(velo) +
                                       U(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).dot(dVelo)));
        } // For Lax-Flux jacobian

        template <int dim = 3, typename TU, typename TF, class TVec>
        inline void GasInviscidFluxFacialIncrement(const TU &U, const TU &dU,
                                                   const TVec &unitNorm,
                                                   const TVec &velo, const TVec &dVelo,
                                                   real dp, real p,
                                                   TF &F)
        {
            real vn = velo.dot(unitNorm);
            real dvn = dVelo.dot(unitNorm);
            F(0) = dU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).dot(unitNorm);
            F(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) =
                dU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) * vn +
                U(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) * dvn + unitNorm * dp;
            F(dim + 1) = (dU(dim + 1) + dp) * vn + (U(dim + 1) + p) * dvn;
        }

        template <int dim = 3, typename TU>
        inline auto IdealGas_EulerGasRightEigenVector(const TU &U, real gamma)
        {
            assert(U(0) > 0);
            Eigen::Matrix<real, dim + 2, dim + 2> ReV;
            Eigen::Vector<real, dim> velo =
                (U(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / U(0)).matrix();
            real vsqr = velo.squaredNorm();
            real asqr, p, H;
            IdealGasThermal(U(dim + 1), U(0), vsqr, gamma, p, asqr, H);
            assert(asqr >= 0);
            EulerGasRightEigenVector<dim>(velo, vsqr, H, std::sqrt(asqr), ReV);
            return ReV;
        }

        template <int dim = 3, typename TU>
        inline auto IdealGas_EulerGasLeftEigenVector(const TU &U, real gamma)
        {
            assert(U(0) > 0);
            Eigen::Matrix<real, dim + 2, dim + 2> LeV;
            Eigen::Vector<real, dim> velo =
                (U(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / U(0)).matrix();
            real vsqr = velo.squaredNorm();
            real asqr, p, H;
            IdealGasThermal(U(dim + 1), U(0), vsqr, gamma, p, asqr, H);
            assert(asqr >= 0);
            EulerGasLeftEigenVector<dim>(velo, vsqr, H, std::sqrt(asqr), gamma, LeV);
            return LeV;
        }

        template <int dim = 3, typename TUL, typename TUR, typename TF, typename TFdumpInfo>
        void HLLEPFlux_IdealGas(const TUL &UL, const TUR &UR, real gamma, TF &F, real dLambda,
                                const TFdumpInfo &dumpInfo)
        {
            static real scaleHartenYee = 0.05;
            using TVec = Eigen::Vector<real, dim>;

            if (!(UL(0) > 0 && UR(0) > 0))
            {
                dumpInfo();
            }
            assert(UL(0) > 0 && UR(0) > 0);
            TVec veloL = (UL(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / UL(0)).matrix();
            TVec veloR = (UR(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / UR(0)).matrix();

            real asqrL, asqrR, pL, pR, HL, HR;
            real vsqrL = veloL.squaredNorm();
            real vsqrR = veloR.squaredNorm();
            IdealGasThermal(UL(dim + 1), UL(0), vsqrL, gamma, pL, asqrL, HL);
            IdealGasThermal(UR(dim + 1), UR(0), vsqrR, gamma, pR, asqrR, HR);
            real sqrtRhoL = std::sqrt(UL(0));
            real sqrtRhoR = std::sqrt(UR(0));

            TVec veloRoe = (sqrtRhoL * veloL + sqrtRhoR * veloR) / (sqrtRhoL + sqrtRhoR);
            real vsqrRoe = veloRoe.squaredNorm();
            real HRoe = (sqrtRhoL * HL + sqrtRhoR * HR) / (sqrtRhoL + sqrtRhoR);
            real asqrRoe = (gamma - 1) * (HRoe - 0.5 * vsqrRoe);
            real rhoRoe = sqrtRhoL * sqrtRhoR;

            if (!(asqrRoe > 0 && asqrL > 0 && asqrR > 0))
            {
                dumpInfo();
            }
            assert((asqrRoe > 0 && asqrL > 0 && asqrR > 0));
            real aRoe = std::sqrt(asqrRoe);

            real lam0 = veloRoe(0) - aRoe;
            real lam123 = veloRoe(0);
            real lam4 = veloRoe(0) + aRoe;
            Eigen::Vector<real, dim + 2> lam;
            lam(0) = lam0;
            lam(dim + 1) = lam4;
            lam(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).setConstant(lam123);
            lam = lam.array().abs();

            //*HY
            // real thresholdHartenYee = std::max(scaleHartenYee * (std::sqrt(vsqrRoe) + aRoe), dLambda);
            // real thresholdHartenYeeS = thresholdHartenYee * thresholdHartenYee;
            // if (std::abs(lam0) < thresholdHartenYee)
            //     lam(0) = (lam0 * lam0 + thresholdHartenYeeS) / (2 * thresholdHartenYee);
            // if (std::abs(lam4) < thresholdHartenYee)
            //     lam(4) = (lam4 * lam4 + thresholdHartenYeeS) / (2 * thresholdHartenYee);
            //*HY
            Eigen::Vector<real, dim + 2> alpha;
            Eigen::Matrix<real, dim + 2, dim + 2> ReVRoe;
            EulerGasRightEigenVector<dim>(veloRoe, vsqrRoe, HRoe, aRoe, ReVRoe);
            // Eigen::Matrix<real, 5, 5> LeVRoe;
            // EulerGasLeftEigenVector(veloRoe, vsqrRoe, HRoe, aRoe, gamma, LeVRoe);
            // alpha = LeVRoe * (UR - UL);
            // std::cout << alpha.transpose() << "\n";

            Eigen::Vector<real, dim + 2> incU =
                UR(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) -
                UL(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>));
            real incP = pR - pL;
            TVec incVelo = veloR - veloL;

            alpha(Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>)) =
                incU(Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>)) -
                veloRoe(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim - 1>)) * incU(0);
            real incU4b = incU(dim + 1) -
                          alpha(Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>))
                              .dot(veloRoe(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim - 1>)));
            alpha(1) = (gamma - 1) / asqrRoe *
                       (incU(0) * (HRoe - veloRoe(0) * veloRoe(0)) +
                        veloRoe(0) * incU(1) - incU4b);
            //! ?
            // alpha(0) = (incU(0) * lam4 - incU(1) - aRoe * alpha(1)) / (2 * aRoe);
            // alpha(dim + 1) = incU(0) - (alpha(0) + alpha(1)); // * HLLEP doesn't need this
            // std::cout << alpha.transpose() << std::endl;
            // std::cout << std::endl;

            real SL = std::min(lam0, veloL(0) - std::sqrt(asqrL));
            real SR = std::max(lam4, veloR(0) + std::sqrt(asqrR));
            real UU = std::abs(veloRoe(0));

            real dfix = aRoe / (aRoe + UU);

            Eigen::Vector<real, dim + 2> FL, FR;
            GasInviscidFlux<dim>(UL, veloL, pL, FL);
            GasInviscidFlux<dim>(UR, veloR, pR, FR);
            real SP = std::max(SR, 0.0);
            real SM = std::min(SL, 0.0);
            real div = SP - SM;
            div += sign(div) * verySmallReal;

            // F = (SP * FL - SM * FR) / div + (SP * SM / div) * (UR - UL - dfix * ReVRoe(Eigen::all, {1, 2, 3}) * alpha({1, 2, 3}));
            F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) =
                (SP * FL - SM * FR) / div +
                (SP * SM / div) *
                    (UR(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) -
                     UL(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) -
                     dfix * ReVRoe(Eigen::all, {1}) * alpha({1}));

            // //! ?

            // real aSound = aRoe;
            // real un_abs = aRoe;
            // real Sp = SP;
            // real Sn = SM;
            // real delta1 = aSound / (un_abs + aSound + verySmallReal);
            // real delta2 = 0.0;
            // real delta3 = 0.0;
            // real un = veloRoe(0);

            // real eV1 = ((Sp + Sn) * un - 2.0 * (1.0 - delta1) * (Sp * Sn)) / (Sp - Sn);
            // real eV2 = ((Sp + Sn) * (un + aSound) - 2.0 * (1.0 - delta2) * (Sp * Sn)) / (Sp - Sn);
            // real eV3 = ((Sp + Sn) * (un - aSound) - 2.0 * (1.0 - delta3) * (Sp * Sn)) / (Sp - Sn);

            // lam(0) = eV3;
            // lam(dim + 1) = eV2;
            // lam(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).setConstant(eV1);
            // lam = lam.array().abs();

            // Eigen::Vector<real, dim + 2> incF = ReVRoe * (lam.array() * alpha.array()).matrix();
            // F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = (FL + FR) * 0.5 - 0.5 * incF;
        }

        template <int dim = 3, typename TUL, typename TUR, typename TF, typename TFdumpInfo>
        void HLLCFlux_IdealGas_HartenYee(const TUL &UL, const TUR &UR, real gamma, TF &F, real dLambda,
                                         const TFdumpInfo &dumpInfo)
        {
            static real scaleHartenYee = 0.05;
            using TVec = Eigen::Vector<real, dim>;

            if (!(UL(0) > 0 && UR(0) > 0))
            {
                dumpInfo();
            }
            assert(UL(0) > 0 && UR(0) > 0);
            TVec veloL = (UL(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / UL(0)).matrix();
            TVec veloR = (UR(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / UR(0)).matrix();

            real asqrL, asqrR, pL, pR, HL, HR;
            real vsqrL = veloL.squaredNorm();
            real vsqrR = veloR.squaredNorm();
            IdealGasThermal(UL(dim + 1), UL(0), vsqrL, gamma, pL, asqrL, HL);
            IdealGasThermal(UR(dim + 1), UR(0), vsqrR, gamma, pR, asqrR, HR);
            real sqrtRhoL = std::sqrt(UL(0));
            real sqrtRhoR = std::sqrt(UR(0));

            TVec veloRoe = (sqrtRhoL * veloL + sqrtRhoR * veloR) / (sqrtRhoL + sqrtRhoR);

            // real lam0 = veloRoe(0) - aRoe;
            // real lam123 = veloRoe(0);
            // real lam4 = veloRoe(0) + aRoe;
            // Eigen::Vector<real, 5> lam = {lam0, lam123, lam123, lam123, lam4};

            real eta2 = 0.5 * (sqrtRhoL * sqrtRhoR) / sqr(sqrtRhoL + sqrtRhoR);
            real dsqr = (asqrL * sqrtRhoL + asqrR * sqrtRhoR) / (sqrtRhoL + sqrtRhoR) + eta2 * sqr(veloR(0) - veloL(0));
            if (!(dsqr > 0))
            {
                dumpInfo();
            }
            assert(dsqr > 0);
            real SL = veloRoe(0) - sqrt(dsqr);
            real SR = veloRoe(0) + sqrt(dsqr);
            dLambda += verySmallReal;
            dLambda *= 2.0;

            // * E-Fix
            // SL += sign(SL) * std::exp(-std::abs(SL) / dLambda) * dLambda;
            // SR += sign(SR) * std::exp(-std::abs(SR) / dLambda) * dLambda;

            Eigen::Vector<real, dim + 2> FL, FR;
            GasInviscidFlux<dim>(UL, veloL, pL, FL);
            GasInviscidFlux<dim>(UR, veloR, pR, FR);

            if (0 <= SL)
            {
                F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = FL;
                return;
            }
            if (SR <= 0)
            {
                F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = FR;
                return;
            }
            real SS = 0;
            real div = (UL(0) * (SL - veloL(0)) - UR(0) * (SR - veloR(0)));
            if (std::abs(div) > verySmallReal)
                SS = (pR - pL + UL(1) * (SL - veloL(0)) - UR(1) * (SR - veloR(0))) / div;
            Eigen::Vector<real, dim + 2> DS;
            DS.setZero();
            DS(1) = 1;
            DS(dim + 1) = SS;
            // SS += sign(SS) * std::exp(-std::abs(SS) / dLambda) * dLambda;
            if (SS >= 0)
            {
                real div = SL - SS;
                if (std::abs(div) < verySmallReal)
                    F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = FL;
                else
                    F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) =
                        ((UL(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) * SL - FL) * SS +
                         DS * ((pL + UL(0) * (SL - veloL(0)) * (SS - veloL(0))) * SL)) /
                        div;
            }
            else
            {
                real div = SR - SS;
                if (std::abs(div) < verySmallReal)
                    F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = FR;
                else
                    F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) =
                        ((UR(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) * SR - FR) * SS +
                         DS * ((pR + UR(0) * (SR - veloR(0)) * (SS - veloR(0))) * SR)) /
                        div;
            }
        }

        // eigScheme: 0 = Roe_HY, 1 = cLLF_M, 2 = lax_0
        template <int dim = 3, int eigScheme = 0, typename TUL, typename TUR, typename TF, typename TFdumpInfo>
        void RoeFlux_IdealGas_HartenYee(const TUL &UL, const TUR &UR, real gamma, TF &F, real dLambda,
                                        const TFdumpInfo &dumpInfo, real &lam0, real &lam123, real &lam4)
        {
            static real scaleHartenYee = 0.05;
            static real scaleLD = 0.2;
            using TVec = Eigen::Vector<real, dim>;

            if (!(UL(0) > 0 && UR(0) > 0))
            {
                dumpInfo();
            }
            assert(UL(0) > 0 && UR(0) > 0);
            TVec veloL = (UL(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / UL(0)).matrix();
            TVec veloR = (UR(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / UR(0)).matrix();

            real asqrL, asqrR, pL, pR, HL, HR;
            real vsqrL = veloL.squaredNorm();
            real vsqrR = veloR.squaredNorm();
            IdealGasThermal(UL(dim + 1), UL(0), vsqrL, gamma, pL, asqrL, HL);
            IdealGasThermal(UR(dim + 1), UR(0), vsqrR, gamma, pR, asqrR, HR);
            real sqrtRhoL = std::sqrt(UL(0));
            real sqrtRhoR = std::sqrt(UR(0));

            TVec veloRoe = (sqrtRhoL * veloL + sqrtRhoR * veloR) / (sqrtRhoL + sqrtRhoR);
            real vsqrRoe = veloRoe.squaredNorm();
            real HRoe = (sqrtRhoL * HL + sqrtRhoR * HR) / (sqrtRhoL + sqrtRhoR);
            real asqrRoe = (gamma - 1) * (HRoe - 0.5 * vsqrRoe);
            real rhoRoe = sqrtRhoL * sqrtRhoR;

            Eigen::Vector<real, dim + 2> FL, FR;
            GasInviscidFlux<dim>(UL, veloL, pL, FL);
            GasInviscidFlux<dim>(UR, veloR, pR, FR);

            if (!(asqrRoe > 0))
            {
                dumpInfo();
            }
            assert(asqrRoe > 0);
            real aRoe = std::sqrt(asqrRoe);

            lam0 = std::abs(veloRoe(0) - aRoe);
            lam123 = std::abs(veloRoe(0));
            lam4 = std::abs(veloRoe(0) + aRoe);

            if constexpr (eigScheme == 0)
            {
                //*HY
                real thresholdHartenYee = std::max(scaleHartenYee * (std::sqrt(vsqrRoe) + aRoe), dLambda);
                real thresholdHartenYeeS = thresholdHartenYee * thresholdHartenYee;
                if (lam0 < thresholdHartenYee)
                    lam0 = (sqr(lam0) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
                if (lam4 < thresholdHartenYee)
                    lam4 = (sqr(lam4) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
                //*HY
            }
            else if constexpr (eigScheme == 1)
            {
                //*LD, cLLF_M
                /**
                 * Nico Fleischmann, Stefan Adami, Xiangyu Y. Hu, Nikolaus A. Adams, A low dissipation method to cure the grid-aligned shock instability, 2020
                 */
                real aL = std::min(std::sqrt(asqrL), std::abs(veloL(0)) / scaleLD);
                real aR = std::min(std::sqrt(asqrR), std::abs(veloR(0)) / scaleLD);
                lam0 = std::max(std::abs(veloL(0) - aL), std::abs(veloR(0) - aR));
                lam123 = std::max(std::abs(veloL(0)), std::abs(veloR(0)));
                lam4 = std::max(std::abs(veloL(0) + aL), std::abs(veloR(0) + aR));
                //*LD, cLLF_M
            }
            else if constexpr (eigScheme == 2)
            {
                // *vanilla Lax
                // lam0 = lam123 = lam4 = std::max({lam0, lam123, lam4});
                lam0 = lam123 = lam4 = std::max(std::abs(veloL(0)) + std::sqrt(asqrL), std::abs(veloR(0)) + std::sqrt(asqrR));
                F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) =
                    (FL + FR) * 0.5 -
                    0.5 * lam0 * (UR(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) - UL(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)));
                return; //* early exit
            }
            else if constexpr (eigScheme == 3)
            {
                //*LD, Roe_M
                /**
                 * Nico Fleischmann, Stefan Adami, Xiangyu Y. Hu, Nikolaus A. Adams, A low dissipation method to cure the grid-aligned shock instability, 2020
                 */
                real LDthreshold = std::abs(veloRoe(0)) / scaleLD;
                real aRoeStar = std::min(LDthreshold, aRoe);
                lam0 = std::abs(veloRoe(0) - aRoeStar);
                lam4 = std::abs(veloRoe(0) + aRoeStar);
            }
            else if constexpr (eigScheme == 4)
            {
                //*ID, Roe_M
                /**
                 * Nico Fleischmann, Stefan Adami, Xiangyu Y. Hu, Nikolaus A. Adams, A low dissipation method to cure the grid-aligned shock instability, 2020
                 */
#ifdef USE_SIGN_MINUS_AT_ROE_M4_FLUX
                real uStar = signM(veloRoe(0)) * std::max(aRoe * scaleLD, std::abs(veloRoe(0)));
#else
                real uStar = sign(veloRoe(0)) * std::max(aRoe * scaleLD, std::abs(veloRoe(0))); //! why signM here?
#endif
                lam0 = std::abs(uStar - aRoe);
                lam123 = std::abs(uStar);
                lam4 = std::abs(uStar + aRoe);

                // real thresholdHartenYee = std::max(scaleLD * (std::sqrt(vsqrRoe) + aRoe), 0);
                // real thresholdHartenYeeS = thresholdHartenYee * thresholdHartenYee;
                // if (lam0 < thresholdHartenYee)
                //     lam0 = (sqr(lam0) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
                // if (lam4 < thresholdHartenYee)
                //     lam4 = (sqr(lam4) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
                // if (lam123 < thresholdHartenYee)
                //     lam123 = (sqr(lam123) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
            }
            else if constexpr (eigScheme == 5)
            {
            }
            else
            {
                assert(false);
            }
            Eigen::Vector<real, dim + 2> lam;
            lam(0) = lam0;
            lam(dim + 1) = lam4;
            lam(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).setConstant(lam123);

            Eigen::Vector<real, dim + 2> alpha;
            Eigen::Matrix<real, dim + 2, dim + 2> ReVRoe;
            EulerGasRightEigenVector<dim>(veloRoe, vsqrRoe, HRoe, aRoe, ReVRoe);
            // std::cout << UL << std::endl;
            // std::cout << UR << std::endl;
            // std::cout << ReVRoe << std::endl;

            Eigen::Vector<real, dim + 2> incU =
                UR(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) -
                UL(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>));
            real incP = pR - pL;
            TVec incVelo = veloR - veloL;

            alpha(Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>)) =
                incU(Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>)) -
                veloRoe(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim - 1>)) * incU(0);
            real incU4b = incU(dim + 1) -
                          alpha(Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>))
                              .dot(veloRoe(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim - 1>)));
            alpha(1) = (gamma - 1) / asqrRoe *
                       (incU(0) * (HRoe - veloRoe(0) * veloRoe(0)) +
                        veloRoe(0) * incU(1) - incU4b);
            alpha(0) = (incU(0) * (veloRoe(0) + aRoe) - incU(1) - aRoe * alpha(1)) / (2 * aRoe);
            alpha(dim + 1) = incU(0) - (alpha(0) + alpha(1)); // * HLLEP doesn't need this

            // * Roe-Modified
            // veloRoe(0) = signM(veloRoe(0)) * lam123;
            // aRoe = std::max(lam0 - lam123, lam4 - lam123);
            // asqrRoe = sqr(aRoe);
            // alpha(Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>)) =
            //     incU(Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>)) -
            //     veloRoe(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim - 1>)) * incU(0);
            // real incU4b = incU(dim + 1) -
            //               alpha(Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>))
            //                   .dot(veloRoe(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim - 1>)));
            // alpha(1) = (gamma - 1) / asqrRoe *
            //            (incU(0) * (HRoe - veloRoe(0) * veloRoe(0)) +
            //             veloRoe(0) * incU(1) - incU4b);
            // alpha(0) = (incU(0) * (veloRoe(0) + aRoe) - incU(1) - aRoe * alpha(1)) / (2 * aRoe);
            // alpha(dim + 1) = incU(0) - (alpha(0) + alpha(1)); // * HLLEP doesn't need this

            // * Roe-HQM
            // real alpha1 = lam123 * (UR(0) - UL(0) - (pR - pL) / asqrRoe);
            // real alpha2 = 0.5 * lam0 * (pR - pL + rhoRoe * aRoe * incVelo(0)) / asqrRoe;
            // real alpha3 = 0.5 * lam4 * (pR - pL - rhoRoe * aRoe * incVelo(0)) / asqrRoe;
            // real alpha4 = alpha1 + alpha2 + alpha3;
            // real alpha5 = aRoe * (alpha2 - alpha3);
            // real alpha6 = 0;
            // Eigen::Vector<real, dim> alpha7s = lam123 * rhoRoe * (veloR - veloL);
            // alpha7s(0) = alpha5;
            // Eigen::Vector<real, dim + 2>
            //     incF;
            // incF(0) = alpha4;
            // incF(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) = veloRoe * alpha4 + alpha7s;
            // incF(dim + 1) = HRoe * alpha4 + veloRoe.dot(alpha7s) - asqrRoe * alpha1 / (gamma - 1);

            // length should be considered
            // flux[0] = (-alpha4);
            // flux[1] = (-uTilde * alpha4 - uNV.x * alpha5 - alpha6);
            // flux[2] = (-vTilde * alpha4 - uNV.y * alpha5 - alpha7);
            // flux[3] = (-HTilde * alpha4 - unTilde * alpha5 - vTilde * alpha7 + aSoundTile * aSoundTile * alpha1 / (gamma - 1));

            // * Roe-Pike
            // alpha(0) = 0.5 / aRoe * (incP - rhoRoe * aRoe * incVelo(0));
            // alpha(1) = incU(0) - incP / sqr(aRoe);
            // alpha(2) = rhoRoe * incVelo(1);
            // alpha(3) = rhoRoe * incVelo(2);
            // alpha(4) = 0.5 / aRoe * (incP + rhoRoe * aRoe * incVelo(0));

            Eigen::Vector<real, dim + 2> incF = ReVRoe * (lam.array() * alpha.array()).matrix();

            F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = (FL + FR) * 0.5 - 0.5 * incF;

            struct t_getRoeFlux
            {
                bool operator()(
                    int NCV, Eigen::Vector<real, 4> &flux,
                    real rL, real uL, real vL, real preL, real HL, real rNuHatL,
                    real rR, real uR, real vR, real preR, real HR, real rNuHatR,
                    real eig1, real eig2, real eig3, real eigSA1, real eigSA2, real eigSA3, real gamma,
                    real rTilde, real uTilde, real vTilde, real aSoundTile, real HTilde,
                    real uNVx, real uNVy, real EPS)
                {
                    struct tunv
                    {
                        real x, y;
                    } uNV{uNVx, uNVy};
                    real dun = uNV.x * (uR - uL) + uNV.y * (vR - vL);
                    real unL = uNV.x * uL + uNV.y * vL;
                    real unR = uNV.x * uR + uNV.y * vR;
                    real runL = rL * unL;
                    real runR = rR * unR;
                    real unTilde = uNV.x * uTilde + uNV.y * vTilde;

                    real alpha1 = eig1 * (rR - rL - (preR - preL) / aSoundTile / aSoundTile);
                    real alpha2 = 0.5 * eig2 * (preR - preL + rTilde * aSoundTile * dun) / aSoundTile / aSoundTile;
                    real alpha3 = 0.5 * eig3 * (preR - preL - rTilde * aSoundTile * dun) / aSoundTile / aSoundTile;
                    real alpha4 = alpha1 + alpha2 + alpha3;
                    real alpha5 = aSoundTile * (alpha2 - alpha3);
                    real alpha6 = eig1 * rTilde * (uR - uL - uNV.x * dun);
                    real alpha7 = eig1 * rTilde * (vR - vL - uNV.y * dun);
                    // length should be considered
                    flux[0] = 0.5 * ((runL + runR) - alpha4);
                    flux[1] = 0.5 * ((runL * uL + runR * uR + uNV.x * (preL + preR)) - uTilde * alpha4 - uNV.x * alpha5 - alpha6);
                    flux[2] = 0.5 * ((runL * vL + runR * vR + uNV.y * (preL + preR)) - vTilde * alpha4 - uNV.y * alpha5 - alpha7);
                    flux[3] = 0.5 * ((runL * HL + runR * HR) - HTilde * alpha4 - unTilde * alpha5 - uTilde * alpha6 - vTilde * alpha7 + aSoundTile * aSoundTile * alpha1 / (gamma - 1));

                    return true;
                }
            } getRoeFlux;

            // Eigen::Vector<real, 4> fluxH;
            // getRoeFlux(-1, fluxH,
            //            UL(0), veloL(0), veloL(1), pL, HL, UnInitReal,
            //            UR(0), veloR(0), veloR(1), pR, HR, UnInitReal,
            //            lam123, lam0, lam4, lam123, lam0, lam4,
            //            gamma, rhoRoe, veloRoe(0), veloRoe(1), aRoe, HRoe,
            //            1, 0, UnInitReal);
            // F(0) = fluxH(0);
            // F(1) = fluxH(1);
            // F(2) = fluxH(2);
            // F(4) = fluxH(3);
        }

        template <typename TUL, typename TUR, typename TF, typename TdFdU, typename TFdumpInfo>
        void RoeFlux_IdealGas_HartenYee_AutoDiff(const TUL &UL, const TUR &UR, real gamma, TF &F,
                                                 TdFdU &dFdUL, TdFdU &dFdUR,
                                                 const TFdumpInfo &dumpInfo)
        {
            using namespace AutoDiff;

            static real scaleHartenYee = 0.05;

            if (!(UL(0) > 0 && UR(0) > 0))
            {
                dumpInfo();
            }
            assert(UL(0) > 0 && UR(0) > 0);
            ADEigenMat ULad(UL), URad(UR);
            ADEigenMat rhoL = ULad({0}, {0});
            ADEigenMat rhoR = URad({0}, {0});
            ADEigenMat veloL = ULad({1, 2, 3}, {0}) / rhoL;
            ADEigenMat veloR = URad({1, 2, 3}, {0}) / rhoR;
            ADEigenMat vsqrL = veloL.dot(veloL);
            ADEigenMat vsqrR = veloR.dot(veloR);
            ADEigenMat EL = ULad({4}, {0});
            ADEigenMat ER = URad({4}, {0});
            ADEigenMat pL = (EL - rhoL * vsqrL * 0.5) * (gamma - 1);
            ADEigenMat pR = (ER - rhoR * vsqrR * 0.5) * (gamma - 1);
            ADEigenMat HL = (EL + pL) / rhoL;
            ADEigenMat HR = (ER + pR) / rhoR;
            ADEigenMat sqrtRhoL = rhoL.sqrt();
            ADEigenMat sqrtRhoR = rhoR.sqrt();
            ADEigenMat sqrtRhoLpR = sqrtRhoL + sqrtRhoR;

            ADEigenMat veloRoe = (veloL * sqrtRhoL + veloR * sqrtRhoR) / sqrtRhoLpR;
            ADEigenMat vsqrRoe = veloRoe.dot(veloRoe);
            ADEigenMat HRoe = (HL * sqrtRhoL + HR * sqrtRhoR) / sqrtRhoLpR;
            ADEigenMat asqrRoe = (HRoe - vsqrRoe * 0.5) * (gamma - 1);
            ADEigenMat rhoRoe = sqrtRhoL * sqrtRhoR;

            // std::cout << asqrRoe.d() << std::endl;
            if (!(asqrRoe.d()(0, 0) > 0))
            {
                dumpInfo();
                assert(false);
            }

            ADEigenMat aRoe = asqrRoe.sqrt();

            ADEigenMat veloRoe0 = veloRoe({0}, {0});
            ADEigenMat veloRoe1 = veloRoe({1}, {0});
            ADEigenMat veloRoe2 = veloRoe({2}, {0});
            ADEigenMat lam0 = (veloRoe0 - aRoe).abs();
            ADEigenMat lam123 = (veloRoe0).abs();
            ADEigenMat lam4 = (veloRoe0 + aRoe).abs();

            //*HY
            ADEigenMat thresholdHartenYee = (vsqrRoe.sqrt() + aRoe) * (scaleHartenYee * 0.5);
            lam0 = lam0.hyFixExp(thresholdHartenYee);
            lam4 = lam4.hyFixExp(thresholdHartenYee);
            //*HY

            ADEigenMat _One{Eigen::MatrixXd{{1}}}, _Zero{Eigen::MatrixXd{{0}}};

            ADEigenMat K0 =
                ADEigenMat::concat0({_One,
                                     veloRoe0 - aRoe,
                                     veloRoe1,
                                     veloRoe({2}, {0}),
                                     HRoe - veloRoe0 * aRoe});
            ADEigenMat K4 =
                ADEigenMat::concat0({_One,
                                     veloRoe0 + aRoe,
                                     veloRoe1,
                                     veloRoe2,
                                     HRoe + veloRoe0 * aRoe});
            ADEigenMat K1 =
                ADEigenMat::concat0({_One,
                                     veloRoe,
                                     vsqrRoe * 0.5});
            ADEigenMat K2 =
                ADEigenMat::concat0({_Zero,
                                     _Zero,
                                     _One,
                                     _Zero,
                                     veloRoe1});
            ADEigenMat K3 =
                ADEigenMat::concat0({_Zero,
                                     _Zero,
                                     _Zero,
                                     _One,
                                     veloRoe2});

            ADEigenMat incRho = rhoR - rhoL;
            ADEigenMat incP = pR - pL;
            ADEigenMat incVelo = veloR - veloL;
            ADEigenMat incVelo0 = incVelo({0}, {0});
            ADEigenMat incVelo1 = incVelo({1}, {0});
            ADEigenMat incVelo2 = incVelo({2}, {0});

            ADEigenMat alpha0 = (incP - rhoRoe * aRoe * incVelo0) / aRoe * 0.5;
            ADEigenMat alpha4 = (incP + rhoRoe * aRoe * incVelo0) / aRoe * 0.5;
            ADEigenMat alpha1 = incRho - incP / asqrRoe;
            ADEigenMat alpha2 = rhoRoe * incVelo1;
            ADEigenMat alpha3 = rhoRoe * incVelo2;

            ADEigenMat veloL0 = veloL({0}, {0});
            ADEigenMat veloR0 = veloR({0}, {0});

            ADEigenMat FL =
                ADEigenMat::concat0({ULad({1}, {0}),
                                     ULad({1}, {0}) * veloL0 + pL,
                                     ULad({2}, {0}) * veloL0,
                                     ULad({3}, {0}) * veloL0,
                                     (EL + pL) * veloL0});
            ADEigenMat FR =
                ADEigenMat::concat0({URad({1}, {0}),
                                     URad({1}, {0}) * veloR0 + pR,
                                     URad({2}, {0}) * veloR0,
                                     URad({3}, {0}) * veloR0,
                                     (ER + pR) * veloR0});
            ADEigenMat FInc = K0 * (alpha0 * lam0) +
                              K4 * (alpha4 * lam4) +
                              K1 * (alpha1 * lam123) +
                              K2 * (alpha2 * lam123) +
                              K3 * (alpha3 * lam123);
            ADEigenMat FOut = (FL + FR - FInc) * 0.5;
            FOut.back();

            F(Eigen::seq(0, 4)) = FOut.d();
            dFdUL = ULad.g();
            dFdUR = URad.g();
            // std::cout << F.transpose() << std::endl;
            // std::cout << dFdUL << std::endl;
            // std::cout << dFdUR << std::endl;
        }

        /**
         * @brief 3x5 TGradU and TFlux
         * GradU is grad of conservatives
         *
         */
        template <int dim = 3, typename TU, typename TGradU, typename TFlux, typename TNorm>
        void ViscousFlux_IdealGas(const TU &U, const TGradU &GradU, TNorm norm, bool adiabatic, real gamma, real mu, real k, real Cp, TFlux &Flux)
        {
            static const auto Seq01234 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>);
            static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
            static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);

            Eigen::Vector<real, dim> velo = U(Seq123) / U(0);
            static const real lambda = -2. / 3.;
            Eigen::Matrix<real, dim, dim> strainRate = (1.0 / sqr(U(0))) *
                                                       (U(0) * GradU(Seq012, Seq123) -
                                                        GradU(Seq012, 0) * Eigen::RowVector<real, dim>(U(Seq123))); // dU_j/dx_i
            Eigen::Vector<real, dim> GradP = (gamma - 1) *
                                             (GradU(Seq012, dim + 1) -
                                              0.5 *
                                                  (GradU(Seq012, Seq123) * velo +
                                                   strainRate * Eigen::Vector<real, dim>(U(Seq123))));
            real vSqr = velo.squaredNorm();
            real p = (gamma - 1) * (U(dim + 1) - U(0) * 0.5 * vSqr);
            Eigen::Vector<real, dim> GradT = (gamma / ((gamma - 1) * Cp * U(0) * U(0))) *
                                             (U(0) * GradP - p * GradU(Seq012, 0));
            if (adiabatic) //! is this fix reasonable?
                GradT -= GradT.dot(norm) * norm;

            Flux(Seq012, 0).setZero();
            Flux(Seq012, Seq123) =
                (strainRate + strainRate.transpose()) * mu +
                Eigen::Matrix<real, dim, dim>::Identity() * (lambda * mu * strainRate.trace());
            // std::cout << "FUCK A.A" << std::endl;
            Flux(Seq012, dim + 1) = Flux(Seq012, Seq123) * velo + k * GradT;
            // std::cout << "FUCK A.B" << std::endl;
        }
    }
}