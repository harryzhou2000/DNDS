#pragma once
#include "DNDS_Defines.h"
#include "Eigen/Dense"

namespace DNDS
{
    namespace Gas
    {
        typedef Eigen::Vector3d tVec;

        /**
         * @brief 3D only warning: ReV should be already initialized
         *
         */
        template <typename TeV>
        inline void EulerGasRightEigenVector(const tVec &velo, real Vsqr, real H, real a, TeV &ReV)
        {
            ReV.setZero();
            ReV(0, 0) = ReV(0, 1) = ReV(2, 2) = ReV(3, 3) = ReV(0, 4) = 1;

            ReV({1, 2, 3}, 0) = ReV({1, 2, 3}, 1) = ReV({1, 2, 3}, 4) = velo;
            ReV(1, 0) -= a;
            ReV(1, 4) += a;

            // Last Row
            ReV(4, 0) = H - velo(0) * a;
            ReV(4, 4) = H + velo(0) * a;
            ReV(4, 1) = 0.5 * Vsqr;
            ReV(4, 2) = velo(1);
            ReV(4, 3) = velo(2);
        }

        /**
         * @brief 3D only warning: LeV should be already initialized
         *
         */
        template <typename TeV>
        inline void EulerGasLeftEigenVector(const tVec &velo, real Vsqr, real H, real a, real gamma, TeV &LeV)
        {
            LeV.setZero();
            real gammaBar = gamma - 1;
            LeV(0, 0) = H + a / gammaBar * (velo(0) - a);
            LeV(0, 1) = -velo(0) - a / gammaBar;
            LeV(0, 2) = -velo(1), LeV(0, 3) = -velo(2);
            LeV(0, 4) = 1;

            LeV(1, 0) = -2 * H + 4 / gammaBar * (a * a);
            LeV(1, {1, 2, 3}) = velo.transpose() * 2;
            LeV(1, 4) = -2;

            LeV(2, 0) = -2 * velo(1) * (a * a) / gammaBar;
            LeV(2, 2) = 2 * (a * a) / gammaBar;

            LeV(3, 0) = -2 * velo(2) * (a * a) / gammaBar;
            LeV(3, 3) = 2 * (a * a) / gammaBar;

            LeV(4, 0) = H - a / gammaBar * (velo(0) + a);
            LeV(4, 1) = -velo(0) + a / gammaBar;
            LeV(4, 2) = -velo(1), LeV(4, 3) = -velo(2);
            LeV(4, 4) = 1;

            LeV *= gammaBar / (2 * a * a);
        }

        inline void IdealGasThermal(real E, real rho, real vSqr, real gamma, real &p, real &asqr, real &H)
        {
            p = (gamma - 1) * (E - rho * 0.5 * vSqr);
            asqr = gamma * p / rho;
            H = (E + p) / rho;
        }

        template <typename TU, typename TF>
        inline void GasInviscidFlux(const TU &U, const tVec &velo, real p, TF &F)
        {
            F = U * velo(0);
            F(1) += p;
            F(4) += velo(0) * p;
        }

        template <typename TU>
        inline auto IdealGas_EulerGasRightEigenVector(const TU &U, real gamma)
        {
            assert(U(0) > 0);
            Eigen::Matrix<real, 5, 5> ReV;
            tVec velo = (U({1, 2, 3}).array() / U(0)).matrix();
            real vsqr = velo.squaredNorm();
            real asqr, p, H;
            IdealGasThermal(U(4), U(0), vsqr, gamma, p, asqr, H);
            assert(asqr >= 0);
            EulerGasRightEigenVector(velo, vsqr, H, std::sqrt(asqr), ReV);
            return ReV;
        }

        template <typename TU>
        inline auto IdealGas_EulerGasLeftEigenVector(const TU &U, real gamma)
        {
            assert(U(0) > 0);
            Eigen::Matrix<real, 5, 5> LeV;
            tVec velo = (U({1, 2, 3}).array() / U(0)).matrix();
            real vsqr = velo.squaredNorm();
            real asqr, p, H;
            IdealGasThermal(U(4), U(0), vsqr, gamma, p, asqr, H);
            assert(asqr >= 0);
            EulerGasLeftEigenVector(velo, vsqr, H, std::sqrt(asqr), gamma, LeV);
            return LeV;
        }

        template <typename TUL, typename TUR, typename TF, typename TFdumpInfo>
        void RoeFlux_IdealGas_HartenYee(const TUL &UL, const TUR &UR, real gamma, TF &F, const TFdumpInfo &dumpInfo)
        {
            static real scaleHartenYee = 0.01;

            if (!(UL(0) > 0 && UR(0) > 0))
            {
                dumpInfo();
            }
            assert(UL(0) > 0 && UR(0) > 0);
            tVec veloL = (UL({1, 2, 3}).array() / UL(0)).matrix();
            tVec veloR = (UR({1, 2, 3}).array() / UR(0)).matrix();
            real asqrL, asqrR, pL, pR, HL, HR;
            real vsqrL = veloL.squaredNorm();
            real vsqrR = veloR.squaredNorm();
            IdealGasThermal(UL(4), UL(0), vsqrL, gamma, pL, asqrL, HL);
            IdealGasThermal(UR(4), UR(0), vsqrR, gamma, pR, asqrR, HR);
            real sqrtRhoL = std::sqrt(UL(0));
            real sqrtRhoR = std::sqrt(UR(0));

            tVec veloRoe = (sqrtRhoL * veloL + sqrtRhoR * veloR) / (sqrtRhoL + sqrtRhoR);
            real vsqrRoe = veloRoe.squaredNorm();
            real HRoe = (sqrtRhoL * HL + sqrtRhoR * HR) / (sqrtRhoL + sqrtRhoR);
            real asqrRoe = (gamma - 1) * (HRoe - 0.5 * vsqrRoe);

            if (!(asqrRoe > 0))
            {
                dumpInfo();
            }
            assert(asqrRoe > 0);
            real aRoe = std::sqrt(asqrRoe);

            real lam0 = veloRoe(0) - aRoe;
            real lam123 = veloRoe(0);
            real lam4 = veloRoe(0) + aRoe;
            Eigen::Vector<real, 5> lam = {lam0, lam123, lam123, lam123, lam4};
            lam = lam.array().abs();

            //*HY
            real thresholdHartenYee = scaleHartenYee * (std::sqrt(vsqrRoe) + aRoe);
            real thresholdHartenYeeS = thresholdHartenYee * thresholdHartenYee;
            if (std::abs(lam0) < thresholdHartenYee)
                lam(0) = (lam0 * lam0 + thresholdHartenYeeS) / (2 * thresholdHartenYee);
            if (std::abs(lam4) < thresholdHartenYee)
                lam(4) = (lam4 * lam4 + thresholdHartenYeeS) / (2 * thresholdHartenYee);
            //*HY

            Eigen::Matrix<real, 5, 5> ReVRoe;
            EulerGasRightEigenVector(veloRoe, vsqrRoe, HRoe, aRoe, ReVRoe);

            Eigen::Vector<real, 5> incU = UR - UL;
            Eigen::Vector<real, 5> alpha;

            alpha(2) = incU(2) - veloRoe(1) * incU(0);
            alpha(3) = incU(3) - veloRoe(2) * incU(0);
            real incU4b = incU(4) - alpha(2) * veloRoe(1) - alpha(3) * veloRoe(2);
            alpha(1) = (gamma - 1) / asqrRoe *
                       (incU(0) * (HRoe - veloRoe(0) * veloRoe(0)) +
                        veloRoe(0) * incU(1) - incU4b);
            alpha(0) = (incU(0) * lam4 - incU(1) - aRoe * alpha(1)) / (2 * aRoe);
            alpha(4) = incU(0) - (alpha(0) + alpha(1));

            Eigen::Vector<real, 5> incF = ReVRoe * (lam.array() * alpha.array()).matrix();
            Eigen::Vector<real, 5> FL, FR;
            GasInviscidFlux(UL, veloL, pL, FL);
            GasInviscidFlux(UR, veloR, pR, FR);
            F = (FL + FR) * 0.5 - 0.5 * incF;
        }

        /**
         * @brief 3x5 TGradU and TFlux
         * GradU is grad of conservatives
         *
         */
        template <typename TU, typename TGradU, typename TFlux>
        void ViscousFlux_IdealGas(const TU &U, const TGradU &GradU, real gamma, real mu, real k, real Cp, TFlux &Flux)
        {
            Eigen::MatrixXd A;
            Eigen::Vector3d velo = U({1, 2, 3}) / U(0);
            static const real lambda = -2. / 3.;
            Eigen::Matrix3d strainRate = (1.0 / (U(0) * U(0))) *
                                         (U(0) * GradU({0, 1, 2}, {1, 2, 3}) -
                                          GradU({0, 1, 2}, 0) * Eigen::RowVector3d(U({1, 2, 3}))); // dU_j/dx_i
            Eigen::Vector3d GradP = (gamma - 1) *
                                    (GradU({0, 1, 2}, 4) -
                                     0.5 *
                                         (GradU({0, 1, 2}, {1, 2, 3}) * velo +
                                          strainRate * Eigen::Vector3d(U({1, 2, 3}))));
            real vSqr = velo.squaredNorm();
            real p = (gamma - 1) * (U(4) - U(0) * 0.5 * vSqr);
            Eigen::Vector3d GradT = (gamma / ((gamma - 1) * Cp * U(0) * U(0))) *
                                    (U(0) * GradP - p * GradU({0, 1, 2}, 0));

            Flux({0, 1, 2}, 0).setZero();
            Flux({0, 1, 2}, {1, 2, 3}) =
                (strainRate + strainRate.transpose()) * mu +
                Eigen::Matrix3d::Identity() * (lambda * mu * strainRate.trace());
            auto viscousStress = Flux({0, 1, 2}, {1, 2, 3});
            Flux({0, 1, 2}, 4) = viscousStress * velo + k * GradT;
        }
    }
}