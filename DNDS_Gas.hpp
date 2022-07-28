#pragma once
#include "DNDS_Defines.h"
#include "Eigen/Dense"

namespace DNDS
{
    namespace Gas
    {
        typedef Eigen::Vector3d tVec;

        template <uint32_t dim>
        void RoeFluxT(Eigen::VectorXd &UL, Eigen::VectorXd &UR, real gamma, real tEF, Eigen::VectorXd &F, real &vmax, int &aim, real distance)
        {
            bool retcent = false;
            if (UL[0] <= 0 || UR[0] <= 0)
            {
                std::cerr << "===RoeFlux===: RHO not Positive!!!" << std::endl;
                if (UL[0] <= 0)
                {
                    UL[0] = 1e-10;
                    for (int i = 0; i < dim; i++)
                        UL[i + 1] = 1e-20;
                }
                if (UR[0] <= 0)
                {
                    UR[0] = 1e-10;
                    for (int i = 0; i < dim; i++)
                        UR[i + 1] = 1e-20;
                }
                retcent = true;
            }
            real rsvL = VecUSQR<dim>(UL) / (UL[0]);
            real rsvR = VecUSQR<dim>(UR) / (UR[0]);
            real pL = (UL[dim + 1] - 0.5 * rsvL) * (gamma - 1);
            real pR = (UR[dim + 1] - 0.5 * rsvR) * (gamma - 1);
            real HL = (UL[dim + 1] + pL) / UL[0];
            real HR = (UR[dim + 1] + pR) / UR[0];
            utype<dim> Umid(0.0);
            for (int i = 0; i < dim; i++)
                Umid[i] = (UL[i + 1] / sqrt(UL[0]) + UR[i + 1] / sqrt(UR[0])) / (sqrt(UL[0]) + sqrt(UR[0]));
            real Hmid = (HL * sqrt(UL[0]) + HR * sqrt(UR[0])) / (sqrt(UL[0]) + sqrt(UR[0]));
            real qsmid = Umid.dot(Umid);
            real asmid = (gamma - 1) * (Hmid - 0.5 * qsmid);
            if (asmid < __FLT_MIN__)
            {
                aim = 1;
                asmid = std::abs(asmid);
            }

            // get eigen vectors
            utype<dim + 2> K[dim + 2];
            // k0
            K[0][0] = 1;
            K[0][1] = Umid[0] - sqrt(asmid);
            for (int i = 1; i < dim; i++)
                K[0][i + 1] = Umid[i];
            K[0][dim + 1] = Hmid - Umid[0] * sqrt(asmid);
            // k1
            K[1][0] = 1;
            for (int i = 0; i < dim; i++)
                K[1][i + 1] = Umid[i];
            K[1][dim + 1] = 0.5 * qsmid;
            // k2~kd
            for (int j = 1; j < dim; j++)
            {
                K[j + 1][0] = K[j + 1][1] = 0;
                for (int i = 1; i < dim; i++)
                    K[j + 1][i + 1] = j == i ? 1 : 0;
                K[j + 1][dim + 1] = Umid[j];
            }
            // kd+1
            K[dim + 1][0] = 1;
            K[dim + 1][1] = Umid[0] + sqrt(asmid);
            for (int i = 1; i < dim; i++)
                K[dim + 1][i + 1] = Umid[i];
            K[dim + 1][dim + 1] = Hmid + Umid[0] * sqrt(asmid);

            // geta
            utype<dim + 2> incU = UR - UL;
            real a[dim + 2];
            for (int i = 1; i < dim; i++)
                a[i + 1] = incU[i + 1] - Umid[i] * incU[0];
            real inc4 = incU[dim + 1];
            for (int i = 1; i < dim; i++)
                inc4 -= a[i + 1] * Umid[i];

            a[1] = (gamma - 1) / asmid * (incU[0] * (Hmid - sqr(Umid[0])) + Umid[0] * incU[1] - inc4);
            a[0] = 0.5 * (incU[0] * (Umid[0] + sqrt(asmid)) - incU[1] - sqrt(asmid) * a[1]) / sqrt(asmid);
            a[dim + 1] = incU[0] - (a[0] + a[1]);

            // lambdas & harten-yee
            utype<dim + 2> Lams;
            Lams[0] = Umid[0] - sqrt(asmid);
            for (int i = 0; i < dim; i++)
                Lams[i + 1] = Umid[0];
            Lams[dim + 1] = Umid[0] + sqrt(asmid);
            real deltaEntropy = tEF * (sqrt(qsmid) + sqrt(asmid));
            for (int i = 0; i < dim + 2; i++)
                if (std::abs(Lams[i]) < deltaEntropy)
                    Lams[i] = (sqr(Lams[i]) + sqr(deltaEntropy)) / deltaEntropy * 0.5 * sign(Lams[i]);

            // sums
            utype<dim + 2> FL, FR;
            EulerFlux<dim>(UL, set, FL), EulerFlux<dim>(UR, set, FR);
            F = FL + FR;
            vmax = Lams[1] / distance;
            if (asmid != 0 && (!retcent))
            {
                for (int i = 0; i < dim + 2; i++)
                    F -= K[i] * (a[i] * std::abs(Lams[i]));
                vmax = std::max(std::abs(Lams[0]), std::abs(Lams[dim + 1])) / distance;
            }

            F *= 0.5;
            for (int i = 0; i < dim + 2; i++)
                if (std::isnan(F[i]) || std::isinf(F[i]))
                {
                    std::cerr << "===RoeFlux===: Return Value " << i << " Inf or Nan!!!" << std::endl;
                    exit(1);
                    return;
                }
        }

        /**
         * @brief 3D only warning: ReV should be already initialized
         *
         */
        template <typename TReV>
        inline void EulerGasRightEigenVector(const tVec &velo, real Vsqr, real H, real a, TReV &ReV)
        {
            Eigen::MatrixXd d;
            ReV.setZero();
            ReV(0, 0) = ReV(0, 1) = ReV(2, 2) = ReV(3, 3) = ReV(0, 4) = 1;

            ReV({1, 2, 3}, 0) = ReV({1, 2, 3}, 1) = ReV({1, 2, 3}, 2) = velo;
            ReV(1, 0) += a;
            ReV(1, 4) -= a;

            // Last Row
            ReV(4, 0) = H - velo(0) * a;
            ReV(4, 4) = H + velo(0) * a;
            ReV(4, 1) = 0.5 * Vsqr;
            ReV(4, 2) = velo(1);
            ReV(4, 3) = velo(2);
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

        template <typename TUL, typename TUR, typename TF>
        void RoeFlux_IdealGas_HartenYee(const TUL &UL, const TUR &UR, real gamma, TF &F)
        {
            static real scaleHartenYee = 0.1;

            assert(UL(0) > 0 && UR(0) > 0);
            tVec veloL = UL({1, 2, 3}) / UL(0);
            tVec veloR = UR({1, 2, 3}) / UR(0);
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

            Eigen::Matrix<real, 5, 5>
                ReVRoe;
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
    }
}