#pragma once
#include "DNDS_Defines.h"
#include "Eigen/Dense"


namespace DNDS
{
    namespace Gas
    {

        

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
    }
}