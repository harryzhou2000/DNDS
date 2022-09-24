#pragma once

#include "DNDS_Defines.h"
#include "Eigen/Dense"

namespace DNDS
{
    namespace Linear
    {
        template <class TDATA>
        class GMRES_LeftPreconditioned
        {
            uint32_t kSubspace;

            std::vector<TDATA> Vs;
            TDATA V_temp;
            TDATA MLb;

        public:
            template <class Finit>
            GMRES_LeftPreconditioned(
                uint32_t NkSubspace,
                Finit &&finit = [](TDATA &) {}) : kSubspace(NkSubspace)
            {
                Vs.resize(kSubspace + 1);
                for (auto &i : Vs)
                    finit(i);
                finit(V_temp);
                finit(MLb);
            }

            /**
             * @brief
             *
             * @tparam TFA
             * @tparam TFML
             * @tparam TFstop
             * @param FA  void FA(TDATA &x, TDATA &Ax)
             * @param FML void FML(TDATA &x, TDATA &MLx)
             * @param b   rhs
             * @param x   input and output
             * @param nRestart
             * @param FStop bool FStop(iRestart, relRes)
             */
            template <class TFA, class TFML, class TFstop>
            void solve(TFA &&FA, TFML &&FML, TDATA &b, TDATA &x, uint32_t nRestart, TFstop &&FStop)
            {
                FML(b, MLb); // MLb = ML * b
                real scale_MLb = MLb.norm2();
                Eigen::MatrixXd h;
                h.resize(kSubspace + 1, kSubspace);
                uint32_t iRestart;
                for (iRestart = 0; iRestart <= nRestart; iRestart++)
                {
                    FA(x, V_temp);                           // V_temp = A * x
                    FML(V_temp, Vs[0]);                      // Vs[0] = ML * A * x
                    Vs[0].addTo(MLb, -1.0);                  // Vs[0] = ML * A * x - ML * b = -r
                    real beta = Vs[0].norm2();               // beta = norm2(r)
                    if (FStop(iRestart, beta / scale_MLb) || // see if converge
                        std::abs(beta) < verySmallReal ||    // beta is floating-point negligible
                        (iRestart == nRestart))
                        break;
                    Vs[0] *= -1.0 / beta; // Vs[0] = r/norm2(r)
                    // std::cout << Vs[0] << std::endl;

                    for (uint32_t j = 0; j < kSubspace; j++) // Arnoldi
                    {
                        FA(Vs[j], V_temp);      // V_temp = A * Vs[j]
                        FML(V_temp, Vs[j + 1]); // Vs[j + 1] = ML * A * Vs[j]
                        for (uint32_t i = 0; i <= j; i++)
                        {
                            // Gram-Schmidt, calculate projected lengths
                            h(i, j) = Vs[j + 1].dot(Vs[i]); // h(i, j) = dot(ML * A * Vs[j],Vs[i])
                        }
                        for (uint32_t i = 0; i <= j; i++)
                        {
                            // Gram-Schmidt, subtract projections
                            Vs[j + 1].addTo(Vs[i], -h(i, j)); // Vs[j + 1] = ML * A * Vs[j] - sum_{i=0,1,...j}(h(i,j) *  Vs[i])
                        }
                        h(j + 1, j) = Vs[j + 1].norm2(); //
                        Vs[j + 1] *= 1.0 / h(j + 1, j);  // normalize
                    }

                    Eigen::VectorXd eBeta;
                    eBeta.resize(kSubspace + 1);
                    eBeta.setZero();
                    eBeta(0) = beta; // eBeta = e_1 * beta
                    Eigen::VectorXd y = h.fullPivHouseholderQr().solve(eBeta);
                    for (uint32_t j = 0; j < kSubspace; j++) // x = V(:, 0,1,2,...kSubspace-1) * y
                    {
                        x.addTo(Vs[j], y(j));
                    }
                    
                }
            }
        };
    }
}