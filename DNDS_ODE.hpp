#pragma once
#include "DNDS_Defines.h"
#include "DNDS_MPI.hpp"

namespace DNDS
{
    namespace ODE
    {
        /**
         * @brief
         * \tparam TDATA vec data, need operator*=(std::vector<real>), operator+=(TDATA), operator*=(scalar)
         */
        template <class TDATA>
        class ExplicitSSPRK4LocalDt
        {
            constexpr static real Coef[4] = {
                0.5, 0.5, 1., 1. / 6.};

        public:
            std::vector<real> dt;
            std::vector<real> t;
            std::vector<TDATA> rhsbuf;
            TDATA rhs;
            TDATA xLast;
            TDATA xInc;
            int DOF;

            /**
             * @brief mind that NDOF is the dof of dt
             * finit(TDATA& data)
             */
            template <class Finit>
            ExplicitSSPRK4LocalDt(
                index NDOF, Finit &&finit = [](TDATA &) {}) : DOF(NDOF)
            {
                dt.resize(NDOF);
                t.resize(NDOF);
                rhsbuf.resize(3);
                for (auto &i : rhsbuf)
                    finit(i);
                finit(rhs);
                finit(xLast);
                finit(xInc);
            }

            /**
             * @brief
             * frhs(TDATA&rhs, TDATA&x)
             * fdt(std::vector<real>& dt)
             */
            template <class Frhs, class Fdt>
            void Step(TDATA &x, Frhs &&frhs, Fdt &&fdt)
            {
                fdt(dt);
                xLast = x;

                frhs(rhs, x);
                rhsbuf[0] = rhs;
                rhs *= dt;
                x += rhs;
                // x *= Coef[0] / (1-Coef[0]);
                x += xLast;
                x *= 1 - Coef[0];

                frhs(rhs, x);
                rhsbuf[1] = rhs;
                rhs *= dt;
                x += rhs;
                // x *= Coef[1] / (1 - Coef[1]);
                x += xLast;
                x *= 1 - Coef[1];

                frhs(rhs, x);
                rhsbuf[2] = rhs;
                rhs *= dt;
                x += rhs;
                // x *= Coef[2] / (1 - Coef[2]);
                // x += xLast;
                // x *= 1 - Coef[2];

                frhs(rhs, x);
                rhs += rhsbuf[0];
                rhsbuf[1] *= 2.0;
                rhs += rhsbuf[1];
                rhsbuf[2] *= 2.0;
                rhs += rhsbuf[2];

                rhs *= dt;
                rhs *= Coef[3];

                x = xLast;
                x += rhs;
            }
        };

        template <class TDATA>
        class ExplicitSSPRK3LocalDt
        {
            constexpr static real Coef[3] = {
                1.0, 0.25, 2.0 / 3.0};

        public:
            std::vector<real> dt;
            std::vector<real> t;
            std::vector<TDATA> rhsbuf;
            TDATA rhs;
            TDATA xLast;
            TDATA xInc;
            int DOF;

            /**
             * @brief mind that NDOF is the dof of dt
             * finit(TDATA& data)
             */
            template <class Finit>
            ExplicitSSPRK3LocalDt(
                index NDOF, Finit &&finit = [](TDATA &) {}) : DOF(NDOF)
            {
                dt.resize(NDOF);
                t.resize(NDOF);
                rhsbuf.resize(2);
                for (auto &i : rhsbuf)
                    finit(i);
                finit(rhs);
                finit(xLast);
                finit(xInc);
            }

            /**
             * @brief
             * frhs(TDATA&rhs, TDATA&x)
             * fdt(std::vector<real>& dt)
             */
            template <class Frhs, class Fdt>
            void Step(TDATA &x, Frhs &&frhs, Fdt &&fdt)
            {
                fdt(dt);
                xLast = x;

                frhs(rhs, x);
                rhsbuf[0] = rhs;
                rhs *= dt;
                x += rhs;

                frhs(rhs, x);
                rhsbuf[1] = rhs;
                rhs *= dt;
                x += rhs;
                x *= Coef[1] / (1 - Coef[1]);
                x += xLast;
                x *= 1 - Coef[1];

                frhs(rhs, x);
                // rhsbuf[2] = rhs;
                rhs *= dt;
                x += rhs;
                x *= Coef[2] / (1 - Coef[2]);
                x += xLast;
                x *= 1 - Coef[2];

                // for (int i = 0; i < 10; i++)
                // {
                //     frhs(rhs, x);
                //     if(i == 0)
                //         rhsbuf[0] = rhs;
                //     rhs *= dt;
                //     x += rhs;
                //     x *= 0.2 / (1 - 0.2);
                //     x += xLast;
                //     x *= (1 - 0.2);
                // }
            }
        };
    }

}