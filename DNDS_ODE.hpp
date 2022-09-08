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
            std::vector<TDATA> rhsbuf;
            TDATA rhs;
            TDATA xLast;
            TDATA xInc;
            index DOF;

            /**
             * @brief mind that NDOF is the dof of dt
             * finit(TDATA& data)
             */
            template <class Finit>
            ExplicitSSPRK4LocalDt(
                index NDOF, Finit &&finit = [](TDATA &) {}) : DOF(NDOF)
            {
                dt.resize(NDOF);
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
            std::vector<TDATA> rhsbuf;
            TDATA rhs;
            TDATA xLast;
            TDATA xInc;
            index DOF;

            /**
             * @brief mind that NDOF is the dof of dt
             * finit(TDATA& data)
             */
            template <class Finit>
            ExplicitSSPRK3LocalDt(
                index NDOF, Finit &&finit = [](TDATA &) {}) : DOF(NDOF)
            {
                dt.resize(NDOF);
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

                //* /////////////////
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
                //* /////////////////

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

                // * /////////////////
                // frhs(rhs, x);
                // rhsbuf[0] = rhs;
                // rhs *= dt;
                // x += rhs;
                // * /////////////////
            }
        };

        template <class TDATA>
        class ImplicitEulerDualTimeStep
        {
        public:
            std::vector<real> dTau;
            std::vector<TDATA> rhsbuf;
            TDATA rhs;
            TDATA xLast;
            TDATA xInc;
            index DOF;

            /**
             * @brief mind that NDOF is the dof of dt
             * finit(TDATA& data)
             */
            template <class Finit>
            ImplicitEulerDualTimeStep(
                index NDOF, Finit &&finit = [](TDATA &) {}) : DOF(NDOF)
            {
                dTau.resize(NDOF);
                rhsbuf.resize(1);
                for (auto &i : rhsbuf)
                    finit(i);
                finit(rhs);
                finit(xLast);
                finit(xInc);
            }

            /**
             * @brief
             * frhs(TDATA &rhs, TDATA &x)
             * fdt(std::vector<real>& dTau)
             * fsolve(TDATA &x, TDATA &rhs, std::vector<real>& dTau, real dt, real alphaDiag, TDATA &xinc)
             * bool fstop(int iter, TDATA &xinc)
             */
            template <class Frhs, class Fdt, class Fsolve, class Fstop>
            void Step(TDATA &x, TDATA &xinc, Frhs &&frhs, Fdt &&fdt, Fsolve &&fsolve,
                      int maxIter, Fstop &&fstop, real dt)
            {
                xLast = x;
                for (int iter = 1; iter <= maxIter; iter++)
                {
                    fdt(dTau);

                    frhs(rhs, x);
                    rhsbuf[0] = rhs;
                    rhs = xLast;
                    rhs -= x;
                    rhs *= 1.0 / dt;
                    rhs += rhsbuf[0]; // crhs = rhs + (x_i - x_j) / dt

                    fsolve(x, rhs, dTau, dt, 1.0, xinc);
                    x += xinc;

                    if (fstop(iter, xinc))
                        break;
                }
            }
        };

        template <class TDATA>
        class ImplicitSDIRK4
        {
            static const real _zeta = 0.128886400515;
            static const Eigen::Matrix<real, 3, 3> butcherA{
                {_zeta, 0, 0},
                {0.5 - _zeta, _zeta, 0},
                {2 * _zeta, 1 - 4 * _zeta, _zeta}};
            static const Eigen::Vector<real, 3> butcherC{
                _zeta, 0.5, 1 - _zeta};
            static const Eigen::RowVector<real, 3> butcherB{
                1. / (6 * sqr(2 * _zeta - 1)),
                (4 * sqr(_zeta) - 4 * _zeta + 2. / 3.) / sqr(2 * _zeta - 1),
                1. / (6 * sqr(2 * _zeta - 1))};

        public:
            std::vector<real> dTau;
            std::vector<TDATA> rhsbuf;
            TDATA rhs;
            TDATA xLast;
            TDATA xInc;
            index DOF;

            /**
             * @brief mind that NDOF is the dof of dt
             * finit(TDATA& data)
             */
            template <class Finit>
            ImplicitSDIRK4(
                index NDOF, Finit &&finit = [](TDATA &) {}) : DOF(NDOF)
            {
                dTau.resize(NDOF);
                rhsbuf.resize(3);
                for (auto &i : rhsbuf)
                    finit(i);
                finit(rhs);
                finit(xLast);
                finit(xInc);
            }

            /**
             * @brief
             * frhs(TDATA &rhs, TDATA &x)
             * fdt(std::vector<real>& dTau)
             * fsolve(TDATA &x, TDATA &rhs, std::vector<real>& dTau, real dt, real alphaDiag, TDATA &xinc)
             * bool fstop(int iter, TDATA &xinc)
             */
            template <class Frhs, class Fdt, class Fsolve, class Fstop>
            void Step(TDATA &x, TDATA &xinc, Frhs &&frhs, Fdt &&fdt, Fsolve &&fsolve,
                      int maxIter, Fstop &&fstop, real dt)
            {
                xLast = x;
                for (int iB = 0; iB < 3; iB++)
                {
                    for (int iter = 1; iter <= maxIter; iter++)
                    {
                        fdt(dTau);

                        frhs(rhsbuf[iB], x);
                        // rhsbuf[0] = rhs;
                        rhs = xLast;
                        rhs -= x;
                        rhs *= 1.0 / dt;
                        for (int jB = 0; jB <= iB; jB++)
                            rhs.addTo(rhsbuf[iB], butcherA(iB, jB)); // crhs = rhs + (x_i - x_j) / dt

                        fsolve(x, rhs, dTau, dt, butcherA(iB, iB), xinc);
                        x += xinc;

                        if (fstop(iter, xinc))
                            break;

                        // TODO: add time dependent rhs
                    }
                }
                x = xLast;
                for (int jB = 0; jB < 3; jB++)
                    x.addTo(rhsbuf[jB], butcherB(jB) * dt);
            }
        };
    }

}