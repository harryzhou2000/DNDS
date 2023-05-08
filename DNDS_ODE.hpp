#pragma once
#include "DNDS_Defines.h"
#include "DNDS_MPI.hpp"
#include "Eigen/Dense"

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

                frhs(rhs, x, 1, 0.5);
                rhsbuf[0] = rhs;
                rhs *= dt;
                x += rhs;
                // x *= Coef[0] / (1-Coef[0]);
                x += xLast;
                x *= 1 - Coef[0];

                frhs(rhs, x, 1, 0.5);
                rhsbuf[1] = rhs;
                rhs *= dt;
                x += rhs;
                // x *= Coef[1] / (1 - Coef[1]);
                x += xLast;
                x *= 1 - Coef[1];

                frhs(rhs, x, 1, 1);
                rhsbuf[2] = rhs;
                rhs *= dt;
                x += rhs;
                // x *= Coef[2] / (1 - Coef[2]);
                // x += xLast;
                // x *= 1 - Coef[2];

                frhs(rhs, x, 1, 1. / 6.);
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
                frhs(rhs, x, 1, 1);
                rhsbuf[0] = rhs;
                rhs *= dt;
                x += rhs;

                frhs(rhs, x, 1, 0.25);
                rhsbuf[1] = rhs;
                rhs *= dt;
                x += rhs;
                x *= Coef[1] / (1 - Coef[1]);
                x += xLast;
                x *= 1 - Coef[1];

                frhs(rhs, x, 1, 2. / 3.);
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
        class ImplicitDualTimeStep
        {
        public:
            using Frhs = std::function<void(TDATA &, TDATA &, int, real)>;
            using Fdt = std::function<void(std::vector<real> &, real)>;
            using Fsolve = std::function<void(TDATA &, TDATA &, std::vector<real> &, real, real, TDATA &, int)>;
            using Fstop = std::function<bool(int, TDATA &, int)>;

            virtual void Step(TDATA &x, TDATA &xinc, const Frhs &frhs, const Fdt &fdt, const Fsolve &fsolve,
                              int maxIter, const Fstop &fstop, real dt) = 0;
            virtual ~ImplicitDualTimeStep(){}

            virtual TDATA& getLatestRHS() = 0;
        };

        template <class TDATA>
        class ImplicitEulerDualTimeStep : public ImplicitDualTimeStep<TDATA>
        {
        public:
            using Frhs = typename ImplicitDualTimeStep<TDATA>::Frhs;
            using Fdt = typename ImplicitDualTimeStep<TDATA>::Fdt;
            using Fsolve = typename ImplicitDualTimeStep<TDATA>::Fsolve;
            using Fstop = typename ImplicitDualTimeStep<TDATA>::Fstop;

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
             * bool fstop(int iter, TDATA &xinc, int iInternal)
             */
            virtual void Step(TDATA &x, TDATA &xinc, const Frhs &frhs, const Fdt &fdt, const Fsolve &fsolve,
                              int maxIter, const Fstop &fstop, real dt) override
            {
                xLast = x;
                for (int iter = 1; iter <= maxIter; iter++)
                {
                    fdt(dTau);

                    frhs(rhs, x, iter, 1);
                    rhsbuf[0] = rhs;
                    rhs = xLast;
                    rhs -= x;
                    rhs *= 1.0 / dt;
                    rhs += rhsbuf[0]; // crhs = rhs + (x_i - x_j) / dt

                    fsolve(x, rhs, dTau, dt, 1.0, xinc, iter);
                    x += xinc;

                    if (fstop(iter, xinc, 1))
                        break;
                }
            }

            virtual ~ImplicitEulerDualTimeStep() {}

            virtual TDATA &getLatestRHS() override
            {
                return rhsbuf[0];
            }

        };

        template <class TDATA>
        class ImplicitSDIRK4DualTimeStep : public ImplicitDualTimeStep<TDATA>
        {

            static const Eigen::Matrix<real, 3, 3> butcherA;
            static const Eigen::Vector<real, 3> butcherC;
            static const Eigen::RowVector<real, 3> butcherB;

        public:
            using Frhs = typename ImplicitDualTimeStep<TDATA>::Frhs;
            using Fdt = typename ImplicitDualTimeStep<TDATA>::Fdt;
            using Fsolve = typename ImplicitDualTimeStep<TDATA>::Fsolve;
            using Fstop = typename ImplicitDualTimeStep<TDATA>::Fstop;

            std::vector<real> dTau;
            std::vector<TDATA> rhsbuf;
            TDATA rhs;
            TDATA xLast;
            TDATA xIncPrev;
            index DOF;

            /**
             * @brief mind that NDOF is the dof of dt
             * finit(TDATA& data)
             */
            template <class Finit>
            ImplicitSDIRK4DualTimeStep(
                index NDOF, Finit &&finit = [](TDATA &) {}) : DOF(NDOF)
            {
                dTau.resize(NDOF);
                rhsbuf.resize(3);
                for (auto &i : rhsbuf)
                    finit(i);
                finit(rhs);
                finit(xLast);
                finit(xIncPrev);
            }

            /**
             * @brief
             * frhs(TDATA &rhs, TDATA &x)
             * fdt(std::vector<real>& dTau)
             * fsolve(TDATA &x, TDATA &rhs, std::vector<real>& dTau, real dt, real alphaDiag, TDATA &xinc)
             * bool fstop(int iter, TDATA &xinc, int iInternal)
             */
            virtual void Step(TDATA &x, TDATA &xinc, const Frhs &frhs, const Fdt &fdt, const Fsolve &fsolve,
                              int maxIter, const Fstop &fstop, real dt) override
            {
                xLast = x;
                for (int iB = 0; iB < 3; iB++)
                {
                    x = xLast;
                    xIncPrev.setConstant(0.0);
                    int iter = 1;
                    for (; iter <= maxIter; iter++)
                    {
                        fdt(dTau, butcherA(iB, iB));

                        frhs(rhsbuf[iB], x, iter, butcherC(iB));

                        // //!test explicit
                        // rhs = rhsbuf[iB];
                        // rhs *= dTau;
                        // xinc = rhs;
                        // x += xinc;
                        // if (fstop(iter, xinc, iB + 1))
                        //     break;
                        // continue;
                        // //! test explicit

                        // rhsbuf[0] = rhs;
                        rhs = xLast;
                        rhs -= x;
                        rhs *= 1.0 / dt;
                        for (int jB = 0; jB <= iB; jB++)
                            rhs.addTo(rhsbuf[jB], butcherA(iB, jB)); // crhs = rhs + (x_i - x_j) / dt

                        fsolve(x, rhs, dTau, dt, butcherA(iB, iB), xinc, iter);
                        // x += xinc;
                        x.addTo(xinc, 1.0);
                        // x.addTo(xIncPrev, -0.5);

                        xIncPrev = xinc;

                        if (fstop(iter, xinc, iB + 1))
                            break;

                        // TODO: add time dependent rhs
                    }
                    if (iter > maxIter)
                        fstop(iter, xinc, iB + 1);
                }
                x = xLast;
                for (int jB = 0; jB < 3; jB++)
                    x.addTo(rhsbuf[jB], butcherB(jB) * dt);
            }

            virtual TDATA &getLatestRHS() override
            {
                return rhsbuf[0];
            }

            virtual ~ImplicitSDIRK4DualTimeStep() {}
        };

#define _zeta 0.128886400515
        template <class TDATA>
        const Eigen::Matrix<real, 3, 3> ImplicitSDIRK4DualTimeStep<TDATA>::butcherA{
            {_zeta, 0, 0},
            {0.5 - _zeta, _zeta, 0},
            {2 * _zeta, 1 - 4 * _zeta, _zeta}};

        template <class TDATA>
        const Eigen::Vector<real, 3> ImplicitSDIRK4DualTimeStep<TDATA>::butcherC{
            _zeta, 0.5, 1 - _zeta};

        template <class TDATA>
        const Eigen::RowVector<real, 3> ImplicitSDIRK4DualTimeStep<TDATA>::butcherB{
            1. / (6 * sqr(2 * _zeta - 1)),
            (4 * sqr(_zeta) - 4 * _zeta + 2. / 3.) / sqr(2 * _zeta - 1),
            1. / (6 * sqr(2 * _zeta - 1))};
#undef _zeta

        template <class TDATA>
        class ImplicitBDFDualTimeStep : public ImplicitDualTimeStep<TDATA>
        {

            static const Eigen::Matrix<real, 4, 5> BDFCoefs;

        public:
            using Frhs = typename ImplicitDualTimeStep<TDATA>::Frhs;
            using Fdt = typename ImplicitDualTimeStep<TDATA>::Fdt;
            using Fsolve = typename ImplicitDualTimeStep<TDATA>::Fsolve;
            using Fstop = typename ImplicitDualTimeStep<TDATA>::Fstop;

            std::vector<real> dTau;
            std::vector<TDATA> xPrevs;
            Eigen::VectorXd dtPrevs;
            std::vector<TDATA> rhsbuf;
            TDATA xLast;
            TDATA xIncPrev;
            index DOF;
            index cnPrev;
            index prevStart;
            index kBDF;

            /**
             * @brief mind that NDOF is the dof of dt
             * finit(TDATA& data)
             */
            template <class Finit>
            ImplicitBDFDualTimeStep(
                index NDOF, Finit &&finit = [](TDATA &) {},
                index k = 2) : DOF(NDOF), cnPrev(0), prevStart(k - 2), kBDF(k)
            {
                assert(k > 0 && k <= 4);
                dTau.resize(NDOF);
                xPrevs.resize(k - 1);
                dtPrevs.resize(k - 1);
                for (auto &i : xPrevs)
                    finit(i);
                rhsbuf.resize(1);
                finit(rhsbuf[0]);
                finit(xLast);
                finit(xIncPrev);
            }

            /**
             * @brief
             * frhs(TDATA &rhs, TDATA &x)
             * fdt(std::vector<real>& dTau)
             * fsolve(TDATA &x, TDATA &rhs, std::vector<real>& dTau, real dt, real alphaDiag, TDATA &xinc)
             * bool fstop(int iter, TDATA &xinc, int iInternal)
             */
            virtual void Step(TDATA &x, TDATA &xinc, const Frhs &frhs, const Fdt &fdt, const Fsolve &fsolve,
                              int maxIter, const Fstop &fstop, real dt) override
            {
                index kCurrent = cnPrev + 1;
                index prevSiz = kBDF - 1;
                for (index iPrev = 0; iPrev < cnPrev; iPrev++)
                    assert(prevSiz && std::abs(dtPrevs[mod(iPrev + prevStart, prevSiz)] - dt) < dt * 1e-8);

                xLast = x;
                // x = xLast;
                xIncPrev.setConstant(0.0);
                int iter = 1;
                for (; iter <= maxIter; iter++)
                {
                    fdt(dTau, BDFCoefs(kCurrent - 1, 0));

                    frhs(rhsbuf[0], x, iter, 1.0);

                    rhsbuf[0] *= BDFCoefs(kCurrent - 1, 0);
                    rhsbuf[0].addTo(x, -1. / dt);
                    rhsbuf[0].addTo(xLast, BDFCoefs(kCurrent - 1, 1) / dt);
                    // std::cout << "add " << BDFCoefs(kCurrent - 1, 1) << " " << "last" << std::endl;
                    if(prevSiz)
                        for (index iPrev = 0; iPrev < cnPrev; iPrev++)
                        {
                            // std::cout << "add " << BDFCoefs(kCurrent - 1, 2 + iPrev) <<" " << mod(iPrev + prevStart, prevSiz) << std::endl;
                            rhsbuf[0].addTo(xPrevs[mod(iPrev + prevStart, prevSiz)], BDFCoefs(kCurrent - 1, 2 + iPrev) / dt);
                        }

                    fsolve(x, rhsbuf[0], dTau, dt, BDFCoefs(kCurrent - 1, 0), xinc, iter);
                    //* xinc = (I/dtau-A*alphaDiag)\rhs

                    // std::cout << "BDF::\n";
                    // std::cout << kCurrent << " " << cnPrev<<" " << BDFCoefs(kCurrent - 1, 0) << std::endl;

                    // x += xinc;
                    x.addTo(xinc, 1.0);
                    // x.addTo(xIncPrev, -0.5);

                    xIncPrev = xinc;

                    if (fstop(iter, xinc, 1))
                        break;
                }
                if (iter > maxIter)
                    fstop(iter, xinc, 1);
                if (prevSiz)
                {
                    prevStart = mod(prevStart - 1, prevSiz);
                    // std::cout << dtPrevs.size() << " " << prevStart << std::endl;
                    xPrevs[prevStart] = xLast;
                    dtPrevs[prevStart] = dt;
                    cnPrev = std::min(cnPrev + 1, prevSiz);
                }
            }

            virtual TDATA &getLatestRHS() override
            {
                return rhsbuf[0];
            }

            virtual ~ImplicitBDFDualTimeStep() {}
        };

        template <class TDATA>
        const Eigen::Matrix<real, 4, 5> ImplicitBDFDualTimeStep<TDATA>::BDFCoefs{
            {1. / 1., 1. / 1., 0. / 0., 0. / 0., 0. / 0.},
            {2. / 3., 4. / 3., -1. / 3., 0. / 0., 0. / 0.},
            {6. / 11., 18. / 11., -9. / 11., 2. / 11., 0. / 0.},
            {12. / 25., 48. / 25., -36. / 25., 16. / 25., -3. / 25.}};
    }

}