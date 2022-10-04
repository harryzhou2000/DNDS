#pragma once

#include "Eigen/Dense"

#include "DNDS_Defines.h"
#include <functional>
#include <iostream>
#include <memory>

namespace DNDS
{
    namespace AutoDiff
    {
        class ADEigenMat
        {
        public:
            class OpBase
            {

            public:
                Eigen::MatrixXd d, g;
                virtual void back() = 0;
                virtual ~OpBase() {}
                int nGrads = 0;
                void InitGrad(int newNGrads)
                {
                    if (nGrads != newNGrads)
                    {
                        nGrads = newNGrads;
                        int sz0 = d.rows();
                        int sz1 = d.cols();
                        g.resize(sz0, sz1 * nGrads);
                        g.setZero();
                    }
                }
            };

        private:
            std::shared_ptr<OpBase> op;

        public:
            typedef std::shared_ptr<OpBase> pData;

            class OpIn : public OpBase
            {
            public:
                OpIn() {}
                void calc(const Eigen::MatrixXd &in)
                {
                    d = in;
                }
                void back() override {}
            };
            class OpCopy : public OpBase
            {
                pData d0;

            public:
                OpCopy(const pData &from) : d0(from) {}
                void calc()
                {
                    d = d0->d;
                }
                void back() override
                {
                    d0->InitGrad(nGrads);
                    d0->g += g;
                }
            };
            class OpAdd : public OpBase
            {
                pData da, db;

            public:
                OpAdd(const pData &a, const pData &b) : da(a), db(b) {}
                void calc()
                {
                    d = da->d + db->d;
                }
                void back()
                {
                    da->InitGrad(nGrads);
                    db->InitGrad(nGrads);
                    da->g += g;
                    db->g += g;
                }
            };
            class OpSubs : public OpBase
            {
                pData da, db;

            public:
                OpSubs(const pData &a, const pData &b) : da(a), db(b) {}
                void calc()
                {
                    d = da->d - db->d;
                }
                void back()
                {
                    da->InitGrad(nGrads);
                    db->InitGrad(nGrads);
                    da->g += g;
                    db->g -= g;
                }
            };
            class OpTimesScalar : public OpBase
            {
                pData da, db;

            public:
                OpTimesScalar(const pData &a, const pData &b) : da(a), db(b) {}
                void calc()
                {
                    assert(db->d.rows() == db->d.cols() && db->d.cols() == 1);
                    d = da->d * db->d(0, 0);
                }
                void back()
                {
                    da->InitGrad(nGrads);
                    db->InitGrad(nGrads);

                    assert(db->d.rows() == db->d.cols() && db->d.cols() == 1);
                    da->g += db->d(0, 0) * g;

                    int acols = da->d.cols();
                    assert(db->g.rows() == 1 && db->g.cols() == nGrads);
                    for (int i = 0; i < nGrads; i++)
                        db->g(0, i) +=
                            (g(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols)).array() *
                             da->d(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols)).array())
                                .sum();
                }
            };
            class OpDivideScalar : public OpBase
            {
                pData da, db;

            public:
                OpDivideScalar(const pData &a, const pData &b) : da(a), db(b) {}
                void calc()
                {
                    assert(db->d.rows() == db->d.cols() && db->d.cols() == 1);
                    d = da->d * db->d(0, 0);
                }
                void back()
                {
                    da->InitGrad(nGrads);
                    db->InitGrad(nGrads);

                    assert(db->d.rows() == db->d.cols() && db->d.cols() == 1);
                    da->g += (1.0 / db->d(0, 0)) * g;

                    int acols = da->d.cols();
                    assert(db->g.rows() == 1 && db->g.cols() == nGrads);
                    for (int i = 0; i < nGrads; i++)
                        db->g(0, i) +=
                            (g(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols)).array() *
                             da->d.array())
                                .sum() *
                            (-1.0 / (db->d(0, 0) * db->d(0, 0)));
                }
            };
            class OpCwiseMul : public OpBase
            {
                pData da, db;

            public:
                OpCwiseMul(const pData &a, const pData &b) : da(a), db(b) {}
                void calc()
                {
                    d = da->d.array() * db->d.array();
                }
                void back()
                {
                    da->InitGrad(nGrads);
                    db->InitGrad(nGrads);

                    int acols = da->d.cols();
                    for (int i = 0; i < nGrads; i++)
                        da->g(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols)).array() +=
                            g(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols)).array() *
                            db->d.array();
                    for (int i = 0; i < nGrads; i++)
                        db->g(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols)).array() +=
                            g(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols)).array() *
                            da->d.array();
                }
            };
            class OpMatMul : public OpBase
            {
                pData da, db;

            public:
                OpMatMul(const pData &a, const pData &b) : da(a), db(b) {}
                void calc()
                {
                    d = da->d * db->d;
                }
                void back()
                {
                    da->InitGrad(nGrads);
                    db->InitGrad(nGrads);

                    int acols = da->d.cols();
                    int bcols = db->d.cols();
                    int cols = d.cols();
                    for (int i = 0; i < nGrads; i++)
                        da->g(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols)) +=
                            g(Eigen::all, Eigen::seq(0 + i * cols, cols - 1 + i * cols)) *
                            db->d.transpose();
                    for (int i = 0; i < nGrads; i++)
                        db->g(Eigen::all, Eigen::seq(0 + i * bcols, bcols - 1 + i * bcols)) +=
                            da->d.transpose() *
                            g(Eigen::all, Eigen::seq(0 + i * cols, cols - 1 + i * cols));
                }
            };
            class OpTimesConstScalar : public OpBase
            {
                pData da;
                real y;

            public:
                OpTimesConstScalar(const pData &a, real ny) : da(a), y(ny) {}
                void calc()
                {
                    d = da->d.array() * y;
                }
                void back()
                {
                    da->InitGrad(nGrads);
                    da->g = g * y;
                }
            };
            class OpMatBlock : public OpBase
            {
            };
            class OpMatConcat : public OpBase
            {
            };
            class OpSqrt : public OpBase
            {
            };

            void operator=(const Eigen::MatrixXd &V) // set as in
            {
            }

            void operator=(const ADEigenMat &R) // set to ref sth
            {
                op = R.op;
            }

            void clone(const ADEigenMat &R) // copy
            {
                // TODO
            }

            ADEigenMat operator+(const ADEigenMat &R)
            {
                // TODO
            }

            ADEigenMat operator-(const ADEigenMat &R)
            {
                // TODO
            }

            ADEigenMat operator*(const ADEigenMat &R)
            {
                // TODO
            }

            ADEigenMat operator*(const real &r)
            {
            }

            ADEigenMat operator/(const ADEigenMat &R)
            {
            }

            template <class TA, class TB>
            ADEigenMat operator()(TA i, TB j)
            {
            }

            friend ADEigenMat concat0(std::vector<ADEigenMat> &mats)
            {
            }
        };
    }
}