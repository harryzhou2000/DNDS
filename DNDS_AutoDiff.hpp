#pragma once

#include "Eigen/Dense"

#include "DNDS_Defines.h"
#include <functional>
#include <iostream>
#include <memory>
#include <initializer_list>
#include <utility>

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
            std::shared_ptr<OpBase> op; // OnlyData

        public:
            ADEigenMat() {}
            ADEigenMat(const std::shared_ptr<OpBase> &nop) : op(nop) {}

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
                void back() override
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
                void back() override
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
                void back() override
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
                void back() override
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
                void back() override
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
                void back() override
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
                void back() override
                {
                    da->InitGrad(nGrads);
                    da->g = g * y;
                }
            };
            class OpMatBlock : public OpBase
            {
                pData d0;
                std::vector<int> iB;
                std::vector<int> jB;

            public:
                template <class Ti, class Tj>
                OpMatBlock(const pData &nd0, Ti &&i, Tj &&j)
                    : d0(nd0), iB(std::forward(i)), jB(std::forward(j)) {}

                void calc()
                {
                    d = d0->d(iB, jB);
                }

                void back() override
                {
                    d0->InitGrad(nGrads);
                    int acols = d0->d.cols();
                    int cols = d.cols();
                    for (int i = 0; i < nGrads; i++)
                        d0->g(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols))(iB, jB) +=
                            g(Eigen::all, Eigen::seq(0 + i * cols, cols - 1 + i * cols));
                }
            };
            class OpMatConcat : public OpBase
            {
                std::vector<pData> datas;

            public:
                OpMatConcat(const std::vector<pData> ndatas) : datas(ndatas) {}

                void calc()
                {
                    int pos = 0;
                    for (auto &i : datas)
                        pos += i->d.rows();
                    d.resize(pos, datas[0]->d.cols());
                    pos = 0;
                    for (auto &i : datas)
                    {
                        d(Eigen::seq(pos, pos + i->d.rows() - 1), Eigen::all) = i->d;
                        pos += i->d.rows();
                    }
                }
                void back() override
                {
                    int pos = 0;
                    for (auto &i : datas)
                    {
                        i->InitGrad(nGrads);
                        i->g += g(Eigen::seq(pos, pos + i->d.rows() - 1), Eigen::all);
                        pos += i->d.rows();
                    }
                }
            };
            class OpSqrt : public OpBase
            {
                pData d0;

            public:
                OpSqrt(const pData &from) : d0(from) {}
                void calc()
                {
                    d = d0->d.array().sqrt();
                }
                void back() override
                {
                    d0->InitGrad(nGrads);
                    d0->g.array() += g.array() / d.array() * 0.5;
                }
            };

            void operator=(const Eigen::MatrixXd &V) // set as in
            {
                auto pNewOp = new OpIn(); //* danger zone
                pNewOp->calc(V);
                op = pData(pNewOp);
            }

            void operator=(const ADEigenMat &R) // set to ref sth
            {
                op = R.op;
            }

            ADEigenMat clone() // copy
            {
                auto pNewOp = new OpCopy(op);
                pNewOp->calc();
                return ADEigenMat(pData(pNewOp));
            }

            ADEigenMat operator+(const ADEigenMat &R)
            {
                auto pNewOp = new OpAdd(op, R.op);
                pNewOp->calc();
                return ADEigenMat(pData(pNewOp));
            }

            ADEigenMat operator-(const ADEigenMat &R)
            {
                auto pNewOp = new OpSubs(op, R.op);
                pNewOp->calc();
                return ADEigenMat(pData(pNewOp));
            }

            ADEigenMat operator*(const ADEigenMat &R)
            {
                if (R.op->d.cols() == 1 && R.op->d.rows() == 1)
                {
                    auto pNewOp = new OpTimesScalar(op, R.op);
                    pNewOp->calc();
                    return ADEigenMat(pData(pNewOp));
                }

                if (R.op->d.cols() == op->d.cols() && R.op->d.rows() == op->d.rows())
                {
                    auto pNewOp = new OpCwiseMul(op, R.op);
                    pNewOp->calc();
                    return ADEigenMat(pData(pNewOp));
                }
            }

            ADEigenMat operator*(const real &r)
            {
                auto pNewOp = new OpTimesConstScalar(op, r);
                pNewOp->calc();
                return ADEigenMat(pData(pNewOp));
            }

            ADEigenMat operator/(const ADEigenMat &R)
            {
                if (R.op->d.cols() == 1 && R.op->d.rows() == 1)
                {
                    auto pNewOp = new OpDivideScalar(op, R.op);
                    pNewOp->calc();
                    return ADEigenMat(pData(pNewOp));
                }
                // TODO: add cwise divide
            }

            template <class TA, class TB>
            ADEigenMat operator()(TA i, TB j)
            {
                auto pNewOp = new OpMatBlock(op, i, j);
                pNewOp->calc();
                return ADEigenMat(pData(pNewOp));
            }

            friend ADEigenMat concat0(const std::vector<ADEigenMat> &mats)
            {
                std::vector<pData> matOps(mats.size());
                for (int i = 0; i < mats.size(); i++)
                    matOps[i] = mats[i].op;
                auto pNewOp = new ADEigenMat::OpMatConcat(matOps);
                pNewOp->calc();
                return ADEigenMat(pData(pNewOp));
            }
        };
    }
}