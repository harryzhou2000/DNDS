#pragma once

#include "Eigen/Dense"

#include "DNDS_Defines.h"
#include <functional>
#include <iostream>
#include <memory>
#include <initializer_list>
#include <utility>
#include <set>
#include <map>
#include <fstream>

// #define DNDS_AUTODIFF_GENERATE_CODE
// #define DNDS_AUTODIFF_DEBUG_PRINTTOPO

#ifdef DNDS_AUTODIFF_GENERATE_CODE
static std::ofstream code_out{"code.txt"};
static std::map<void *, int> objID;
static std::map<void *, std::string> inPutName;
#endif

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
                std::vector<OpBase *> sons;
                int nFather = 0;
                int nDone = 0;

                Eigen::MatrixXd d, g;
                virtual void back() = 0;
                virtual ~OpBase()
                {
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    objID.erase(this);
#endif
                }
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
                        nDone = 0;
                    }
                }

                void InitGradZero(int newNGrads)
                {
                    nGrads = newNGrads;
                    int sz0 = d.rows();
                    int sz1 = d.cols();
                    g.resize(sz0, sz1 * nGrads);
                    g.setZero();
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "Eigen::Matrix<double, "
                             << g.rows() << ", "
                             << g.cols() << " >g_T" << objID[this] << "; //Init Grad Zero" << std::endl;
                    code_out << "g_T" << objID[this] << ".setZero(); //Init Grad Zero" << std::endl;
#endif
                }

                void backMain()
                {
                    InitGrad(d.size());
                    int cols = d.cols();
                    for (int i = 0; i < nGrads; i++)
                        g(Eigen::all, Eigen::seq(0 + i * cols, cols - 1 + i * cols))(i) = 1.0;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "Eigen::Matrix<double, "
                             << g.rows() << ", "
                             << g.cols() << " >g_T" << objID[this] << "; //Init Grad" << std::endl;
                    code_out << "g_T" << objID[this] << ".setZero(); //Init Grad" << std::endl;
                    for (int i = 0; i < nGrads; i++)
                        code_out << "g_T" << objID[this]
                                 << "(Eigen::all, Eigen::seq("
                                 << 0 + i * cols << ", "
                                 << cols - 1 + i * cols << "))("
                                 << i << ") = 1.0; //Init Grad" << std::endl;
#endif

                    std::set<OpBase *> cstack;

#ifdef DNDS_AUTODIFF_DEBUG_PRINTTOPO
                    std::map<OpBase *, int> nodes;                 // *debug
                    std::set<std::pair<OpBase *, OpBase *>> edges; // *debug
                    nodes[this] = 0;
#endif

                    cstack.insert(this);
                    while (cstack.size() > 0)
                    {
                        auto iter = cstack.begin();
                        auto i = *iter;
                        i->nFather = 0;
                        i->nDone = -1;
                        iter = cstack.erase(iter);
                        for (auto j : i->sons)
                        {
                            cstack.insert(j);
#ifdef DNDS_AUTODIFF_DEBUG_PRINTTOPO
                            auto findJ = nodes.find(j);         // *debug
                            if (findJ == nodes.end())           // *debug
                                nodes[j] = nodes.size();        // *debug
                            edges.insert(std::make_pair(i, j)); // *debug
#endif
                        }
                    }
#ifdef DNDS_AUTODIFF_DEBUG_PRINTTOPO
                    Eigen::MatrixXd Adj;                                     // *debug
                    Adj.setZero(nodes.size(), nodes.size());                 // *debug
                    for (auto &e : edges)                                    // *debug
                        Adj(nodes[e.first], nodes[e.second]) = 1;            // *debug
                    std::cout << "Back From [" << this << "] " << std::endl; // *debug
                    std::cout << "Size = " << nodes.size() << std::endl;
                    std::cout << Adj << std::endl;
#endif

                    cstack.insert(this);
                    while (cstack.size() > 0)
                    {
                        auto iter = cstack.begin();

                        auto i = *iter;
                        iter = cstack.erase(iter);
                        if (i->nDone == -1)
                        {
                            i->nDone = 0;
                            if (i != this) // root has been set to eye!
                            {
                                i->InitGradZero(nGrads);
                            }
                            for (auto j : i->sons)
                                j->nFather++;
                            for (auto j : i->sons)
                                cstack.insert(j);
                        }
                    }

                    cstack.insert(this);
                    while (cstack.size() > 0)
                    {
                        int dCount = 0;
                        for (auto iter = cstack.begin(); iter != cstack.end();)
                        {
                            auto i = *iter;
                            if (i->nFather == i->nDone)
                            {
                                iter = cstack.erase(iter), dCount++; //! iterator validity!
                                i->back();                           // i is still ok
                                for (auto j : i->sons)
                                    cstack.insert(j);
                                break;
                            }
                            else
                            {
                                ++iter; //! iterator validity!
                            }
                        }
                        if (dCount == 0)
                        {
                            std::cout << "ADError: has hanging nodes except for launch, check for unused midterms"
                                      << std::endl;
                            assert(false);
                        }
                    }
                }
            };

        private:
            std::shared_ptr<OpBase> op; // OnlyData

        public:
            ADEigenMat() {}
            ADEigenMat(const std::shared_ptr<OpBase> &nop) : op(nop) {}
            ADEigenMat(const Eigen::MatrixXd &nd)
            {
                (*this) = nd;
            }

            Eigen::MatrixXd &g()
            {
                return op->g;
            }

            Eigen::MatrixXd &d()
            {
                return op->d;
            }

            typedef std::shared_ptr<OpBase> pData;

            /**********************************************************************************/
            /*                                                                                */
            /*                      Derived Operator Class                                    */
            /*                                                                                */
            /*                                                                                */
            /**********************************************************************************/
            class OpIn : public OpBase
            {
            public:
                OpIn()
                {
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    objID[this] = objID.size();
#endif
                }
                void calc(const Eigen::MatrixXd &in)
                {
                    d = in;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "Eigen::Matrix<double, "
                             << d.rows() << ", "
                             << d.cols() << "> T"
                             << objID[this] << " = IN{ " << in << "}; //OpIn" << std::endl;
                    // code_out << "Eigen::Matrix<double," << d.rows() << "," << d.rows()
                    //          << "> T" << objID[this] << " = IN{ " << in << "};"<< std::endl;
#endif
                }
                void back() override
                {
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "// grad end is at g_T" << objID[this] << std::endl;
#endif
                }
            };
            class OpCopy : public OpBase
            {
                pData d0;

            public:
                OpCopy(const pData &from) : d0(from)
                {
                    sons.push_back(d0.get());
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    objID[this] = objID.size();
#endif
                }
                void calc()
                {
                    d = d0->d;
                    d0->nFather++;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "Eigen::Matrix<double, "
                             << d.rows() << ", "
                             << d.cols() << "> T"
                             << objID[this] << " = T"
                             << objID[d0.get()] << "; //OpCopy" << std::endl;
                    // code_out << "Eigen::Matrix<double," << d.rows() << "," << d.rows()
                    //          << "> T" << objID[this] << " = IN{ " << in << "};"<< std::endl;
#endif
                }
                void back() override
                {
                    d0->InitGrad(nGrads);
                    d0->g += g;
                    d0->nDone++;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "g_T"
                             << objID[d0.get()] << " += g_T"
                             << objID[this] << "; //OpCopy" << std::endl;
#endif
                }
            };
            class OpAdd : public OpBase
            {
                pData da, db;

            public:
                OpAdd(const pData &a, const pData &b) : da(a), db(b)
                {
                    sons.push_back(da.get());
                    sons.push_back(db.get());
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    objID[this] = objID.size();
#endif
                }
                void calc()
                {
                    d = da->d + db->d;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "Eigen::Matrix<double, "
                             << d.rows() << ", "
                             << d.cols() << "> T"
                             << objID[this] << " = T"
                             << objID[da.get()] << " + T"
                             << objID[db.get()] << "; //OpAdd" << std::endl;
                    // code_out << "Eigen::Matrix<double," << d.rows() << "," << d.rows()
                    //          << "> T" << objID[this] << " = IN{ " << in << "};"<< std::endl;
#endif
                }
                void back() override
                {
                    da->g += g;
                    db->g += g;
                    da->nDone++;
                    db->nDone++;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "g_T"
                             << objID[da.get()] << " += g_T"
                             << objID[this] << "; //OpAdd" << std::endl;
                    code_out << "g_T"
                             << objID[db.get()] << " += g_T"
                             << objID[this] << "; //OpAdd" << std::endl;
#endif
                }
            };
            class OpSubs : public OpBase
            {
                pData da, db;

            public:
                OpSubs(const pData &a, const pData &b) : da(a), db(b)
                {
                    sons.push_back(da.get());
                    sons.push_back(db.get());
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    objID[this] = objID.size();
#endif
                }
                void calc()
                {
                    d = da->d - db->d;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "Eigen::Matrix<double, "
                             << d.rows() << ", "
                             << d.cols() << "> T"
                             << objID[this] << " = T"
                             << objID[da.get()] << " - T"
                             << objID[db.get()] << "; //OpSubs" << std::endl;
                    // code_out << "Eigen::Matrix<double," << d.rows() << "," << d.rows()
                    //          << "> T" << objID[this] << " = IN{ " << in << "};"<< std::endl;
#endif
                }
                void back() override
                {
                    da->g += g;
                    db->g -= g;
                    da->nDone++;
                    db->nDone++;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "g_T"
                             << objID[da.get()] << " += g_T"
                             << objID[this] << "; //OpSubs" << std::endl;
                    code_out << "g_T"
                             << objID[db.get()] << " -= g_T"
                             << objID[this] << "; //OpSubs" << std::endl;
#endif
                }
            };
            class OpTimesScalar : public OpBase
            {
                pData da, db;

            public:
                OpTimesScalar(const pData &a, const pData &b) : da(a), db(b)
                {
                    sons.push_back(da.get());
                    sons.push_back(db.get());
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    objID[this] = objID.size();
#endif
                }
                void calc()
                {
                    assert(db->d.rows() == db->d.cols() && db->d.cols() == 1);
                    d = da->d * db->d(0, 0);
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "Eigen::Matrix<double, "
                             << d.rows() << ", "
                             << d.cols() << "> T"
                             << objID[this] << " = T"
                             << objID[da.get()] << " * T"
                             << objID[db.get()] << "(0,0); //OpTimesScalar" << std::endl;
                    // code_out << "Eigen::Matrix<double," << d.rows() << "," << d.rows()
                    //          << "> T" << objID[this] << " = IN{ " << in << "};"<< std::endl;
#endif
                }
                void back() override
                {

                    assert(db->d.rows() == db->d.cols() && db->d.cols() == 1);
                    da->g += db->d(0, 0) * g;

                    int acols = da->d.cols();
                    assert(db->g.rows() == 1 && db->g.cols() == nGrads);
                    for (int i = 0; i < nGrads; i++)
                        db->g(0, i) +=
                            (g(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols)).array() *
                             da->d.array())
                                .sum();

                    da->nDone++;
                    db->nDone++;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "g_T"
                             << objID[da.get()] << " += g_T"
                             << objID[this] << " * T"
                             << objID[db.get()] << "(0,0); //OpTimesScalar" << std::endl;
                    for (int i = 0; i < nGrads; i++)
                        code_out << "g_T"
                                 << objID[db.get()] << "("
                                 << i << ") += (g_T"
                                 << objID[this] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * acols << ", "
                                 << acols - 1 + i * acols << ")).array() * T"
                                 << objID[da.get()] << ".array()).sum(); //OpTimesScalar" << std::endl;
#endif
                }
            };
            class OpDivideScalar : public OpBase
            {
                pData da, db;

            public:
                OpDivideScalar(const pData &a, const pData &b) : da(a), db(b)
                {
                    sons.push_back(da.get());
                    sons.push_back(db.get());
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    objID[this] = objID.size();
#endif
                }
                void calc()
                {
                    assert(db->d.rows() == db->d.cols() && db->d.cols() == 1);
                    d = da->d / db->d(0, 0);
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "Eigen::Matrix<double, "
                             << d.rows() << ", "
                             << d.cols() << "> T"
                             << objID[this] << " = T"
                             << objID[da.get()] << " / T"
                             << objID[db.get()] << "(0,0); //OpDivideScalar" << std::endl;
                    // code_out << "Eigen::Matrix<double," << d.rows() << "," << d.rows()
                    //          << "> T" << objID[this] << " = IN{ " << in << "};"<< std::endl;
#endif
                }
                void back() override
                {

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
                    da->nDone++;
                    db->nDone++;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "g_T"
                             << objID[da.get()] << " += g_T"
                             << objID[this] << " / T"
                             << objID[db.get()] << "(0,0); //OpDivideScalar" << std::endl;
                    for (int i = 0; i < nGrads; i++)
                        code_out << "g_T"
                                 << objID[db.get()] << "("
                                 << i << ") += (g_T"
                                 << objID[this] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * acols << ", "
                                 << acols - 1 + i * acols << ")).array() * T"
                                 << objID[da.get()] << ".array()).sum() * (-1.0 / (T"
                                 << objID[db.get()] << "(0, 0) * T"
                                 << objID[db.get()] << "(0, 0))); //OpDivideScalar" << std::endl;
#endif
                }
            };
            class OpCwiseMul : public OpBase
            {
                pData da, db;

            public:
                OpCwiseMul(const pData &a, const pData &b) : da(a), db(b)
                {
                    sons.push_back(da.get());
                    sons.push_back(db.get());
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    objID[this] = objID.size();
#endif
                }
                void calc()
                {
                    d = da->d.array() * db->d.array();
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "Eigen::Matrix<double, "
                             << d.rows() << ", "
                             << d.cols() << "> T"
                             << objID[this] << " = (T"
                             << objID[da.get()] << ".array() * T"
                             << objID[db.get()] << ".array()).matrix(); //OpCwiseMul" << std::endl;
#endif
                }
                void back() override
                {

                    int acols = da->d.cols();
                    for (int i = 0; i < nGrads; i++)
                        da->g(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols)).array() +=
                            g(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols)).array() *
                            db->d.array();
                    for (int i = 0; i < nGrads; i++)
                        db->g(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols)).array() +=
                            g(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols)).array() *
                            da->d.array();
                    da->nDone++;
                    db->nDone++;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    for (int i = 0; i < nGrads; i++)
                        code_out << "g_T" << objID[da.get()] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * acols << ", "
                                 << acols - 1 + i * acols << ")).array() += g_T"
                                 << objID[this] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * acols << ", "
                                 << acols - 1 + i * acols << ")).array() * T"
                                 << objID[db.get()] << ".array();  //OpCwiseMul" << std::endl;
                    for (int i = 0; i < nGrads; i++)
                        code_out << "g_T" << objID[db.get()] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * acols << ", "
                                 << acols - 1 + i * acols << ")).array() += g_T"
                                 << objID[this] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * acols << ", "
                                 << acols - 1 + i * acols << ")).array() * T"
                                 << objID[da.get()] << ".array();  //OpCwiseMul" << std::endl;
#endif
                }
            };
            class OpCwiseDiv : public OpBase
            {
                pData da, db;

            public:
                OpCwiseDiv(const pData &a, const pData &b) : da(a), db(b)
                {
                    sons.push_back(da.get());
                    sons.push_back(db.get());
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    objID[this] = objID.size();
#endif
                }
                void calc()
                {
                    d = da->d.array() / db->d.array();
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "Eigen::Matrix<double, "
                             << d.rows() << ", "
                             << d.cols() << "> T"
                             << objID[this] << " = (T"
                             << objID[da.get()] << ".array() / T"
                             << objID[db.get()] << ".array()).matrix(); //OpCwiseDiv" << std::endl;
#endif
                }
                void back() override
                {

                    int acols = da->d.cols();
                    for (int i = 0; i < nGrads; i++)
                        da->g(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols)).array() +=
                            g(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols)).array() /
                            db->d.array();
                    for (int i = 0; i < nGrads; i++)
                        db->g(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols)).array() -=
                            g(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols)).array() *
                            d.array() / db->d.array();
                    da->nDone++;
                    db->nDone++;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    for (int i = 0; i < nGrads; i++)
                        code_out << "g_T" << objID[da.get()] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * acols << ", "
                                 << acols - 1 + i * acols << ")).array() += g_T"
                                 << objID[this] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * acols << ", "
                                 << acols - 1 + i * acols << ")).array() / T"
                                 << objID[db.get()] << ".array();  //OpCwiseDiv" << std::endl;
                    for (int i = 0; i < nGrads; i++)
                        code_out << "g_T" << objID[db.get()] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * acols << ", "
                                 << acols - 1 + i * acols << ")).array() -= g_T"
                                 << objID[this] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * acols << ", "
                                 << acols - 1 + i * acols << ")).array() * T"
                                 << objID[this] << ".array() / T"
                                 << objID[db.get()] << ".array();  //OpCwiseDiv" << std::endl;
#endif
                }
            };
            class OpMatMul : public OpBase
            {
                pData da, db;

            public:
                OpMatMul(const pData &a, const pData &b) : da(a), db(b)
                {
                    sons.push_back(da.get());
                    sons.push_back(db.get());
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    objID[this] = objID.size();
#endif
                }
                void calc()
                {
                    d = da->d * db->d;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "Eigen::Matrix<double, "
                             << d.rows() << ", "
                             << d.cols() << "> T"
                             << objID[this] << " = T"
                             << objID[da.get()] << " * T"
                             << objID[db.get()] << "; //OpMatMul" << std::endl;
#endif
                }
                void back() override
                {
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
                    da->nDone++;
                    db->nDone++;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    for (int i = 0; i < nGrads; i++)
                        code_out << "g_T"
                                 << objID[da.get()] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * acols << ", "
                                 << acols - 1 + i * acols << ")) += g_T"
                                 << objID[this] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * cols << ", "
                                 << cols - 1 + i * cols << ")) * T"
                                 << objID[db.get()] << ".transpose(); //OpMatMul" << std::endl;
                    for (int i = 0; i < nGrads; i++)
                        code_out << "g_T"
                                 << objID[db.get()] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * bcols << ", "
                                 << bcols - 1 + i * bcols << ")) += T"
                                 << objID[da.get()] << ".transpose() * g_T"
                                 << objID[this] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * cols << ", "
                                 << cols - 1 + i * cols << ")); //OpMatMul" << std::endl;
#endif
                }
            };
            class OpMatTrans : public OpBase
            {
                pData da;

            public:
                OpMatTrans(const pData &a) : da(a)
                {
                    sons.push_back(da.get());
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    objID[this] = objID.size();
#endif
                }
                void calc()
                {
                    d = da->d.transpose();
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "Eigen::Matrix<double, "
                             << d.rows() << ", "
                             << d.cols() << "> T"
                             << objID[this] << " = T"
                             << objID[da.get()] << ".transpose(); //OpMatTrans" << std::endl;
#endif
                }
                void back() override
                {
                    int acols = da->d.cols();
                    int cols = d.cols();
                    for (int i = 0; i < nGrads; i++)
                        da->g(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols)) +=
                            g(Eigen::all, Eigen::seq(0 + i * cols, cols - 1 + i * cols)).transpose();
                    da->nDone++;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    for (int i = 0; i < nGrads; i++)
                        code_out << "g_T"
                                 << objID[da.get()] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * acols << ", "
                                 << acols - 1 + i * acols << ")) += g_T"
                                 << objID[this] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * cols << ", "
                                 << cols - 1 + i * cols << ")).transpose(); //OpMatTrans" << std::endl;
#endif
                }
            };
            class OpMatDot : public OpBase
            {
                pData da, db;

            public:
                OpMatDot(const pData &a, const pData &b) : da(a), db(b)
                {
                    sons.push_back(da.get());
                    sons.push_back(db.get());
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    objID[this] = objID.size();
#endif
                }
                void calc()
                {
                    d.resize(1, 1);
                    d(0, 0) = (da->d.array() * db->d.array()).sum();
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "Eigen::Matrix<double, "
                             << d.rows() << ", "
                             << d.cols() << "> T"
                             << objID[this] << " = Eigen::Matrix<double,1,1>{{(T"
                             << objID[da.get()] << ".array() * T"
                             << objID[db.get()] << ".array()).sum()}}; //OpMatDot" << std::endl;
#endif
                }
                void back() override
                {
                    int acols = da->d.cols();
                    int bcols = db->d.cols();
                    int cols = d.cols();
                    for (int i = 0; i < nGrads; i++)
                        da->g(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols)) +=
                            g(i) * db->d;
                    for (int i = 0; i < nGrads; i++)
                        db->g(Eigen::all, Eigen::seq(0 + i * bcols, bcols - 1 + i * bcols)) +=
                            g(i) * da->d;
                    da->nDone++;
                    db->nDone++;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    for (int i = 0; i < nGrads; i++)
                        code_out << "g_T"
                                 << objID[da.get()] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * acols << ", "
                                 << acols - 1 + i * acols << ")) += g_T"
                                 << objID[this] << "("
                                 << i << ") * T"
                                 << objID[db.get()] << "; //OpMatDot" << std::endl;
                    for (int i = 0; i < nGrads; i++)
                        code_out << "g_T"
                                 << objID[db.get()] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * bcols << ", "
                                 << bcols - 1 + i * bcols << ")) += g_T"
                                 << objID[this] << "("
                                 << i << ") * T"
                                 << objID[da.get()] << "; // OpMatDot" << std::endl;

#endif
                }
            };
            class OpTimesConstScalar : public OpBase
            {
                pData da;
                real y;

            public:
                OpTimesConstScalar(const pData &a, real ny) : da(a), y(ny)
                {
                    sons.push_back(da.get());
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    objID[this] = objID.size();
#endif
                }
                void calc()
                {
                    d = da->d.array() * y;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "Eigen::Matrix<double, "
                             << d.rows() << ", "
                             << d.cols() << "> T"
                             << objID[this] << " = T"
                             << objID[da.get()] << " * " << y << "; //OpTimesConstScalar" << std::endl;
#endif
                }
                void back() override
                {
                    da->g += g * y;
                    da->nDone++;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "g_T"
                             << objID[da.get()] << " += g_T"
                             << objID[this] << " * "
                             << y << "; //OpTimesConstScalar" << std::endl;
#endif
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
                    : d0(nd0), iB(std::forward<Ti>(i)), jB(std::forward<Tj>(j))
                {
                    sons.push_back(d0.get());
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    objID[this] = objID.size();
#endif
                }

                void calc()
                {
                    d = d0->d(iB, jB);
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "Eigen::Matrix<double, "
                             << d.rows() << ", "
                             << d.cols() << "> T"
                             << objID[this] << " = T"
                             << objID[d0.get()] << " ({";
                    for (int i = 0; i < iB.size(); i++)
                        code_out << iB[i] << ((i == (iB.size() - 1)) ? "" : ",");
                    code_out << "},{";
                    for (int i = 0; i < jB.size(); i++)
                        code_out << jB[i] << ((i == (jB.size() - 1)) ? "" : ",");
                    code_out << "}); //OpMatBlock" << std::endl;
#endif
                }

                void back() override
                {
                    int acols = d0->d.cols();
                    int cols = d.cols();
                    for (int i = 0; i < nGrads; i++)
                        d0->g(Eigen::all, Eigen::seq(0 + i * acols, acols - 1 + i * acols))(iB, jB) +=
                            g(Eigen::all, Eigen::seq(0 + i * cols, cols - 1 + i * cols));
                    d0->nDone++;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    for (int i = 0; i < nGrads; i++)
                    {
                        code_out << "g_T"
                                 << objID[d0.get()] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * acols << ", "
                                 << acols - 1 + i * acols << "))({";
                        for (int i = 0; i < iB.size(); i++)
                            code_out << iB[i] << ((i == (iB.size() - 1)) ? "" : ",");
                        code_out << "}, {";
                        for (int i = 0; i < jB.size(); i++)
                            code_out << jB[i] << ((i == (jB.size() - 1)) ? "" : ",");
                        code_out << "}) += g_T"
                                 << objID[this] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * cols << ", "
                                 << cols - 1 + i * cols << "));  //OpMatBlock" << std::endl;
                    }

#endif
                }
            };
            class OpMatConcat : public OpBase
            {
                std::vector<pData> datas;

            public:
                OpMatConcat(const std::vector<pData> ndatas) : datas(ndatas)
                {
                    for (auto &i : datas)
                    {
                        sons.push_back(i.get());
                    }
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    objID[this] = objID.size();
#endif
                }

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
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "Eigen::Matrix<double," << pos << "," << datas[0]->d.cols() << "> T"
                             << objID[this] << "; //OpMatConcat" << std::endl;
                    // code_out << "T"<< objID[this] <<".setZero(); //OpMatConcat" << std::endl;
                    pos = 0;
                    for (auto &i : datas)
                    {
                        code_out << "T" << objID[this] << "(Eigen::seq(" << pos << ", "
                                 << pos + i->d.rows() - 1 << "), Eigen::all) = T"
                                 << objID[i.get()] << "; //OpMatConcat" << std::endl;
                        pos += i->d.rows();
                    }
#endif
                }
                void back() override
                {
                    int pos = 0;
                    for (auto &i : datas)
                    {
                        i->g += g(Eigen::seq(pos, pos + i->d.rows() - 1), Eigen::all);
                        pos += i->d.rows();
                    }

                    for (auto &i : datas)
                    {
                        i->nDone++;
                    }
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    pos = 0;
                    for (auto &i : datas)
                    {
                        code_out << "g_T"
                                 << objID[i.get()] << " += g_T"
                                 << objID[this] << "(Eigen::seq("
                                 << pos << ", "
                                 << pos + i->d.rows() - 1 << "), Eigen::all); //OpMatConcat" << std::endl;
                        pos += i->d.rows();
                    }
#endif
                }
            };
            class OpSqrt : public OpBase
            {
                pData d0;

            public:
                OpSqrt(const pData &from) : d0(from)
                {
                    sons.push_back(d0.get());
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    objID[this] = objID.size();
#endif
                }
                void calc()
                {
                    d = d0->d.array().sqrt();
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "Eigen::Matrix<double, "
                             << d.rows() << ", "
                             << d.cols() << "> T"
                             << objID[this] << " = T"
                             << objID[d0.get()] << ".array().sqrt().matrix(); //OpSqrt" << std::endl;
#endif
                }
                void back() override
                {
                    int cols = d0->d.cols();
                    Eigen::MatrixXd T1 = 1 / d.array();
                    for (int j = 0; j < d.size(); j++)
                        if (d(j) == 0)
                            T1(j) = 0;
                    for (int i = 0; i < nGrads; i++)
                        d0->g(Eigen::all, Eigen::seq(0 + i * cols, cols - 1 + i * cols)).array() +=
                            g(Eigen::all, Eigen::seq(0 + i * cols, cols - 1 + i * cols)).array() * T1.array() * 0.5;

                    d0->nDone++;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "Eigen::Matrix<double,"
                             << d.rows() << ","
                             << d.cols() << "> T1_T" << objID[this] << "= 1 / T"
                             << objID[this] << ".array();// OpSqrt" << std::endl;
                    for (int j = 0; j < d.size(); j++)
                        code_out << "if(T"
                                 << objID[this] << "("
                                 << j << ")==0) T1_T"
                                 << objID[this] << "("
                                 << j << ")=0;// OpSqrt" << std::endl;
                    for (int i = 0; i < nGrads; i++)
                    {

                        code_out << "g_T"
                                 << objID[d0.get()] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * cols << ", "
                                 << cols - 1 + i * cols << ")).array() += g_T"
                                 << objID[this] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * cols << ", "
                                 << cols - 1 + i * cols << ")).array() * T1_T"
                                 << objID[this] << ".array() * 0.5; // OpSqrt" << std::endl;
                    }
#endif
                }
            };
            class OpHYFixExp : public OpBase
            {
                pData d0, a;

            public:
                OpHYFixExp(const pData &from, const pData &na) : d0(from), a(na)
                {
                    sons.push_back(d0.get());
                    sons.push_back(a.get());
                    assert(a->d.cols() == a->d.rows() && a->d.rows() == 1);
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    objID[this] = objID.size();
#endif
                }
                void calc()
                {
                    d = d0->d.array() +
                        d0->d.array().sign() *
                            (d0->d.array().abs() / (-a->d(0, 0))).exp() * a->d(0, 0);
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "Eigen::Matrix<double, "
                             << d.rows() << ", "
                             << d.cols() << "> T"
                             << objID[this] << " = (T"
                             << objID[d0.get()] << ".array() + (T"
                             << objID[d0.get()] << ".array().abs() / (-T"
                             << objID[a.get()] << "(0,0))).exp() * T"
                             << objID[a.get()] << "(0,0) ).matrix(); //OpHYFixExp" << std::endl;
#endif
                }
                void back() override
                {
                    int cols = d0->d.cols();
                    for (int i = 0; i < nGrads; i++)
                        d0->g(Eigen::all, Eigen::seq(0 + i * cols, cols - 1 + i * cols)).array() +=
                            g(Eigen::all, Eigen::seq(0 + i * cols, cols - 1 + i * cols)).array() *
                            (1 + (d0->d.array() - d.array()) * d0->d.array().sign() / a->d(0, 0));
                    for (int i = 0; i < nGrads; i++)
                        a->g(i) +=
                            (g(Eigen::all, Eigen::seq(0 + i * cols, cols - 1 + i * cols)).array() *
                             (d.array() - d0->d.array()) *
                             (d0->d.array().abs() + a->d(0, 0)) /
                             (a->d(0, 0) * a->d(0, 0)))
                                .sum();

                    d0->nDone++;
                    a->nDone++;

#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    for (int i = 0; i < nGrads; i++)
                        code_out << "g_T"
                                 << objID[d0.get()] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * cols << ", "
                                 << cols - 1 + i * cols << ")).array() += g_T"
                                 << objID[this] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * cols << ", "
                                 << cols - 1 + i * cols << ")).array() * (1 + (T"
                                 << objID[d0.get()] << ".array() - T"
                                 << objID[this] << ".array()) * T"
                                 << objID[d0.get()] << ".array().sign() / T"
                                 << objID[a.get()] << "(0, 0)); //OpHYFixExp" << std::endl;
                    for (int i = 0; i < nGrads; i++)
                        code_out << "g_T"
                                 << objID[a.get()] << "("
                                 << i << ") += (g_T"
                                 << objID[this] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * cols << ", "
                                 << cols - 1 + i * cols << ")).array() * (T"
                                 << objID[this] << ".array() - T"
                                 << objID[d0.get()] << ".array()) * (T"
                                 << objID[d0.get()] << ".array().abs() + T"
                                 << objID[a.get()] << "(0, 0)) / (T"
                                 << objID[a.get()] << "(0, 0) * T"
                                 << objID[a.get()] << "(0, 0))).sum(); // OpHYFixExp" << std::endl;
#endif
                }
            };
            class OpAbs : public OpBase
            {
                pData d0;

            public:
                OpAbs(const pData &from) : d0(from)
                {
                    sons.push_back(d0.get());
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    objID[this] = objID.size();
#endif
                }
                void calc()
                {
                    d = d0->d.array().abs();
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    code_out << "Eigen::Matrix<double, "
                             << d.rows() << ", "
                             << d.cols() << "> T"
                             << objID[this] << " = T"
                             << objID[d0.get()] << ".array().abs().matrix(); //OpAbs" << std::endl;
#endif
                }
                void back() override
                {
                    int cols = d0->d.cols();
                    for (int i = 0; i < nGrads; i++)
                        d0->g(Eigen::all, Eigen::seq(0 + i * cols, cols - 1 + i * cols)).array() +=
                            g(Eigen::all, Eigen::seq(0 + i * cols, cols - 1 + i * cols)).array() * d0->d.array().sign();
                    d0->nDone++;
#ifdef DNDS_AUTODIFF_GENERATE_CODE
                    for (int i = 0; i < nGrads; i++)
                        code_out << "g_T"
                                 << objID[d0.get()] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * cols << ", "
                                 << cols - 1 + i * cols << ")).array() += g_T"
                                 << objID[this] << "(Eigen::all, Eigen::seq("
                                 << 0 + i * cols << ", "
                                 << cols - 1 + i * cols << ")).array() * T"
                                 << objID[d0.get()] << ".array().sign(); //OpAbs" << std::endl;
#endif
                }
            };

            /**********************************************************************************/
            /*                                                                                */
            /*                                                                                */
            /*                                                                                */
            /*                                                                                */
            /**********************************************************************************/

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
                if (R.op->d.cols() == op->d.cols() && R.op->d.rows() == op->d.rows())
                {
                    auto pNewOp = new OpCwiseMul(op, R.op);
                    pNewOp->calc();
                    return ADEigenMat(pData(pNewOp));
                }
                if (R.op->d.cols() == 1 && R.op->d.rows() == 1)
                {
                    auto pNewOp = new OpTimesScalar(op, R.op);
                    pNewOp->calc();
                    return ADEigenMat(pData(pNewOp));
                }

                assert(false);
                return (ADEigenMat());
            }

            ADEigenMat operator*(const real &r)
            {
                auto pNewOp = new OpTimesConstScalar(op, r);
                pNewOp->calc();
                return ADEigenMat(pData(pNewOp));
            }

            ADEigenMat operator/(const ADEigenMat &R)
            {
                if (R.op->d.cols() == op->d.cols() && R.op->d.rows() == op->d.rows())
                {
                    auto pNewOp = new OpCwiseDiv(op, R.op);
                    pNewOp->calc();
                    return ADEigenMat(pData(pNewOp));
                }
                if (R.op->d.cols() == 1 && R.op->d.rows() == 1)
                {
                    auto pNewOp = new OpDivideScalar(op, R.op);
                    pNewOp->calc();
                    return ADEigenMat(pData(pNewOp));
                }
                assert(false);
                return (ADEigenMat());
            }

            template <class TA, class TB>
            ADEigenMat operator()(TA i, TB j)
            {
                auto pNewOp = new OpMatBlock(op, i, j);
                pNewOp->calc();
                return ADEigenMat(pData(pNewOp));
            }

            ADEigenMat operator()(std::initializer_list<int> i, std::initializer_list<int> j)
            {
                auto pNewOp = new OpMatBlock(op, i, j);
                pNewOp->calc();
                return ADEigenMat(pData(pNewOp));
            }

            ADEigenMat sqrt()
            {
                auto pNewOp = new OpSqrt(op);
                pNewOp->calc();
                return ADEigenMat(pData(pNewOp));
            }

            ADEigenMat abs() //! TEST
            {
                auto pNewOp = new OpAbs(op);
                pNewOp->calc();
                return ADEigenMat(pData(pNewOp));
            }

            ADEigenMat hyFixExp(const ADEigenMat &a)
            {
                auto pNewOp = new OpHYFixExp(op, a.op);
                pNewOp->calc();
                return ADEigenMat(pData(pNewOp));
            }

            ADEigenMat matmul(const ADEigenMat &R)
            {
                auto pNewOp = new OpMatMul(op, R.op);
                pNewOp->calc();
                return ADEigenMat(pData(pNewOp));
            }

            ADEigenMat dot(const ADEigenMat &R)
            {
                auto pNewOp = new OpMatDot(op, R.op);
                pNewOp->calc();
                return ADEigenMat(pData(pNewOp));
            }

            ADEigenMat transpose()
            {
                auto pNewOp = new OpMatTrans(op);
                pNewOp->calc();
                return ADEigenMat(pData(pNewOp));
            }

            static ADEigenMat concat0(const std::vector<ADEigenMat> &mats)
            {
                std::vector<pData> matOps(mats.size());
                for (int i = 0; i < mats.size(); i++)
                    matOps[i] = mats[i].op;
                auto pNewOp = new ADEigenMat::OpMatConcat(matOps);
                pNewOp->calc();
                return ADEigenMat(pData(pNewOp));
            }

            void back() // TODO: add if clear gradient?
            {
                // assert(op->nFather == op->nDone && op->nFather == 0);
                op->backMain();
            }

            friend std::ostream &operator<<(std::ostream &o, const ADEigenMat &m)
            {
                o << "D = \n"
                  << m.op->d << "\nG = \n"
                  << m.op->g << "\n nFather/nDone = " << m.op->nFather << "/" << m.op->nDone << "\n";
                return o;
            }
        };
    }
}