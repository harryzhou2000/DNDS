#include <iostream>
#include <time.h>

#define DNDS_AUTODIFF_DEBUG_PRINTTOPO
#define DNDS_AUTODIFF_GENERATE_CODE
#include "../../DNDS_AutoDiff.hpp"

static const int N = 10000;

int main()
{
    using namespace DNDS;

    real eps = 1e-4;
    auto U = Eigen::Vector<real, 5>{1.0000, 0.5916, 0, 0, 2.6750};
    auto U1 = Eigen::Vector<real, 5>{1.0000, 0.5916, 0, 0, 2.6750};
    auto GU = Eigen::Matrix<real, 3, 5>{{1, 2, 0.2, 0, 0.5},
                                        {0, 1, 0.5, 0, 1.0},
                                        {2, 0, 0.1, 0, 1.5}};
    auto GU1 = Eigen::Matrix<real, 3, 5>{{1, 2 + eps, 0.2, 0, 0.5},
                                         {0, 1, 0.5, 0, 1.0},
                                         {2, 0, 0.1, 0, 1.5}};
    auto n = Eigen::Vector<real, 3>{0.6, 0.8, 0};
    auto gammaD = Eigen::MatrixXd{{1.4}};
    auto scaleHY = Eigen::MatrixXd{{0.01}};

    auto mu = Eigen::MatrixXd{{1e-3}};
    auto k = Eigen::MatrixXd{{1e-3}};
    auto cp = Eigen::MatrixXd{{1e3}};

    Eigen::Matrix<real, 4, 1> F;
    Eigen::Matrix<real, 5, 4> G_U;
    Eigen::Matrix<real, 3, 20> G_GU;

    real gamma = gammaD(0);

    auto tic = clock();
    for (int iter = 0; iter < 10000; iter++)
    {
        Eigen::Matrix<double, 3, 3> T0 = Eigen::Matrix3d::Identity();                    // OpIn
        Eigen::Matrix<double, 1, 1> T1 = Eigen::Matrix<real, 1, 1>{gamma - 1};           // OpIn
        Eigen::Matrix<double, 1, 1> T2 = Eigen::Matrix<real, 1, 1>{gamma / (gamma - 1)}; // OpIn
        Eigen::Matrix<double, 3, 1> T3 = n;                                              // OpIn
        Eigen::Matrix<double, 1, 1> T4 = mu;                                             // OpIn
        Eigen::Matrix<double, 1, 1> T5 = k;                                              // OpIn
        Eigen::Matrix<double, 1, 1> T6 = cp;                                             // OpIn
        Eigen::Matrix<double, 5, 1> T7 = U;                                              // OpIn
        Eigen::Matrix<double, 3, 5> T8 = GU;                                             // OpIn

        Eigen::Matrix<double, 3, 1> T9 = T7({1, 2, 3}, {0});                                                // OpMatBlock
        Eigen::Matrix<double, 1, 1> T10 = T7({0}, {0});                                                     // OpMatBlock
        Eigen::Matrix<double, 1, 1> T11 = T7({4}, {0});                                                     // OpMatBlock
        Eigen::Matrix<double, 3, 1> T12 = T9 / T10(0, 0);                                                   // OpDivideScalar
        Eigen::Matrix<double, 1, 1> T13 = Eigen::Matrix<double, 1, 1>{{(T12.array() * T12.array()).sum()}}; // OpMatDot
        Eigen::Matrix<double, 3, 1> T14 = T8({0, 1, 2}, {0});                                               // OpMatBlock
        Eigen::Matrix<double, 3, 3> T15 = T8({0, 1, 2}, {1, 2, 3});                                         // OpMatBlock
        Eigen::Matrix<double, 3, 1> T16 = T8({0, 1, 2}, {4});                                               // OpMatBlock
        Eigen::Matrix<double, 1, 3> T17 = T12.transpose();                                                  // OpMatTrans
        Eigen::Matrix<double, 3, 3> T18 = T14 * T17;                                                        // OpMatMul
        Eigen::Matrix<double, 3, 3> T19 = T15 - T18;                                                        // OpSubs
        Eigen::Matrix<double, 3, 3> T20 = T19 / T10(0, 0);                                                  // OpDivideScalar
        Eigen::Matrix<double, 3, 1> T21 = T15 * T12;                                                        // OpMatMul
        Eigen::Matrix<double, 3, 1> T22 = T20 * T9;                                                         // OpMatMul
        Eigen::Matrix<double, 3, 1> T23 = T21 + T22;                                                        // OpAdd
        Eigen::Matrix<double, 3, 1> T24 = T23 * 0.5;                                                        // OpTimesConstScalar
        Eigen::Matrix<double, 3, 1> T25 = T16 - T24;                                                        // OpSubs
        Eigen::Matrix<double, 3, 1> T26 = T25 * T1(0, 0);                                                   // OpTimesScalar
        Eigen::Matrix<double, 1, 1> T27 = (T10.array() * T13.array()).matrix();                             // OpCwiseMul
        Eigen::Matrix<double, 1, 1> T28 = T27 * 0.5;                                                        // OpTimesConstScalar
        Eigen::Matrix<double, 1, 1> T29 = T11 - T28;                                                        // OpSubs
        Eigen::Matrix<double, 1, 1> T30 = (T29.array() * T1.array()).matrix();                              // OpCwiseMul
        Eigen::Matrix<double, 1, 1> T31 = (T10.array() * T10.array()).matrix();                             // OpCwiseMul
        Eigen::Matrix<double, 3, 1> T32 = T26 * T10(0, 0);                                                  // OpTimesScalar
        Eigen::Matrix<double, 3, 1> T33 = T14 * T30(0, 0);                                                  // OpTimesScalar
        Eigen::Matrix<double, 3, 1> T34 = T32 - T33;                                                        // OpSubs
        Eigen::Matrix<double, 1, 1> T35 = (T6.array() * T31.array()).matrix();                              // OpCwiseMul
        Eigen::Matrix<double, 1, 1> T36 = (T2.array() / T35.array()).matrix();                              // OpCwiseDiv
        Eigen::Matrix<double, 3, 1> T37 = T34 * T36(0, 0);                                                  // OpTimesScalar
        Eigen::Matrix<double, 3, 3> T38 = T20.transpose();                                                  // OpMatTrans
        Eigen::Matrix<double, 3, 3> T39 = T20 + T38;                                                        // OpAdd
        Eigen::Matrix<double, 1, 1> T40 = T20({0}, {0});                                                    // OpMatBlock
        Eigen::Matrix<double, 1, 1> T41 = T20({1}, {1});                                                    // OpMatBlock
        Eigen::Matrix<double, 1, 1> T42 = T40 + T41;                                                        // OpAdd
        Eigen::Matrix<double, 1, 1> T43 = T20({2}, {2});                                                    // OpMatBlock
        Eigen::Matrix<double, 1, 1> T44 = T42 + T43;                                                        // OpAdd
        Eigen::Matrix<double, 1, 1> T45 = T44 * -2. / 3.;                                                   // OpTimesConstScalar
        Eigen::Matrix<double, 3, 3> T46 = T0 * T45(0, 0);                                                   // OpTimesScalar
        Eigen::Matrix<double, 3, 3> T47 = T39 + T46;                                                        // OpAdd
        Eigen::Matrix<double, 3, 3> T48 = T47 * T4(0, 0);                                                   // OpTimesScalar
        Eigen::Matrix<double, 3, 1> T49 = T48 * T12;                                                        // OpMatMul
        Eigen::Matrix<double, 3, 1> T50 = T37 * T5(0, 0);                                                   // OpTimesScalar
        Eigen::Matrix<double, 3, 1> T51 = T49 + T50;                                                        // OpAdd
        Eigen::Matrix<double, 3, 1> T52 = T48 * T3;                                                         // OpMatMul
        Eigen::Matrix<double, 1, 1> T53 = Eigen::Matrix<double, 1, 1>{{(T51.array() * T3.array()).sum()}};  // OpMatDot
        Eigen::Matrix<double, 4, 1> T54;                                                                    // OpMatConcat
        T54(Eigen::seq(0, 2), Eigen::all) = T52;                                                            // OpMatConcat
        T54(Eigen::seq(3, 3), Eigen::all) = T53;                                                            // OpMatConcat
        Eigen::Matrix<double, 4, 4> g_T54;                                                                  // Init Grad
        g_T54.setZero();                                                                                    // Init Grad
        g_T54(Eigen::all, Eigen::seq(0, 0))(0) = 1.0;                                                       // Init Grad
        g_T54(Eigen::all, Eigen::seq(1, 1))(1) = 1.0;                                                       // Init Grad
        g_T54(Eigen::all, Eigen::seq(2, 2))(2) = 1.0;                                                       // Init Grad
        g_T54(Eigen::all, Eigen::seq(3, 3))(3) = 1.0;                                                       // Init Grad
        Eigen::Matrix<double, 3, 4> g_T52;                                                                  // Init Grad Zero
        g_T52.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 4> g_T3;                                                                   // Init Grad Zero
        g_T3.setZero();                                                                                     // Init Grad Zero
        Eigen::Matrix<double, 3, 12> g_T48;                                                                 // Init Grad Zero
        g_T48.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T4;                                                                   // Init Grad Zero
        g_T4.setZero();                                                                                     // Init Grad Zero
        Eigen::Matrix<double, 3, 12> g_T47;                                                                 // Init Grad Zero
        g_T47.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 12> g_T39;                                                                 // Init Grad Zero
        g_T39.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 12> g_T20;                                                                 // Init Grad Zero
        g_T20.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T10;                                                                  // Init Grad Zero
        g_T10.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 5, 4> g_T7;                                                                   // Init Grad Zero
        g_T7.setZero();                                                                                     // Init Grad Zero
        Eigen::Matrix<double, 3, 12> g_T19;                                                                 // Init Grad Zero
        g_T19.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 12> g_T15;                                                                 // Init Grad Zero
        g_T15.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 20> g_T8;                                                                  // Init Grad Zero
        g_T8.setZero();                                                                                     // Init Grad Zero
        Eigen::Matrix<double, 3, 12> g_T18;                                                                 // Init Grad Zero
        g_T18.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 1, 12> g_T17;                                                                 // Init Grad Zero
        g_T17.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 4> g_T12;                                                                  // Init Grad Zero
        g_T12.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 4> g_T9;                                                                   // Init Grad Zero
        g_T9.setZero();                                                                                     // Init Grad Zero
        Eigen::Matrix<double, 3, 4> g_T14;                                                                  // Init Grad Zero
        g_T14.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 12> g_T38;                                                                 // Init Grad Zero
        g_T38.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 12> g_T46;                                                                 // Init Grad Zero
        g_T46.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 12> g_T0;                                                                  // Init Grad Zero
        g_T0.setZero();                                                                                     // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T45;                                                                  // Init Grad Zero
        g_T45.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T44;                                                                  // Init Grad Zero
        g_T44.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T42;                                                                  // Init Grad Zero
        g_T42.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T40;                                                                  // Init Grad Zero
        g_T40.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T41;                                                                  // Init Grad Zero
        g_T41.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T43;                                                                  // Init Grad Zero
        g_T43.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T53;                                                                  // Init Grad Zero
        g_T53.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 4> g_T51;                                                                  // Init Grad Zero
        g_T51.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 4> g_T49;                                                                  // Init Grad Zero
        g_T49.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 4> g_T50;                                                                  // Init Grad Zero
        g_T50.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T5;                                                                   // Init Grad Zero
        g_T5.setZero();                                                                                     // Init Grad Zero
        Eigen::Matrix<double, 3, 4> g_T37;                                                                  // Init Grad Zero
        g_T37.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 4> g_T34;                                                                  // Init Grad Zero
        g_T34.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 4> g_T32;                                                                  // Init Grad Zero
        g_T32.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 4> g_T26;                                                                  // Init Grad Zero
        g_T26.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T1;                                                                   // Init Grad Zero
        g_T1.setZero();                                                                                     // Init Grad Zero
        Eigen::Matrix<double, 3, 4> g_T25;                                                                  // Init Grad Zero
        g_T25.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 4> g_T16;                                                                  // Init Grad Zero
        g_T16.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 4> g_T24;                                                                  // Init Grad Zero
        g_T24.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 4> g_T23;                                                                  // Init Grad Zero
        g_T23.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 4> g_T21;                                                                  // Init Grad Zero
        g_T21.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 4> g_T22;                                                                  // Init Grad Zero
        g_T22.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 3, 4> g_T33;                                                                  // Init Grad Zero
        g_T33.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T30;                                                                  // Init Grad Zero
        g_T30.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T29;                                                                  // Init Grad Zero
        g_T29.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T11;                                                                  // Init Grad Zero
        g_T11.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T28;                                                                  // Init Grad Zero
        g_T28.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T27;                                                                  // Init Grad Zero
        g_T27.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T13;                                                                  // Init Grad Zero
        g_T13.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T36;                                                                  // Init Grad Zero
        g_T36.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T2;                                                                   // Init Grad Zero
        g_T2.setZero();                                                                                     // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T35;                                                                  // Init Grad Zero
        g_T35.setZero();                                                                                    // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T6;                                                                   // Init Grad Zero
        g_T6.setZero();                                                                                     // Init Grad Zero
        Eigen::Matrix<double, 1, 4> g_T31;                                                                  // Init Grad Zero
        g_T31.setZero();                                                                                    // Init Grad Zero
        g_T52 += g_T54(Eigen::seq(0, 2), Eigen::all);                                                       // OpMatConcat
        g_T53 += g_T54(Eigen::seq(3, 3), Eigen::all);                                                       // OpMatConcat
        g_T48(Eigen::all, Eigen::seq(0, 2)) += g_T52(Eigen::all, Eigen::seq(0, 0)) * T3.transpose();        // OpMatMul
        g_T48(Eigen::all, Eigen::seq(3, 5)) += g_T52(Eigen::all, Eigen::seq(1, 1)) * T3.transpose();        // OpMatMul
        g_T48(Eigen::all, Eigen::seq(6, 8)) += g_T52(Eigen::all, Eigen::seq(2, 2)) * T3.transpose();        // OpMatMul
        g_T48(Eigen::all, Eigen::seq(9, 11)) += g_T52(Eigen::all, Eigen::seq(3, 3)) * T3.transpose();       // OpMatMul
        g_T3(Eigen::all, Eigen::seq(0, 0)) += T48.transpose() * g_T52(Eigen::all, Eigen::seq(0, 0));        // OpMatMul
        g_T3(Eigen::all, Eigen::seq(1, 1)) += T48.transpose() * g_T52(Eigen::all, Eigen::seq(1, 1));        // OpMatMul
        g_T3(Eigen::all, Eigen::seq(2, 2)) += T48.transpose() * g_T52(Eigen::all, Eigen::seq(2, 2));        // OpMatMul
        g_T3(Eigen::all, Eigen::seq(3, 3)) += T48.transpose() * g_T52(Eigen::all, Eigen::seq(3, 3));        // OpMatMul
        g_T51(Eigen::all, Eigen::seq(0, 0)) += g_T53(0) * T3;                                               // OpMatDot
        g_T51(Eigen::all, Eigen::seq(1, 1)) += g_T53(1) * T3;                                               // OpMatDot
        g_T51(Eigen::all, Eigen::seq(2, 2)) += g_T53(2) * T3;                                               // OpMatDot
        g_T51(Eigen::all, Eigen::seq(3, 3)) += g_T53(3) * T3;                                               // OpMatDot
        g_T3(Eigen::all, Eigen::seq(0, 0)) += g_T53(0) * T51;                                               // OpMatDot
        g_T3(Eigen::all, Eigen::seq(1, 1)) += g_T53(1) * T51;                                               // OpMatDot
        g_T3(Eigen::all, Eigen::seq(2, 2)) += g_T53(2) * T51;                                               // OpMatDot
        g_T3(Eigen::all, Eigen::seq(3, 3)) += g_T53(3) * T51;                                               // OpMatDot
        // grad end is at g_T3
        g_T49 += g_T51;                                                                                // OpAdd
        g_T50 += g_T51;                                                                                // OpAdd
        g_T48(Eigen::all, Eigen::seq(0, 2)) += g_T49(Eigen::all, Eigen::seq(0, 0)) * T12.transpose();  // OpMatMul
        g_T48(Eigen::all, Eigen::seq(3, 5)) += g_T49(Eigen::all, Eigen::seq(1, 1)) * T12.transpose();  // OpMatMul
        g_T48(Eigen::all, Eigen::seq(6, 8)) += g_T49(Eigen::all, Eigen::seq(2, 2)) * T12.transpose();  // OpMatMul
        g_T48(Eigen::all, Eigen::seq(9, 11)) += g_T49(Eigen::all, Eigen::seq(3, 3)) * T12.transpose(); // OpMatMul
        g_T12(Eigen::all, Eigen::seq(0, 0)) += T48.transpose() * g_T49(Eigen::all, Eigen::seq(0, 0));  // OpMatMul
        g_T12(Eigen::all, Eigen::seq(1, 1)) += T48.transpose() * g_T49(Eigen::all, Eigen::seq(1, 1));  // OpMatMul
        g_T12(Eigen::all, Eigen::seq(2, 2)) += T48.transpose() * g_T49(Eigen::all, Eigen::seq(2, 2));  // OpMatMul
        g_T12(Eigen::all, Eigen::seq(3, 3)) += T48.transpose() * g_T49(Eigen::all, Eigen::seq(3, 3));  // OpMatMul
        g_T47 += g_T48 * T4(0, 0);                                                                     // OpTimesScalar
        g_T4(0) += (g_T48(Eigen::all, Eigen::seq(0, 2)).array() * T47.array()).sum();                  // OpTimesScalar
        g_T4(1) += (g_T48(Eigen::all, Eigen::seq(3, 5)).array() * T47.array()).sum();                  // OpTimesScalar
        g_T4(2) += (g_T48(Eigen::all, Eigen::seq(6, 8)).array() * T47.array()).sum();                  // OpTimesScalar
        g_T4(3) += (g_T48(Eigen::all, Eigen::seq(9, 11)).array() * T47.array()).sum();                 // OpTimesScalar
        // grad end is at g_T4
        g_T39 += g_T47;                                                                           // OpAdd
        g_T46 += g_T47;                                                                           // OpAdd
        g_T20 += g_T39;                                                                           // OpAdd
        g_T38 += g_T39;                                                                           // OpAdd
        g_T20(Eigen::all, Eigen::seq(0, 2)) += g_T38(Eigen::all, Eigen::seq(0, 2)).transpose();   // OpMatTrans
        g_T20(Eigen::all, Eigen::seq(3, 5)) += g_T38(Eigen::all, Eigen::seq(3, 5)).transpose();   // OpMatTrans
        g_T20(Eigen::all, Eigen::seq(6, 8)) += g_T38(Eigen::all, Eigen::seq(6, 8)).transpose();   // OpMatTrans
        g_T20(Eigen::all, Eigen::seq(9, 11)) += g_T38(Eigen::all, Eigen::seq(9, 11)).transpose(); // OpMatTrans
        g_T0 += g_T46 * T45(0, 0);                                                                // OpTimesScalar
        g_T45(0) += (g_T46(Eigen::all, Eigen::seq(0, 2)).array() * T0.array()).sum();             // OpTimesScalar
        g_T45(1) += (g_T46(Eigen::all, Eigen::seq(3, 5)).array() * T0.array()).sum();             // OpTimesScalar
        g_T45(2) += (g_T46(Eigen::all, Eigen::seq(6, 8)).array() * T0.array()).sum();             // OpTimesScalar
        g_T45(3) += (g_T46(Eigen::all, Eigen::seq(9, 11)).array() * T0.array()).sum();            // OpTimesScalar
        // grad end is at g_T0
        g_T44 += g_T45 * -2. / 3.;                                                             // OpTimesConstScalar
        g_T42 += g_T44;                                                                        // OpAdd
        g_T43 += g_T44;                                                                        // OpAdd
        g_T40 += g_T42;                                                                        // OpAdd
        g_T41 += g_T42;                                                                        // OpAdd
        g_T20(Eigen::all, Eigen::seq(0, 2))({0}, {0}) += g_T40(Eigen::all, Eigen::seq(0, 0));  // OpMatBlock
        g_T20(Eigen::all, Eigen::seq(3, 5))({0}, {0}) += g_T40(Eigen::all, Eigen::seq(1, 1));  // OpMatBlock
        g_T20(Eigen::all, Eigen::seq(6, 8))({0}, {0}) += g_T40(Eigen::all, Eigen::seq(2, 2));  // OpMatBlock
        g_T20(Eigen::all, Eigen::seq(9, 11))({0}, {0}) += g_T40(Eigen::all, Eigen::seq(3, 3)); // OpMatBlock
        g_T20(Eigen::all, Eigen::seq(0, 2))({1}, {1}) += g_T41(Eigen::all, Eigen::seq(0, 0));  // OpMatBlock
        g_T20(Eigen::all, Eigen::seq(3, 5))({1}, {1}) += g_T41(Eigen::all, Eigen::seq(1, 1));  // OpMatBlock
        g_T20(Eigen::all, Eigen::seq(6, 8))({1}, {1}) += g_T41(Eigen::all, Eigen::seq(2, 2));  // OpMatBlock
        g_T20(Eigen::all, Eigen::seq(9, 11))({1}, {1}) += g_T41(Eigen::all, Eigen::seq(3, 3)); // OpMatBlock
        g_T20(Eigen::all, Eigen::seq(0, 2))({2}, {2}) += g_T43(Eigen::all, Eigen::seq(0, 0));  // OpMatBlock
        g_T20(Eigen::all, Eigen::seq(3, 5))({2}, {2}) += g_T43(Eigen::all, Eigen::seq(1, 1));  // OpMatBlock
        g_T20(Eigen::all, Eigen::seq(6, 8))({2}, {2}) += g_T43(Eigen::all, Eigen::seq(2, 2));  // OpMatBlock
        g_T20(Eigen::all, Eigen::seq(9, 11))({2}, {2}) += g_T43(Eigen::all, Eigen::seq(3, 3)); // OpMatBlock
        g_T37 += g_T50 * T5(0, 0);                                                             // OpTimesScalar
        g_T5(0) += (g_T50(Eigen::all, Eigen::seq(0, 0)).array() * T37.array()).sum();          // OpTimesScalar
        g_T5(1) += (g_T50(Eigen::all, Eigen::seq(1, 1)).array() * T37.array()).sum();          // OpTimesScalar
        g_T5(2) += (g_T50(Eigen::all, Eigen::seq(2, 2)).array() * T37.array()).sum();          // OpTimesScalar
        g_T5(3) += (g_T50(Eigen::all, Eigen::seq(3, 3)).array() * T37.array()).sum();          // OpTimesScalar
        // grad end is at g_T5
        g_T34 += g_T37 * T36(0, 0);                                                                                        // OpTimesScalar
        g_T36(0) += (g_T37(Eigen::all, Eigen::seq(0, 0)).array() * T34.array()).sum();                                     // OpTimesScalar
        g_T36(1) += (g_T37(Eigen::all, Eigen::seq(1, 1)).array() * T34.array()).sum();                                     // OpTimesScalar
        g_T36(2) += (g_T37(Eigen::all, Eigen::seq(2, 2)).array() * T34.array()).sum();                                     // OpTimesScalar
        g_T36(3) += (g_T37(Eigen::all, Eigen::seq(3, 3)).array() * T34.array()).sum();                                     // OpTimesScalar
        g_T32 += g_T34;                                                                                                    // OpSubs
        g_T33 -= g_T34;                                                                                                    // OpSubs
        g_T26 += g_T32 * T10(0, 0);                                                                                        // OpTimesScalar
        g_T10(0) += (g_T32(Eigen::all, Eigen::seq(0, 0)).array() * T26.array()).sum();                                     // OpTimesScalar
        g_T10(1) += (g_T32(Eigen::all, Eigen::seq(1, 1)).array() * T26.array()).sum();                                     // OpTimesScalar
        g_T10(2) += (g_T32(Eigen::all, Eigen::seq(2, 2)).array() * T26.array()).sum();                                     // OpTimesScalar
        g_T10(3) += (g_T32(Eigen::all, Eigen::seq(3, 3)).array() * T26.array()).sum();                                     // OpTimesScalar
        g_T25 += g_T26 * T1(0, 0);                                                                                         // OpTimesScalar
        g_T1(0) += (g_T26(Eigen::all, Eigen::seq(0, 0)).array() * T25.array()).sum();                                      // OpTimesScalar
        g_T1(1) += (g_T26(Eigen::all, Eigen::seq(1, 1)).array() * T25.array()).sum();                                      // OpTimesScalar
        g_T1(2) += (g_T26(Eigen::all, Eigen::seq(2, 2)).array() * T25.array()).sum();                                      // OpTimesScalar
        g_T1(3) += (g_T26(Eigen::all, Eigen::seq(3, 3)).array() * T25.array()).sum();                                      // OpTimesScalar
        g_T16 += g_T25;                                                                                                    // OpSubs
        g_T24 -= g_T25;                                                                                                    // OpSubs
        g_T8(Eigen::all, Eigen::seq(0, 4))({0, 1, 2}, {4}) += g_T16(Eigen::all, Eigen::seq(0, 0));                         // OpMatBlock
        g_T8(Eigen::all, Eigen::seq(5, 9))({0, 1, 2}, {4}) += g_T16(Eigen::all, Eigen::seq(1, 1));                         // OpMatBlock
        g_T8(Eigen::all, Eigen::seq(10, 14))({0, 1, 2}, {4}) += g_T16(Eigen::all, Eigen::seq(2, 2));                       // OpMatBlock
        g_T8(Eigen::all, Eigen::seq(15, 19))({0, 1, 2}, {4}) += g_T16(Eigen::all, Eigen::seq(3, 3));                       // OpMatBlock
        g_T23 += g_T24 * 0.5;                                                                                              // OpTimesConstScalar
        g_T21 += g_T23;                                                                                                    // OpAdd
        g_T22 += g_T23;                                                                                                    // OpAdd
        g_T15(Eigen::all, Eigen::seq(0, 2)) += g_T21(Eigen::all, Eigen::seq(0, 0)) * T12.transpose();                      // OpMatMul
        g_T15(Eigen::all, Eigen::seq(3, 5)) += g_T21(Eigen::all, Eigen::seq(1, 1)) * T12.transpose();                      // OpMatMul
        g_T15(Eigen::all, Eigen::seq(6, 8)) += g_T21(Eigen::all, Eigen::seq(2, 2)) * T12.transpose();                      // OpMatMul
        g_T15(Eigen::all, Eigen::seq(9, 11)) += g_T21(Eigen::all, Eigen::seq(3, 3)) * T12.transpose();                     // OpMatMul
        g_T12(Eigen::all, Eigen::seq(0, 0)) += T15.transpose() * g_T21(Eigen::all, Eigen::seq(0, 0));                      // OpMatMul
        g_T12(Eigen::all, Eigen::seq(1, 1)) += T15.transpose() * g_T21(Eigen::all, Eigen::seq(1, 1));                      // OpMatMul
        g_T12(Eigen::all, Eigen::seq(2, 2)) += T15.transpose() * g_T21(Eigen::all, Eigen::seq(2, 2));                      // OpMatMul
        g_T12(Eigen::all, Eigen::seq(3, 3)) += T15.transpose() * g_T21(Eigen::all, Eigen::seq(3, 3));                      // OpMatMul
        g_T20(Eigen::all, Eigen::seq(0, 2)) += g_T22(Eigen::all, Eigen::seq(0, 0)) * T9.transpose();                       // OpMatMul
        g_T20(Eigen::all, Eigen::seq(3, 5)) += g_T22(Eigen::all, Eigen::seq(1, 1)) * T9.transpose();                       // OpMatMul
        g_T20(Eigen::all, Eigen::seq(6, 8)) += g_T22(Eigen::all, Eigen::seq(2, 2)) * T9.transpose();                       // OpMatMul
        g_T20(Eigen::all, Eigen::seq(9, 11)) += g_T22(Eigen::all, Eigen::seq(3, 3)) * T9.transpose();                      // OpMatMul
        g_T9(Eigen::all, Eigen::seq(0, 0)) += T20.transpose() * g_T22(Eigen::all, Eigen::seq(0, 0));                       // OpMatMul
        g_T9(Eigen::all, Eigen::seq(1, 1)) += T20.transpose() * g_T22(Eigen::all, Eigen::seq(1, 1));                       // OpMatMul
        g_T9(Eigen::all, Eigen::seq(2, 2)) += T20.transpose() * g_T22(Eigen::all, Eigen::seq(2, 2));                       // OpMatMul
        g_T9(Eigen::all, Eigen::seq(3, 3)) += T20.transpose() * g_T22(Eigen::all, Eigen::seq(3, 3));                       // OpMatMul
        g_T19 += g_T20 / T10(0, 0);                                                                                        // OpDivideScalar
        g_T10(0) += (g_T20(Eigen::all, Eigen::seq(0, 2)).array() * T19.array()).sum() * (-1.0 / (T10(0, 0) * T10(0, 0)));  // OpDivideScalar
        g_T10(1) += (g_T20(Eigen::all, Eigen::seq(3, 5)).array() * T19.array()).sum() * (-1.0 / (T10(0, 0) * T10(0, 0)));  // OpDivideScalar
        g_T10(2) += (g_T20(Eigen::all, Eigen::seq(6, 8)).array() * T19.array()).sum() * (-1.0 / (T10(0, 0) * T10(0, 0)));  // OpDivideScalar
        g_T10(3) += (g_T20(Eigen::all, Eigen::seq(9, 11)).array() * T19.array()).sum() * (-1.0 / (T10(0, 0) * T10(0, 0))); // OpDivideScalar
        g_T15 += g_T19;                                                                                                    // OpSubs
        g_T18 -= g_T19;                                                                                                    // OpSubs
        g_T8(Eigen::all, Eigen::seq(0, 4))({0, 1, 2}, {1, 2, 3}) += g_T15(Eigen::all, Eigen::seq(0, 2));                   // OpMatBlock
        g_T8(Eigen::all, Eigen::seq(5, 9))({0, 1, 2}, {1, 2, 3}) += g_T15(Eigen::all, Eigen::seq(3, 5));                   // OpMatBlock
        g_T8(Eigen::all, Eigen::seq(10, 14))({0, 1, 2}, {1, 2, 3}) += g_T15(Eigen::all, Eigen::seq(6, 8));                 // OpMatBlock
        g_T8(Eigen::all, Eigen::seq(15, 19))({0, 1, 2}, {1, 2, 3}) += g_T15(Eigen::all, Eigen::seq(9, 11));                // OpMatBlock
        g_T14(Eigen::all, Eigen::seq(0, 0)) += g_T18(Eigen::all, Eigen::seq(0, 2)) * T17.transpose();                      // OpMatMul
        g_T14(Eigen::all, Eigen::seq(1, 1)) += g_T18(Eigen::all, Eigen::seq(3, 5)) * T17.transpose();                      // OpMatMul
        g_T14(Eigen::all, Eigen::seq(2, 2)) += g_T18(Eigen::all, Eigen::seq(6, 8)) * T17.transpose();                      // OpMatMul
        g_T14(Eigen::all, Eigen::seq(3, 3)) += g_T18(Eigen::all, Eigen::seq(9, 11)) * T17.transpose();                     // OpMatMul
        g_T17(Eigen::all, Eigen::seq(0, 2)) += T14.transpose() * g_T18(Eigen::all, Eigen::seq(0, 2));                      // OpMatMul
        g_T17(Eigen::all, Eigen::seq(3, 5)) += T14.transpose() * g_T18(Eigen::all, Eigen::seq(3, 5));                      // OpMatMul
        g_T17(Eigen::all, Eigen::seq(6, 8)) += T14.transpose() * g_T18(Eigen::all, Eigen::seq(6, 8));                      // OpMatMul
        g_T17(Eigen::all, Eigen::seq(9, 11)) += T14.transpose() * g_T18(Eigen::all, Eigen::seq(9, 11));                    // OpMatMul
        g_T12(Eigen::all, Eigen::seq(0, 0)) += g_T17(Eigen::all, Eigen::seq(0, 2)).transpose();                            // OpMatTrans
        g_T12(Eigen::all, Eigen::seq(1, 1)) += g_T17(Eigen::all, Eigen::seq(3, 5)).transpose();                            // OpMatTrans
        g_T12(Eigen::all, Eigen::seq(2, 2)) += g_T17(Eigen::all, Eigen::seq(6, 8)).transpose();                            // OpMatTrans
        g_T12(Eigen::all, Eigen::seq(3, 3)) += g_T17(Eigen::all, Eigen::seq(9, 11)).transpose();                           // OpMatTrans
        g_T14 += g_T33 * T30(0, 0);                                                                                        // OpTimesScalar
        g_T30(0) += (g_T33(Eigen::all, Eigen::seq(0, 0)).array() * T14.array()).sum();                                     // OpTimesScalar
        g_T30(1) += (g_T33(Eigen::all, Eigen::seq(1, 1)).array() * T14.array()).sum();                                     // OpTimesScalar
        g_T30(2) += (g_T33(Eigen::all, Eigen::seq(2, 2)).array() * T14.array()).sum();                                     // OpTimesScalar
        g_T30(3) += (g_T33(Eigen::all, Eigen::seq(3, 3)).array() * T14.array()).sum();                                     // OpTimesScalar
        g_T8(Eigen::all, Eigen::seq(0, 4))({0, 1, 2}, {0}) += g_T14(Eigen::all, Eigen::seq(0, 0));                         // OpMatBlock
        g_T8(Eigen::all, Eigen::seq(5, 9))({0, 1, 2}, {0}) += g_T14(Eigen::all, Eigen::seq(1, 1));                         // OpMatBlock
        g_T8(Eigen::all, Eigen::seq(10, 14))({0, 1, 2}, {0}) += g_T14(Eigen::all, Eigen::seq(2, 2));                       // OpMatBlock
        g_T8(Eigen::all, Eigen::seq(15, 19))({0, 1, 2}, {0}) += g_T14(Eigen::all, Eigen::seq(3, 3));                       // OpMatBlock
        // grad end is at g_T8
        g_T29(Eigen::all, Eigen::seq(0, 0)).array() += g_T30(Eigen::all, Eigen::seq(0, 0)).array() * T1.array(); // OpCwiseMul
        g_T29(Eigen::all, Eigen::seq(1, 1)).array() += g_T30(Eigen::all, Eigen::seq(1, 1)).array() * T1.array(); // OpCwiseMul
        g_T29(Eigen::all, Eigen::seq(2, 2)).array() += g_T30(Eigen::all, Eigen::seq(2, 2)).array() * T1.array(); // OpCwiseMul
        g_T29(Eigen::all, Eigen::seq(3, 3)).array() += g_T30(Eigen::all, Eigen::seq(3, 3)).array() * T1.array(); // OpCwiseMul
        g_T1(Eigen::all, Eigen::seq(0, 0)).array() += g_T30(Eigen::all, Eigen::seq(0, 0)).array() * T29.array(); // OpCwiseMul
        g_T1(Eigen::all, Eigen::seq(1, 1)).array() += g_T30(Eigen::all, Eigen::seq(1, 1)).array() * T29.array(); // OpCwiseMul
        g_T1(Eigen::all, Eigen::seq(2, 2)).array() += g_T30(Eigen::all, Eigen::seq(2, 2)).array() * T29.array(); // OpCwiseMul
        g_T1(Eigen::all, Eigen::seq(3, 3)).array() += g_T30(Eigen::all, Eigen::seq(3, 3)).array() * T29.array(); // OpCwiseMul
        // grad end is at g_T1
        g_T11 += g_T29;                                                                                                         // OpSubs
        g_T28 -= g_T29;                                                                                                         // OpSubs
        g_T7(Eigen::all, Eigen::seq(0, 0))({4}, {0}) += g_T11(Eigen::all, Eigen::seq(0, 0));                                    // OpMatBlock
        g_T7(Eigen::all, Eigen::seq(1, 1))({4}, {0}) += g_T11(Eigen::all, Eigen::seq(1, 1));                                    // OpMatBlock
        g_T7(Eigen::all, Eigen::seq(2, 2))({4}, {0}) += g_T11(Eigen::all, Eigen::seq(2, 2));                                    // OpMatBlock
        g_T7(Eigen::all, Eigen::seq(3, 3))({4}, {0}) += g_T11(Eigen::all, Eigen::seq(3, 3));                                    // OpMatBlock
        g_T27 += g_T28 * 0.5;                                                                                                   // OpTimesConstScalar
        g_T10(Eigen::all, Eigen::seq(0, 0)).array() += g_T27(Eigen::all, Eigen::seq(0, 0)).array() * T13.array();               // OpCwiseMul
        g_T10(Eigen::all, Eigen::seq(1, 1)).array() += g_T27(Eigen::all, Eigen::seq(1, 1)).array() * T13.array();               // OpCwiseMul
        g_T10(Eigen::all, Eigen::seq(2, 2)).array() += g_T27(Eigen::all, Eigen::seq(2, 2)).array() * T13.array();               // OpCwiseMul
        g_T10(Eigen::all, Eigen::seq(3, 3)).array() += g_T27(Eigen::all, Eigen::seq(3, 3)).array() * T13.array();               // OpCwiseMul
        g_T13(Eigen::all, Eigen::seq(0, 0)).array() += g_T27(Eigen::all, Eigen::seq(0, 0)).array() * T10.array();               // OpCwiseMul
        g_T13(Eigen::all, Eigen::seq(1, 1)).array() += g_T27(Eigen::all, Eigen::seq(1, 1)).array() * T10.array();               // OpCwiseMul
        g_T13(Eigen::all, Eigen::seq(2, 2)).array() += g_T27(Eigen::all, Eigen::seq(2, 2)).array() * T10.array();               // OpCwiseMul
        g_T13(Eigen::all, Eigen::seq(3, 3)).array() += g_T27(Eigen::all, Eigen::seq(3, 3)).array() * T10.array();               // OpCwiseMul
        g_T12(Eigen::all, Eigen::seq(0, 0)) += g_T13(0) * T12;                                                                  // OpMatDot
        g_T12(Eigen::all, Eigen::seq(1, 1)) += g_T13(1) * T12;                                                                  // OpMatDot
        g_T12(Eigen::all, Eigen::seq(2, 2)) += g_T13(2) * T12;                                                                  // OpMatDot
        g_T12(Eigen::all, Eigen::seq(3, 3)) += g_T13(3) * T12;                                                                  // OpMatDot
        g_T12(Eigen::all, Eigen::seq(0, 0)) += g_T13(0) * T12;                                                                  // OpMatDot
        g_T12(Eigen::all, Eigen::seq(1, 1)) += g_T13(1) * T12;                                                                  // OpMatDot
        g_T12(Eigen::all, Eigen::seq(2, 2)) += g_T13(2) * T12;                                                                  // OpMatDot
        g_T12(Eigen::all, Eigen::seq(3, 3)) += g_T13(3) * T12;                                                                  // OpMatDot
        g_T9 += g_T12 / T10(0, 0);                                                                                              // OpDivideScalar
        g_T10(0) += (g_T12(Eigen::all, Eigen::seq(0, 0)).array() * T9.array()).sum() * (-1.0 / (T10(0, 0) * T10(0, 0)));        // OpDivideScalar
        g_T10(1) += (g_T12(Eigen::all, Eigen::seq(1, 1)).array() * T9.array()).sum() * (-1.0 / (T10(0, 0) * T10(0, 0)));        // OpDivideScalar
        g_T10(2) += (g_T12(Eigen::all, Eigen::seq(2, 2)).array() * T9.array()).sum() * (-1.0 / (T10(0, 0) * T10(0, 0)));        // OpDivideScalar
        g_T10(3) += (g_T12(Eigen::all, Eigen::seq(3, 3)).array() * T9.array()).sum() * (-1.0 / (T10(0, 0) * T10(0, 0)));        // OpDivideScalar
        g_T7(Eigen::all, Eigen::seq(0, 0))({1, 2, 3}, {0}) += g_T9(Eigen::all, Eigen::seq(0, 0));                               // OpMatBlock
        g_T7(Eigen::all, Eigen::seq(1, 1))({1, 2, 3}, {0}) += g_T9(Eigen::all, Eigen::seq(1, 1));                               // OpMatBlock
        g_T7(Eigen::all, Eigen::seq(2, 2))({1, 2, 3}, {0}) += g_T9(Eigen::all, Eigen::seq(2, 2));                               // OpMatBlock
        g_T7(Eigen::all, Eigen::seq(3, 3))({1, 2, 3}, {0}) += g_T9(Eigen::all, Eigen::seq(3, 3));                               // OpMatBlock
        g_T2(Eigen::all, Eigen::seq(0, 0)).array() += g_T36(Eigen::all, Eigen::seq(0, 0)).array() / T35.array();                // OpCwiseDiv
        g_T2(Eigen::all, Eigen::seq(1, 1)).array() += g_T36(Eigen::all, Eigen::seq(1, 1)).array() / T35.array();                // OpCwiseDiv
        g_T2(Eigen::all, Eigen::seq(2, 2)).array() += g_T36(Eigen::all, Eigen::seq(2, 2)).array() / T35.array();                // OpCwiseDiv
        g_T2(Eigen::all, Eigen::seq(3, 3)).array() += g_T36(Eigen::all, Eigen::seq(3, 3)).array() / T35.array();                // OpCwiseDiv
        g_T35(Eigen::all, Eigen::seq(0, 0)).array() -= g_T36(Eigen::all, Eigen::seq(0, 0)).array() * T36.array() / T35.array(); // OpCwiseDiv
        g_T35(Eigen::all, Eigen::seq(1, 1)).array() -= g_T36(Eigen::all, Eigen::seq(1, 1)).array() * T36.array() / T35.array(); // OpCwiseDiv
        g_T35(Eigen::all, Eigen::seq(2, 2)).array() -= g_T36(Eigen::all, Eigen::seq(2, 2)).array() * T36.array() / T35.array(); // OpCwiseDiv
        g_T35(Eigen::all, Eigen::seq(3, 3)).array() -= g_T36(Eigen::all, Eigen::seq(3, 3)).array() * T36.array() / T35.array(); // OpCwiseDiv
        // grad end is at g_T2
        g_T6(Eigen::all, Eigen::seq(0, 0)).array() += g_T35(Eigen::all, Eigen::seq(0, 0)).array() * T31.array(); // OpCwiseMul
        g_T6(Eigen::all, Eigen::seq(1, 1)).array() += g_T35(Eigen::all, Eigen::seq(1, 1)).array() * T31.array(); // OpCwiseMul
        g_T6(Eigen::all, Eigen::seq(2, 2)).array() += g_T35(Eigen::all, Eigen::seq(2, 2)).array() * T31.array(); // OpCwiseMul
        g_T6(Eigen::all, Eigen::seq(3, 3)).array() += g_T35(Eigen::all, Eigen::seq(3, 3)).array() * T31.array(); // OpCwiseMul
        g_T31(Eigen::all, Eigen::seq(0, 0)).array() += g_T35(Eigen::all, Eigen::seq(0, 0)).array() * T6.array(); // OpCwiseMul
        g_T31(Eigen::all, Eigen::seq(1, 1)).array() += g_T35(Eigen::all, Eigen::seq(1, 1)).array() * T6.array(); // OpCwiseMul
        g_T31(Eigen::all, Eigen::seq(2, 2)).array() += g_T35(Eigen::all, Eigen::seq(2, 2)).array() * T6.array(); // OpCwiseMul
        g_T31(Eigen::all, Eigen::seq(3, 3)).array() += g_T35(Eigen::all, Eigen::seq(3, 3)).array() * T6.array(); // OpCwiseMul
        // grad end is at g_T6
        g_T10(Eigen::all, Eigen::seq(0, 0)).array() += g_T31(Eigen::all, Eigen::seq(0, 0)).array() * T10.array(); // OpCwiseMul
        g_T10(Eigen::all, Eigen::seq(1, 1)).array() += g_T31(Eigen::all, Eigen::seq(1, 1)).array() * T10.array(); // OpCwiseMul
        g_T10(Eigen::all, Eigen::seq(2, 2)).array() += g_T31(Eigen::all, Eigen::seq(2, 2)).array() * T10.array(); // OpCwiseMul
        g_T10(Eigen::all, Eigen::seq(3, 3)).array() += g_T31(Eigen::all, Eigen::seq(3, 3)).array() * T10.array(); // OpCwiseMul
        g_T10(Eigen::all, Eigen::seq(0, 0)).array() += g_T31(Eigen::all, Eigen::seq(0, 0)).array() * T10.array(); // OpCwiseMul
        g_T10(Eigen::all, Eigen::seq(1, 1)).array() += g_T31(Eigen::all, Eigen::seq(1, 1)).array() * T10.array(); // OpCwiseMul
        g_T10(Eigen::all, Eigen::seq(2, 2)).array() += g_T31(Eigen::all, Eigen::seq(2, 2)).array() * T10.array(); // OpCwiseMul
        g_T10(Eigen::all, Eigen::seq(3, 3)).array() += g_T31(Eigen::all, Eigen::seq(3, 3)).array() * T10.array(); // OpCwiseMul
        g_T7(Eigen::all, Eigen::seq(0, 0))({0}, {0}) += g_T10(Eigen::all, Eigen::seq(0, 0));                      // OpMatBlock
        g_T7(Eigen::all, Eigen::seq(1, 1))({0}, {0}) += g_T10(Eigen::all, Eigen::seq(1, 1));                      // OpMatBlock
        g_T7(Eigen::all, Eigen::seq(2, 2))({0}, {0}) += g_T10(Eigen::all, Eigen::seq(2, 2));                      // OpMatBlock
        g_T7(Eigen::all, Eigen::seq(3, 3))({0}, {0}) += g_T10(Eigen::all, Eigen::seq(3, 3));                      // OpMatBlock
        // grad end is at g_T7

        //
        //
        //
        //

        F = T54;
        //

        G_U = g_T7;
        // std::cout << g_T4 << std::endl
        //           << std::endl;
        G_GU = g_T8;
        // std::cout << g_T5 << std::endl
        //           << std::endl;
    }
    std::cout << F.transpose() << std::endl
              << std::endl;

    std::cout << G_U << std::endl
              << std::endl;
    std::cout << G_GU << std::endl
              << std::endl;

    std::cout << "Time = " << double(clock() - tic) / CLOCKS_PER_SEC << std::endl;

    return 0;
}