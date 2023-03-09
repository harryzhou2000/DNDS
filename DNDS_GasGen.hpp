#pragma once

#include "DNDS_Gas.hpp"

namespace DNDS
{
    namespace Gas
    {
        template <typename TUL, typename TUR, typename TF, typename TdFdU, typename TFdumpInfo>
        void RoeFlux_IdealGas_HartenYee_AutoDiffGen(const TUL &UL, const TUR &UR, real gamma, TF &F,
                                                    TdFdU &dFdUL, TdFdU &dFdUR,
                                                    const TFdumpInfo &dumpInfo)
        {
            static const real HYThres = 0.05;

            const auto T0 = Eigen::Matrix<real, 1, 1>({{1}});         // OpIn
            const auto T1 = Eigen::Matrix<real, 1, 1>({{0}});         // OpIn
            const auto T2 = Eigen::Matrix<real, 1, 1>({{gamma - 1}}); // OpIn
            const auto T3 = Eigen::Matrix<real, 1, 1>({{HYThres}});   // OpIn
            auto T4 = UL;                                             // OpIn
            auto T5 = UR;                                             // OpIn

            // std::cout << "FUCK0" << std::endl;
            Eigen::Matrix<double, 1, 1> T6 = T4({0}, {0});                                                                                                                   // OpMatBlock
            Eigen::Matrix<double, 1, 1> T7 = T5({0}, {0});                                                                                                                   // OpMatBlock
            Eigen::Matrix<double, 3, 1> T8 = T4({1, 2, 3}, {0});                                                                                                             // OpMatBlock
            Eigen::Matrix<double, 3, 1> T9 = T8 / T6(0, 0);                                                                                                                  // OpDivideScalar
            Eigen::Matrix<double, 3, 1> T10 = T5({1, 2, 3}, {0});                                                                                                            // OpMatBlock
            Eigen::Matrix<double, 3, 1> T11 = T10 / T7(0, 0);                                                                                                                // OpDivideScalar
            Eigen::Matrix<double, 1, 1> T12 = Eigen::Matrix<double, 1, 1>{{(T9.array() * T9.array()).sum()}};                                                                // OpMatDot
            Eigen::Matrix<double, 1, 1> T13 = Eigen::Matrix<double, 1, 1>{{(T11.array() * T11.array()).sum()}};                                                              // OpMatDot
            Eigen::Matrix<double, 1, 1> T14 = T4({4}, {0});                                                                                                                  // OpMatBlock
            Eigen::Matrix<double, 1, 1> T15 = T5({4}, {0});                                                                                                                  // OpMatBlock
            Eigen::Matrix<double, 1, 1> T16 = (T6.array() * T12.array()).matrix();                                                                                           // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T17 = T16 * 0.5;                                                                                                                     // OpTimesConstScalar
            Eigen::Matrix<double, 1, 1> T18 = T14 - T17;                                                                                                                     // OpSubs
            Eigen::Matrix<double, 1, 1> T19 = (T18.array() * T2.array()).matrix();                                                                                           // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T20 = (T7.array() * T13.array()).matrix();                                                                                           // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T21 = T20 * 0.5;                                                                                                                     // OpTimesConstScalar
            Eigen::Matrix<double, 1, 1> T22 = T15 - T21;                                                                                                                     // OpSubs
            Eigen::Matrix<double, 1, 1> T23 = (T22.array() * T2.array()).matrix();                                                                                           // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T24 = T14 + T19;                                                                                                                     // OpAdd
            Eigen::Matrix<double, 1, 1> T25 = (T24.array() / T6.array()).matrix();                                                                                           // OpCwiseDiv
            Eigen::Matrix<double, 1, 1> T26 = T15 + T23;                                                                                                                     // OpAdd
            Eigen::Matrix<double, 1, 1> T27 = (T26.array() / T7.array()).matrix();                                                                                           // OpCwiseDiv
            Eigen::Matrix<double, 1, 1> T28 = T6.array().sqrt().matrix();                                                                                                    // OpSqrt
            Eigen::Matrix<double, 1, 1> T29 = T7.array().sqrt().matrix();                                                                                                    // OpSqrt
            Eigen::Matrix<double, 1, 1> T30 = T28 + T29;                                                                                                                     // OpAdd
            Eigen::Matrix<double, 3, 1> T31 = T9 * T28(0, 0);                                                                                                                // OpTimesScalar
            Eigen::Matrix<double, 3, 1> T32 = T11 * T29(0, 0);                                                                                                               // OpTimesScalar
            Eigen::Matrix<double, 3, 1> T33 = T31 + T32;                                                                                                                     // OpAdd
            Eigen::Matrix<double, 3, 1> T34 = T33 / T30(0, 0);                                                                                                               // OpDivideScalar
            Eigen::Matrix<double, 1, 1> T35 = Eigen::Matrix<double, 1, 1>{{(T34.array() * T34.array()).sum()}};                                                              // OpMatDot
            Eigen::Matrix<double, 1, 1> T36 = (T25.array() * T28.array()).matrix();                                                                                          // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T37 = (T27.array() * T29.array()).matrix();                                                                                          // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T38 = T36 + T37;                                                                                                                     // OpAdd
            Eigen::Matrix<double, 1, 1> T39 = (T38.array() / T30.array()).matrix();                                                                                          // OpCwiseDiv
            Eigen::Matrix<double, 1, 1> T40 = T35 * 0.5;                                                                                                                     // OpTimesConstScalar
            Eigen::Matrix<double, 1, 1> T41 = T39 - T40;                                                                                                                     // OpSubs
            Eigen::Matrix<double, 1, 1> T42 = (T41.array() * T2.array()).matrix();                                                                                           // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T43 = (T28.array() * T29.array()).matrix();                                                                                          // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T44 = T42.array().sqrt().matrix();                                                                                                   // OpSqrt
            Eigen::Matrix<double, 1, 1> T45 = T34({0}, {0});                                                                                                                 // OpMatBlock
            Eigen::Matrix<double, 1, 1> T46 = T34({1}, {0});                                                                                                                 // OpMatBlock
            Eigen::Matrix<double, 1, 1> T47 = T34({2}, {0});                                                                                                                 // OpMatBlock
            Eigen::Matrix<double, 1, 1> T48 = T45 - T44;                                                                                                                     // OpSubs
            Eigen::Matrix<double, 1, 1> T49 = T48.array().abs().matrix();                                                                                                    // OpAbs
            Eigen::Matrix<double, 1, 1> T50 = T45.array().abs().matrix();                                                                                                    // OpAbs
            Eigen::Matrix<double, 1, 1> T51 = T45 + T44;                                                                                                                     // OpAdd
            Eigen::Matrix<double, 1, 1> T52 = T51.array().abs().matrix();                                                                                                    // OpAbs
            Eigen::Matrix<double, 1, 1> T53 = T35.array().sqrt().matrix();                                                                                                   // OpSqrt
            Eigen::Matrix<double, 1, 1> T54 = T53 + T44;                                                                                                                     // OpAdd
            Eigen::Matrix<double, 1, 1> T55 = T3 * 0.5;                                                                                                                      // OpTimesConstScalar
            Eigen::Matrix<double, 1, 1> T56 = (T54.array() * T55.array()).matrix();                                                                                          // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T57 = (T49.array() + (T49.array().abs() / (-T56(0, 0))).exp() * T56(0, 0)).matrix();                                                 // OpHYFixExp
            Eigen::Matrix<double, 1, 1> T58 = (T52.array() + (T52.array().abs() / (-T56(0, 0))).exp() * T56(0, 0)).matrix();                                                 // OpHYFixExp
            Eigen::Matrix<double, 1, 1> T59 = T45 - T44;                                                                                                                     // OpSubs
            Eigen::Matrix<double, 1, 1> T60 = T34({2}, {0});                                                                                                                 // OpMatBlock
            Eigen::Matrix<double, 1, 1> T61 = (T45.array() * T44.array()).matrix();                                                                                          // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T62 = T39 - T61;                                                                                                                     // OpSubs
            Eigen::Matrix<double, 5, 1> T63;                                                                                                                                 // OpMatConcat
            T63(Eigen::seq(0, 0), Eigen::all) = T0;                                                                                                                          // OpMatConcat
            T63(Eigen::seq(1, 1), Eigen::all) = T59;                                                                                                                         // OpMatConcat
            T63(Eigen::seq(2, 2), Eigen::all) = T46;                                                                                                                         // OpMatConcat
            T63(Eigen::seq(3, 3), Eigen::all) = T60;                                                                                                                         // OpMatConcat
            T63(Eigen::seq(4, 4), Eigen::all) = T62;                                                                                                                         // OpMatConcat
            Eigen::Matrix<double, 1, 1> T64 = T45 + T44;                                                                                                                     // OpAdd
            Eigen::Matrix<double, 1, 1> T65 = (T45.array() * T44.array()).matrix();                                                                                          // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T66 = T39 + T65;                                                                                                                     // OpAdd
            Eigen::Matrix<double, 5, 1> T67;                                                                                                                                 // OpMatConcat
            T67(Eigen::seq(0, 0), Eigen::all) = T0;                                                                                                                          // OpMatConcat
            T67(Eigen::seq(1, 1), Eigen::all) = T64;                                                                                                                         // OpMatConcat
            T67(Eigen::seq(2, 2), Eigen::all) = T46;                                                                                                                         // OpMatConcat
            T67(Eigen::seq(3, 3), Eigen::all) = T47;                                                                                                                         // OpMatConcat
            T67(Eigen::seq(4, 4), Eigen::all) = T66;                                                                                                                         // OpMatConcat
            Eigen::Matrix<double, 1, 1> T68 = T35 * 0.5;                                                                                                                     // OpTimesConstScalar
            Eigen::Matrix<double, 5, 1> T69;                                                                                                                                 // OpMatConcat
            T69(Eigen::seq(0, 0), Eigen::all) = T0;                                                                                                                          // OpMatConcat
            T69(Eigen::seq(1, 3), Eigen::all) = T34;                                                                                                                         // OpMatConcat
            T69(Eigen::seq(4, 4), Eigen::all) = T68;                                                                                                                         // OpMatConcat
            Eigen::Matrix<double, 5, 1> T70;                                                                                                                                 // OpMatConcat
            T70(Eigen::seq(0, 0), Eigen::all) = T1;                                                                                                                          // OpMatConcat
            T70(Eigen::seq(1, 1), Eigen::all) = T1;                                                                                                                          // OpMatConcat
            T70(Eigen::seq(2, 2), Eigen::all) = T0;                                                                                                                          // OpMatConcat
            T70(Eigen::seq(3, 3), Eigen::all) = T1;                                                                                                                          // OpMatConcat
            T70(Eigen::seq(4, 4), Eigen::all) = T46;                                                                                                                         // OpMatConcat
            Eigen::Matrix<double, 5, 1> T71;                                                                                                                                 // OpMatConcat
            T71(Eigen::seq(0, 0), Eigen::all) = T1;                                                                                                                          // OpMatConcat
            T71(Eigen::seq(1, 1), Eigen::all) = T1;                                                                                                                          // OpMatConcat
            T71(Eigen::seq(2, 2), Eigen::all) = T1;                                                                                                                          // OpMatConcat
            T71(Eigen::seq(3, 3), Eigen::all) = T0;                                                                                                                          // OpMatConcat
            T71(Eigen::seq(4, 4), Eigen::all) = T47;                                                                                                                         // OpMatConcat
            Eigen::Matrix<double, 1, 1> T72 = T7 - T6;                                                                                                                       // OpSubs
            Eigen::Matrix<double, 1, 1> T73 = T23 - T19;                                                                                                                     // OpSubs
            Eigen::Matrix<double, 3, 1> T74 = T11 - T9;                                                                                                                      // OpSubs
            Eigen::Matrix<double, 1, 1> T75 = T74({0}, {0});                                                                                                                 // OpMatBlock
            Eigen::Matrix<double, 1, 1> T76 = T74({1}, {0});                                                                                                                 // OpMatBlock
            Eigen::Matrix<double, 1, 1> T77 = T74({2}, {0});                                                                                                                 // OpMatBlock
            Eigen::Matrix<double, 1, 1> T78 = (T43.array() * T44.array()).matrix();                                                                                          // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T79 = (T78.array() * T75.array()).matrix();                                                                                          // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T80 = T73 - T79;                                                                                                                     // OpSubs
            Eigen::Matrix<double, 1, 1> T81 = (T80.array() / T44.array()).matrix();                                                                                          // OpCwiseDiv
            Eigen::Matrix<double, 1, 1> T82 = T81 * 0.5;                                                                                                                     // OpTimesConstScalar
            Eigen::Matrix<double, 1, 1> T83 = (T43.array() * T44.array()).matrix();                                                                                          // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T84 = (T83.array() * T75.array()).matrix();                                                                                          // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T85 = T73 + T84;                                                                                                                     // OpAdd
            Eigen::Matrix<double, 1, 1> T86 = (T85.array() / T44.array()).matrix();                                                                                          // OpCwiseDiv
            Eigen::Matrix<double, 1, 1> T87 = T86 * 0.5;                                                                                                                     // OpTimesConstScalar
            Eigen::Matrix<double, 1, 1> T88 = (T73.array() / T42.array()).matrix();                                                                                          // OpCwiseDiv
            Eigen::Matrix<double, 1, 1> T89 = T72 - T88;                                                                                                                     // OpSubs
            Eigen::Matrix<double, 1, 1> T90 = (T43.array() * T76.array()).matrix();                                                                                          // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T91 = (T43.array() * T77.array()).matrix();                                                                                          // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T92 = T9({0}, {0});                                                                                                                  // OpMatBlock
            Eigen::Matrix<double, 1, 1> T93 = T11({0}, {0});                                                                                                                 // OpMatBlock
            Eigen::Matrix<double, 1, 1> T94 = T4({1}, {0});                                                                                                                  // OpMatBlock
            Eigen::Matrix<double, 1, 1> T95 = T4({1}, {0});                                                                                                                  // OpMatBlock
            Eigen::Matrix<double, 1, 1> T96 = (T95.array() * T92.array()).matrix();                                                                                          // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T97 = T96 + T19;                                                                                                                     // OpAdd
            Eigen::Matrix<double, 1, 1> T98 = T4({2}, {0});                                                                                                                  // OpMatBlock
            Eigen::Matrix<double, 1, 1> T99 = (T98.array() * T92.array()).matrix();                                                                                          // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T100 = T4({3}, {0});                                                                                                                 // OpMatBlock
            Eigen::Matrix<double, 1, 1> T101 = (T100.array() * T92.array()).matrix();                                                                                        // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T102 = T14 + T19;                                                                                                                    // OpAdd
            Eigen::Matrix<double, 1, 1> T103 = (T102.array() * T92.array()).matrix();                                                                                        // OpCwiseMul
            Eigen::Matrix<double, 5, 1> T104;                                                                                                                                // OpMatConcat
            T104(Eigen::seq(0, 0), Eigen::all) = T94;                                                                                                                        // OpMatConcat
            T104(Eigen::seq(1, 1), Eigen::all) = T97;                                                                                                                        // OpMatConcat
            T104(Eigen::seq(2, 2), Eigen::all) = T99;                                                                                                                        // OpMatConcat
            T104(Eigen::seq(3, 3), Eigen::all) = T101;                                                                                                                       // OpMatConcat
            T104(Eigen::seq(4, 4), Eigen::all) = T103;                                                                                                                       // OpMatConcat
            Eigen::Matrix<double, 1, 1> T105 = T5({1}, {0});                                                                                                                 // OpMatBlock
            Eigen::Matrix<double, 1, 1> T106 = T5({1}, {0});                                                                                                                 // OpMatBlock
            Eigen::Matrix<double, 1, 1> T107 = (T106.array() * T93.array()).matrix();                                                                                        // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T108 = T107 + T23;                                                                                                                   // OpAdd
            Eigen::Matrix<double, 1, 1> T109 = T5({2}, {0});                                                                                                                 // OpMatBlock
            Eigen::Matrix<double, 1, 1> T110 = (T109.array() * T93.array()).matrix();                                                                                        // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T111 = T5({3}, {0});                                                                                                                 // OpMatBlock
            Eigen::Matrix<double, 1, 1> T112 = (T111.array() * T93.array()).matrix();                                                                                        // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T113 = T15 + T23;                                                                                                                    // OpAdd
            Eigen::Matrix<double, 1, 1> T114 = (T113.array() * T93.array()).matrix();                                                                                        // OpCwiseMul
            Eigen::Matrix<double, 5, 1> T115;                                                                                                                                // OpMatConcat
            T115(Eigen::seq(0, 0), Eigen::all) = T105;                                                                                                                       // OpMatConcat
            T115(Eigen::seq(1, 1), Eigen::all) = T108;                                                                                                                       // OpMatConcat
            T115(Eigen::seq(2, 2), Eigen::all) = T110;                                                                                                                       // OpMatConcat
            T115(Eigen::seq(3, 3), Eigen::all) = T112;                                                                                                                       // OpMatConcat
            T115(Eigen::seq(4, 4), Eigen::all) = T114;                                                                                                                       // OpMatConcat
            Eigen::Matrix<double, 1, 1> T116 = (T82.array() * T57.array()).matrix();                                                                                         // OpCwiseMul
            Eigen::Matrix<double, 5, 1> T117 = T63 * T116(0, 0);                                                                                                             // OpTimesScalar
            Eigen::Matrix<double, 1, 1> T118 = (T87.array() * T58.array()).matrix();                                                                                         // OpCwiseMul
            Eigen::Matrix<double, 5, 1> T119 = T67 * T118(0, 0);                                                                                                             // OpTimesScalar
            Eigen::Matrix<double, 5, 1> T120 = T117 + T119;                                                                                                                  // OpAdd
            Eigen::Matrix<double, 1, 1> T121 = (T89.array() * T50.array()).matrix();                                                                                         // OpCwiseMul
            Eigen::Matrix<double, 5, 1> T122 = T69 * T121(0, 0);                                                                                                             // OpTimesScalar
            Eigen::Matrix<double, 5, 1> T123 = T120 + T122;                                                                                                                  // OpAdd
            Eigen::Matrix<double, 1, 1> T124 = (T90.array() * T50.array()).matrix();                                                                                         // OpCwiseMul
            Eigen::Matrix<double, 5, 1> T125 = T70 * T124(0, 0);                                                                                                             // OpTimesScalar
            Eigen::Matrix<double, 5, 1> T126 = T123 + T125;                                                                                                                  // OpAdd
            Eigen::Matrix<double, 1, 1> T127 = (T91.array() * T50.array()).matrix();                                                                                         // OpCwiseMul
            Eigen::Matrix<double, 5, 1> T128 = T71 * T127(0, 0);                                                                                                             // OpTimesScalar
            Eigen::Matrix<double, 5, 1> T129 = T126 + T128;                                                                                                                  // OpAdd
            Eigen::Matrix<double, 5, 1> T130 = T104 + T115;                                                                                                                  // OpAdd
            Eigen::Matrix<double, 5, 1> T131 = T130 - T129;                                                                                                                  // OpSubs
            Eigen::Matrix<double, 5, 1> T132 = T131 * 0.5;                                                                                                                   // OpTimesConstScalar
            Eigen::Matrix<double, 5, 5> g_T132;                                                                                                                              // Init Grad
            g_T132.setZero();                                                                                                                                                // Init Grad
            g_T132(Eigen::all, Eigen::seq(0, 0))(0) = 1.0;                                                                                                                   // Init Grad
            g_T132(Eigen::all, Eigen::seq(1, 1))(1) = 1.0;                                                                                                                   // Init Grad
            g_T132(Eigen::all, Eigen::seq(2, 2))(2) = 1.0;                                                                                                                   // Init Grad
            g_T132(Eigen::all, Eigen::seq(3, 3))(3) = 1.0;                                                                                                                   // Init Grad
            g_T132(Eigen::all, Eigen::seq(4, 4))(4) = 1.0;                                                                                                                   // Init Grad
            Eigen::Matrix<double, 5, 5> g_T131;                                                                                                                              // Init Grad Zero
            g_T131.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 5, 5> g_T129;                                                                                                                              // Init Grad Zero
            g_T129.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 5, 5> g_T126;                                                                                                                              // Init Grad Zero
            g_T126.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 5, 5> g_T123;                                                                                                                              // Init Grad Zero
            g_T123.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 5, 5> g_T120;                                                                                                                              // Init Grad Zero
            g_T120.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 5, 5> g_T117;                                                                                                                              // Init Grad Zero
            g_T117.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 5, 5> g_T63;                                                                                                                               // Init Grad Zero
            g_T63.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T0;                                                                                                                                // Init Grad Zero
            g_T0.setZero();                                                                                                                                                  // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T46;                                                                                                                               // Init Grad Zero
            g_T46.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 3, 5> g_T34;                                                                                                                               // Init Grad Zero
            g_T34.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T30;                                                                                                                               // Init Grad Zero
            g_T30.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T28;                                                                                                                               // Init Grad Zero
            g_T28.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T6;                                                                                                                                // Init Grad Zero
            g_T6.setZero();                                                                                                                                                  // Init Grad Zero
            Eigen::Matrix<double, 5, 5> g_T4;                                                                                                                                // Init Grad Zero
            g_T4.setZero();                                                                                                                                                  // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T29;                                                                                                                               // Init Grad Zero
            g_T29.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T7;                                                                                                                                // Init Grad Zero
            g_T7.setZero();                                                                                                                                                  // Init Grad Zero
            Eigen::Matrix<double, 5, 5> g_T5;                                                                                                                                // Init Grad Zero
            g_T5.setZero();                                                                                                                                                  // Init Grad Zero
            Eigen::Matrix<double, 3, 5> g_T33;                                                                                                                               // Init Grad Zero
            g_T33.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 3, 5> g_T31;                                                                                                                               // Init Grad Zero
            g_T31.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 3, 5> g_T9;                                                                                                                                // Init Grad Zero
            g_T9.setZero();                                                                                                                                                  // Init Grad Zero
            Eigen::Matrix<double, 3, 5> g_T8;                                                                                                                                // Init Grad Zero
            g_T8.setZero();                                                                                                                                                  // Init Grad Zero
            Eigen::Matrix<double, 3, 5> g_T32;                                                                                                                               // Init Grad Zero
            g_T32.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 3, 5> g_T11;                                                                                                                               // Init Grad Zero
            g_T11.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 3, 5> g_T10;                                                                                                                               // Init Grad Zero
            g_T10.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T59;                                                                                                                               // Init Grad Zero
            g_T59.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T44;                                                                                                                               // Init Grad Zero
            g_T44.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T42;                                                                                                                               // Init Grad Zero
            g_T42.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T2;                                                                                                                                // Init Grad Zero
            g_T2.setZero();                                                                                                                                                  // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T41;                                                                                                                               // Init Grad Zero
            g_T41.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T39;                                                                                                                               // Init Grad Zero
            g_T39.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T38;                                                                                                                               // Init Grad Zero
            g_T38.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T36;                                                                                                                               // Init Grad Zero
            g_T36.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T25;                                                                                                                               // Init Grad Zero
            g_T25.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T24;                                                                                                                               // Init Grad Zero
            g_T24.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T14;                                                                                                                               // Init Grad Zero
            g_T14.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T19;                                                                                                                               // Init Grad Zero
            g_T19.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T18;                                                                                                                               // Init Grad Zero
            g_T18.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T17;                                                                                                                               // Init Grad Zero
            g_T17.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T16;                                                                                                                               // Init Grad Zero
            g_T16.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T12;                                                                                                                               // Init Grad Zero
            g_T12.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T37;                                                                                                                               // Init Grad Zero
            g_T37.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T27;                                                                                                                               // Init Grad Zero
            g_T27.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T26;                                                                                                                               // Init Grad Zero
            g_T26.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T15;                                                                                                                               // Init Grad Zero
            g_T15.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T23;                                                                                                                               // Init Grad Zero
            g_T23.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T22;                                                                                                                               // Init Grad Zero
            g_T22.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T21;                                                                                                                               // Init Grad Zero
            g_T21.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T20;                                                                                                                               // Init Grad Zero
            g_T20.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T13;                                                                                                                               // Init Grad Zero
            g_T13.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T40;                                                                                                                               // Init Grad Zero
            g_T40.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T35;                                                                                                                               // Init Grad Zero
            g_T35.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T45;                                                                                                                               // Init Grad Zero
            g_T45.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T60;                                                                                                                               // Init Grad Zero
            g_T60.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T62;                                                                                                                               // Init Grad Zero
            g_T62.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T61;                                                                                                                               // Init Grad Zero
            g_T61.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T116;                                                                                                                              // Init Grad Zero
            g_T116.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T57;                                                                                                                               // Init Grad Zero
            g_T57.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T49;                                                                                                                               // Init Grad Zero
            g_T49.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T48;                                                                                                                               // Init Grad Zero
            g_T48.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T56;                                                                                                                               // Init Grad Zero
            g_T56.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T54;                                                                                                                               // Init Grad Zero
            g_T54.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T53;                                                                                                                               // Init Grad Zero
            g_T53.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T55;                                                                                                                               // Init Grad Zero
            g_T55.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T3;                                                                                                                                // Init Grad Zero
            g_T3.setZero();                                                                                                                                                  // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T82;                                                                                                                               // Init Grad Zero
            g_T82.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T81;                                                                                                                               // Init Grad Zero
            g_T81.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T80;                                                                                                                               // Init Grad Zero
            g_T80.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T73;                                                                                                                               // Init Grad Zero
            g_T73.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T79;                                                                                                                               // Init Grad Zero
            g_T79.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T75;                                                                                                                               // Init Grad Zero
            g_T75.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 3, 5> g_T74;                                                                                                                               // Init Grad Zero
            g_T74.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T78;                                                                                                                               // Init Grad Zero
            g_T78.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T43;                                                                                                                               // Init Grad Zero
            g_T43.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 5, 5> g_T119;                                                                                                                              // Init Grad Zero
            g_T119.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 5, 5> g_T67;                                                                                                                               // Init Grad Zero
            g_T67.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T47;                                                                                                                               // Init Grad Zero
            g_T47.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T64;                                                                                                                               // Init Grad Zero
            g_T64.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T66;                                                                                                                               // Init Grad Zero
            g_T66.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T65;                                                                                                                               // Init Grad Zero
            g_T65.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T118;                                                                                                                              // Init Grad Zero
            g_T118.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T58;                                                                                                                               // Init Grad Zero
            g_T58.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T52;                                                                                                                               // Init Grad Zero
            g_T52.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T51;                                                                                                                               // Init Grad Zero
            g_T51.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T87;                                                                                                                               // Init Grad Zero
            g_T87.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T86;                                                                                                                               // Init Grad Zero
            g_T86.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T85;                                                                                                                               // Init Grad Zero
            g_T85.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T84;                                                                                                                               // Init Grad Zero
            g_T84.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T83;                                                                                                                               // Init Grad Zero
            g_T83.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 5, 5> g_T122;                                                                                                                              // Init Grad Zero
            g_T122.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 5, 5> g_T69;                                                                                                                               // Init Grad Zero
            g_T69.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T68;                                                                                                                               // Init Grad Zero
            g_T68.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T121;                                                                                                                              // Init Grad Zero
            g_T121.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T50;                                                                                                                               // Init Grad Zero
            g_T50.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T89;                                                                                                                               // Init Grad Zero
            g_T89.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T72;                                                                                                                               // Init Grad Zero
            g_T72.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T88;                                                                                                                               // Init Grad Zero
            g_T88.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 5, 5> g_T125;                                                                                                                              // Init Grad Zero
            g_T125.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 5, 5> g_T70;                                                                                                                               // Init Grad Zero
            g_T70.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T1;                                                                                                                                // Init Grad Zero
            g_T1.setZero();                                                                                                                                                  // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T124;                                                                                                                              // Init Grad Zero
            g_T124.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T90;                                                                                                                               // Init Grad Zero
            g_T90.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T76;                                                                                                                               // Init Grad Zero
            g_T76.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 5, 5> g_T128;                                                                                                                              // Init Grad Zero
            g_T128.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 5, 5> g_T71;                                                                                                                               // Init Grad Zero
            g_T71.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T127;                                                                                                                              // Init Grad Zero
            g_T127.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T91;                                                                                                                               // Init Grad Zero
            g_T91.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T77;                                                                                                                               // Init Grad Zero
            g_T77.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 5, 5> g_T130;                                                                                                                              // Init Grad Zero
            g_T130.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 5, 5> g_T104;                                                                                                                              // Init Grad Zero
            g_T104.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T94;                                                                                                                               // Init Grad Zero
            g_T94.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T97;                                                                                                                               // Init Grad Zero
            g_T97.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T96;                                                                                                                               // Init Grad Zero
            g_T96.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T92;                                                                                                                               // Init Grad Zero
            g_T92.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T95;                                                                                                                               // Init Grad Zero
            g_T95.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T99;                                                                                                                               // Init Grad Zero
            g_T99.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T98;                                                                                                                               // Init Grad Zero
            g_T98.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T101;                                                                                                                              // Init Grad Zero
            g_T101.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T100;                                                                                                                              // Init Grad Zero
            g_T100.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T103;                                                                                                                              // Init Grad Zero
            g_T103.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T102;                                                                                                                              // Init Grad Zero
            g_T102.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 5, 5> g_T115;                                                                                                                              // Init Grad Zero
            g_T115.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T105;                                                                                                                              // Init Grad Zero
            g_T105.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T108;                                                                                                                              // Init Grad Zero
            g_T108.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T107;                                                                                                                              // Init Grad Zero
            g_T107.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T93;                                                                                                                               // Init Grad Zero
            g_T93.setZero();                                                                                                                                                 // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T106;                                                                                                                              // Init Grad Zero
            g_T106.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T110;                                                                                                                              // Init Grad Zero
            g_T110.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T109;                                                                                                                              // Init Grad Zero
            g_T109.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T112;                                                                                                                              // Init Grad Zero
            g_T112.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T111;                                                                                                                              // Init Grad Zero
            g_T111.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T114;                                                                                                                              // Init Grad Zero
            g_T114.setZero();                                                                                                                                                // Init Grad Zero
            Eigen::Matrix<double, 1, 5> g_T113;                                                                                                                              // Init Grad Zero
            g_T113.setZero();                                                                                                                                                // Init Grad Zero
            g_T131 += g_T132 * 0.5;                                                                                                                                          // OpTimesConstScalar
            g_T130 += g_T131;                                                                                                                                                // OpSubs
            g_T129 -= g_T131;                                                                                                                                                // OpSubs
            g_T126 += g_T129;                                                                                                                                                // OpAdd
            g_T128 += g_T129;                                                                                                                                                // OpAdd
            g_T123 += g_T126;                                                                                                                                                // OpAdd
            g_T125 += g_T126;                                                                                                                                                // OpAdd
            g_T120 += g_T123;                                                                                                                                                // OpAdd
            g_T122 += g_T123;                                                                                                                                                // OpAdd
            g_T117 += g_T120;                                                                                                                                                // OpAdd
            g_T119 += g_T120;                                                                                                                                                // OpAdd
            g_T63 += g_T117 * T116(0, 0);                                                                                                                                    // OpTimesScalar
            g_T116(0) += (g_T117(Eigen::all, Eigen::seq(0, 0)).array() * T63.array()).sum();                                                                                 // OpTimesScalar
            g_T116(1) += (g_T117(Eigen::all, Eigen::seq(1, 1)).array() * T63.array()).sum();                                                                                 // OpTimesScalar
            g_T116(2) += (g_T117(Eigen::all, Eigen::seq(2, 2)).array() * T63.array()).sum();                                                                                 // OpTimesScalar
            g_T116(3) += (g_T117(Eigen::all, Eigen::seq(3, 3)).array() * T63.array()).sum();                                                                                 // OpTimesScalar
            g_T116(4) += (g_T117(Eigen::all, Eigen::seq(4, 4)).array() * T63.array()).sum();                                                                                 // OpTimesScalar
            g_T0 += g_T63(Eigen::seq(0, 0), Eigen::all);                                                                                                                     // OpMatConcat
            g_T59 += g_T63(Eigen::seq(1, 1), Eigen::all);                                                                                                                    // OpMatConcat
            g_T46 += g_T63(Eigen::seq(2, 2), Eigen::all);                                                                                                                    // OpMatConcat
            g_T60 += g_T63(Eigen::seq(3, 3), Eigen::all);                                                                                                                    // OpMatConcat
            g_T62 += g_T63(Eigen::seq(4, 4), Eigen::all);                                                                                                                    // OpMatConcat
            g_T45 += g_T59;                                                                                                                                                  // OpSubs
            g_T44 -= g_T59;                                                                                                                                                  // OpSubs
            g_T34(Eigen::all, Eigen::seq(0, 0))({2}, {0}) += g_T60(Eigen::all, Eigen::seq(0, 0));                                                                            // OpMatBlock
            g_T34(Eigen::all, Eigen::seq(1, 1))({2}, {0}) += g_T60(Eigen::all, Eigen::seq(1, 1));                                                                            // OpMatBlock
            g_T34(Eigen::all, Eigen::seq(2, 2))({2}, {0}) += g_T60(Eigen::all, Eigen::seq(2, 2));                                                                            // OpMatBlock
            g_T34(Eigen::all, Eigen::seq(3, 3))({2}, {0}) += g_T60(Eigen::all, Eigen::seq(3, 3));                                                                            // OpMatBlock
            g_T34(Eigen::all, Eigen::seq(4, 4))({2}, {0}) += g_T60(Eigen::all, Eigen::seq(4, 4));                                                                            // OpMatBlock
            g_T39 += g_T62;                                                                                                                                                  // OpSubs
            g_T61 -= g_T62;                                                                                                                                                  // OpSubs
            g_T45(Eigen::all, Eigen::seq(0, 0)).array() += g_T61(Eigen::all, Eigen::seq(0, 0)).array() * T44.array();                                                        // OpCwiseMul
            g_T45(Eigen::all, Eigen::seq(1, 1)).array() += g_T61(Eigen::all, Eigen::seq(1, 1)).array() * T44.array();                                                        // OpCwiseMul
            g_T45(Eigen::all, Eigen::seq(2, 2)).array() += g_T61(Eigen::all, Eigen::seq(2, 2)).array() * T44.array();                                                        // OpCwiseMul
            g_T45(Eigen::all, Eigen::seq(3, 3)).array() += g_T61(Eigen::all, Eigen::seq(3, 3)).array() * T44.array();                                                        // OpCwiseMul
            g_T45(Eigen::all, Eigen::seq(4, 4)).array() += g_T61(Eigen::all, Eigen::seq(4, 4)).array() * T44.array();                                                        // OpCwiseMul
            g_T44(Eigen::all, Eigen::seq(0, 0)).array() += g_T61(Eigen::all, Eigen::seq(0, 0)).array() * T45.array();                                                        // OpCwiseMul
            g_T44(Eigen::all, Eigen::seq(1, 1)).array() += g_T61(Eigen::all, Eigen::seq(1, 1)).array() * T45.array();                                                        // OpCwiseMul
            g_T44(Eigen::all, Eigen::seq(2, 2)).array() += g_T61(Eigen::all, Eigen::seq(2, 2)).array() * T45.array();                                                        // OpCwiseMul
            g_T44(Eigen::all, Eigen::seq(3, 3)).array() += g_T61(Eigen::all, Eigen::seq(3, 3)).array() * T45.array();                                                        // OpCwiseMul
            g_T44(Eigen::all, Eigen::seq(4, 4)).array() += g_T61(Eigen::all, Eigen::seq(4, 4)).array() * T45.array();                                                        // OpCwiseMul
            g_T82(Eigen::all, Eigen::seq(0, 0)).array() += g_T116(Eigen::all, Eigen::seq(0, 0)).array() * T57.array();                                                       // OpCwiseMul
            g_T82(Eigen::all, Eigen::seq(1, 1)).array() += g_T116(Eigen::all, Eigen::seq(1, 1)).array() * T57.array();                                                       // OpCwiseMul
            g_T82(Eigen::all, Eigen::seq(2, 2)).array() += g_T116(Eigen::all, Eigen::seq(2, 2)).array() * T57.array();                                                       // OpCwiseMul
            g_T82(Eigen::all, Eigen::seq(3, 3)).array() += g_T116(Eigen::all, Eigen::seq(3, 3)).array() * T57.array();                                                       // OpCwiseMul
            g_T82(Eigen::all, Eigen::seq(4, 4)).array() += g_T116(Eigen::all, Eigen::seq(4, 4)).array() * T57.array();                                                       // OpCwiseMul
            g_T57(Eigen::all, Eigen::seq(0, 0)).array() += g_T116(Eigen::all, Eigen::seq(0, 0)).array() * T82.array();                                                       // OpCwiseMul
            g_T57(Eigen::all, Eigen::seq(1, 1)).array() += g_T116(Eigen::all, Eigen::seq(1, 1)).array() * T82.array();                                                       // OpCwiseMul
            g_T57(Eigen::all, Eigen::seq(2, 2)).array() += g_T116(Eigen::all, Eigen::seq(2, 2)).array() * T82.array();                                                       // OpCwiseMul
            g_T57(Eigen::all, Eigen::seq(3, 3)).array() += g_T116(Eigen::all, Eigen::seq(3, 3)).array() * T82.array();                                                       // OpCwiseMul
            g_T57(Eigen::all, Eigen::seq(4, 4)).array() += g_T116(Eigen::all, Eigen::seq(4, 4)).array() * T82.array();                                                       // OpCwiseMul
            g_T49(Eigen::all, Eigen::seq(0, 0)).array() += g_T57(Eigen::all, Eigen::seq(0, 0)).array() * (1 + (T49.array() - T57.array()) * T49.array().sign() / T56(0, 0)); // OpHYFixExp
            g_T49(Eigen::all, Eigen::seq(1, 1)).array() += g_T57(Eigen::all, Eigen::seq(1, 1)).array() * (1 + (T49.array() - T57.array()) * T49.array().sign() / T56(0, 0)); // OpHYFixExp
            g_T49(Eigen::all, Eigen::seq(2, 2)).array() += g_T57(Eigen::all, Eigen::seq(2, 2)).array() * (1 + (T49.array() - T57.array()) * T49.array().sign() / T56(0, 0)); // OpHYFixExp
            g_T49(Eigen::all, Eigen::seq(3, 3)).array() += g_T57(Eigen::all, Eigen::seq(3, 3)).array() * (1 + (T49.array() - T57.array()) * T49.array().sign() / T56(0, 0)); // OpHYFixExp
            g_T49(Eigen::all, Eigen::seq(4, 4)).array() += g_T57(Eigen::all, Eigen::seq(4, 4)).array() * (1 + (T49.array() - T57.array()) * T49.array().sign() / T56(0, 0)); // OpHYFixExp
            g_T56(0) += (g_T57(Eigen::all, Eigen::seq(0, 0)).array() * (T57.array() - T49.array()) * (T49.array().abs() + T56(0, 0)) / (T56(0, 0) * T56(0, 0))).sum();       // OpHYFixExp
            g_T56(1) += (g_T57(Eigen::all, Eigen::seq(1, 1)).array() * (T57.array() - T49.array()) * (T49.array().abs() + T56(0, 0)) / (T56(0, 0) * T56(0, 0))).sum();       // OpHYFixExp
            g_T56(2) += (g_T57(Eigen::all, Eigen::seq(2, 2)).array() * (T57.array() - T49.array()) * (T49.array().abs() + T56(0, 0)) / (T56(0, 0) * T56(0, 0))).sum();       // OpHYFixExp
            g_T56(3) += (g_T57(Eigen::all, Eigen::seq(3, 3)).array() * (T57.array() - T49.array()) * (T49.array().abs() + T56(0, 0)) / (T56(0, 0) * T56(0, 0))).sum();       // OpHYFixExp
            g_T56(4) += (g_T57(Eigen::all, Eigen::seq(4, 4)).array() * (T57.array() - T49.array()) * (T49.array().abs() + T56(0, 0)) / (T56(0, 0) * T56(0, 0))).sum();       // OpHYFixExp
            g_T48(Eigen::all, Eigen::seq(0, 0)).array() += g_T49(Eigen::all, Eigen::seq(0, 0)).array() * T48.array().sign();                                                 // OpAbs
            g_T48(Eigen::all, Eigen::seq(1, 1)).array() += g_T49(Eigen::all, Eigen::seq(1, 1)).array() * T48.array().sign();                                                 // OpAbs
            g_T48(Eigen::all, Eigen::seq(2, 2)).array() += g_T49(Eigen::all, Eigen::seq(2, 2)).array() * T48.array().sign();                                                 // OpAbs
            g_T48(Eigen::all, Eigen::seq(3, 3)).array() += g_T49(Eigen::all, Eigen::seq(3, 3)).array() * T48.array().sign();                                                 // OpAbs
            g_T48(Eigen::all, Eigen::seq(4, 4)).array() += g_T49(Eigen::all, Eigen::seq(4, 4)).array() * T48.array().sign();                                                 // OpAbs
            g_T45 += g_T48;                                                                                                                                                  // OpSubs
            g_T44 -= g_T48;                                                                                                                                                  // OpSubs
            g_T81 += g_T82 * 0.5;                                                                                                                                            // OpTimesConstScalar
            g_T80(Eigen::all, Eigen::seq(0, 0)).array() += g_T81(Eigen::all, Eigen::seq(0, 0)).array() / T44.array();                                                        // OpCwiseDiv
            g_T80(Eigen::all, Eigen::seq(1, 1)).array() += g_T81(Eigen::all, Eigen::seq(1, 1)).array() / T44.array();                                                        // OpCwiseDiv
            g_T80(Eigen::all, Eigen::seq(2, 2)).array() += g_T81(Eigen::all, Eigen::seq(2, 2)).array() / T44.array();                                                        // OpCwiseDiv
            g_T80(Eigen::all, Eigen::seq(3, 3)).array() += g_T81(Eigen::all, Eigen::seq(3, 3)).array() / T44.array();                                                        // OpCwiseDiv
            g_T80(Eigen::all, Eigen::seq(4, 4)).array() += g_T81(Eigen::all, Eigen::seq(4, 4)).array() / T44.array();                                                        // OpCwiseDiv
            g_T44(Eigen::all, Eigen::seq(0, 0)).array() -= g_T81(Eigen::all, Eigen::seq(0, 0)).array() * T81.array() / T44.array();                                          // OpCwiseDiv
            g_T44(Eigen::all, Eigen::seq(1, 1)).array() -= g_T81(Eigen::all, Eigen::seq(1, 1)).array() * T81.array() / T44.array();                                          // OpCwiseDiv
            g_T44(Eigen::all, Eigen::seq(2, 2)).array() -= g_T81(Eigen::all, Eigen::seq(2, 2)).array() * T81.array() / T44.array();                                          // OpCwiseDiv
            g_T44(Eigen::all, Eigen::seq(3, 3)).array() -= g_T81(Eigen::all, Eigen::seq(3, 3)).array() * T81.array() / T44.array();                                          // OpCwiseDiv
            g_T44(Eigen::all, Eigen::seq(4, 4)).array() -= g_T81(Eigen::all, Eigen::seq(4, 4)).array() * T81.array() / T44.array();                                          // OpCwiseDiv
            g_T73 += g_T80;                                                                                                                                                  // OpSubs
            g_T79 -= g_T80;                                                                                                                                                  // OpSubs
            g_T78(Eigen::all, Eigen::seq(0, 0)).array() += g_T79(Eigen::all, Eigen::seq(0, 0)).array() * T75.array();                                                        // OpCwiseMul
            g_T78(Eigen::all, Eigen::seq(1, 1)).array() += g_T79(Eigen::all, Eigen::seq(1, 1)).array() * T75.array();                                                        // OpCwiseMul
            g_T78(Eigen::all, Eigen::seq(2, 2)).array() += g_T79(Eigen::all, Eigen::seq(2, 2)).array() * T75.array();                                                        // OpCwiseMul
            g_T78(Eigen::all, Eigen::seq(3, 3)).array() += g_T79(Eigen::all, Eigen::seq(3, 3)).array() * T75.array();                                                        // OpCwiseMul
            g_T78(Eigen::all, Eigen::seq(4, 4)).array() += g_T79(Eigen::all, Eigen::seq(4, 4)).array() * T75.array();                                                        // OpCwiseMul
            g_T75(Eigen::all, Eigen::seq(0, 0)).array() += g_T79(Eigen::all, Eigen::seq(0, 0)).array() * T78.array();                                                        // OpCwiseMul
            g_T75(Eigen::all, Eigen::seq(1, 1)).array() += g_T79(Eigen::all, Eigen::seq(1, 1)).array() * T78.array();                                                        // OpCwiseMul
            g_T75(Eigen::all, Eigen::seq(2, 2)).array() += g_T79(Eigen::all, Eigen::seq(2, 2)).array() * T78.array();                                                        // OpCwiseMul
            g_T75(Eigen::all, Eigen::seq(3, 3)).array() += g_T79(Eigen::all, Eigen::seq(3, 3)).array() * T78.array();                                                        // OpCwiseMul
            g_T75(Eigen::all, Eigen::seq(4, 4)).array() += g_T79(Eigen::all, Eigen::seq(4, 4)).array() * T78.array();                                                        // OpCwiseMul
            g_T43(Eigen::all, Eigen::seq(0, 0)).array() += g_T78(Eigen::all, Eigen::seq(0, 0)).array() * T44.array();                                                        // OpCwiseMul
            g_T43(Eigen::all, Eigen::seq(1, 1)).array() += g_T78(Eigen::all, Eigen::seq(1, 1)).array() * T44.array();                                                        // OpCwiseMul
            g_T43(Eigen::all, Eigen::seq(2, 2)).array() += g_T78(Eigen::all, Eigen::seq(2, 2)).array() * T44.array();                                                        // OpCwiseMul
            g_T43(Eigen::all, Eigen::seq(3, 3)).array() += g_T78(Eigen::all, Eigen::seq(3, 3)).array() * T44.array();                                                        // OpCwiseMul
            g_T43(Eigen::all, Eigen::seq(4, 4)).array() += g_T78(Eigen::all, Eigen::seq(4, 4)).array() * T44.array();                                                        // OpCwiseMul
            g_T44(Eigen::all, Eigen::seq(0, 0)).array() += g_T78(Eigen::all, Eigen::seq(0, 0)).array() * T43.array();                                                        // OpCwiseMul
            g_T44(Eigen::all, Eigen::seq(1, 1)).array() += g_T78(Eigen::all, Eigen::seq(1, 1)).array() * T43.array();                                                        // OpCwiseMul
            g_T44(Eigen::all, Eigen::seq(2, 2)).array() += g_T78(Eigen::all, Eigen::seq(2, 2)).array() * T43.array();                                                        // OpCwiseMul
            g_T44(Eigen::all, Eigen::seq(3, 3)).array() += g_T78(Eigen::all, Eigen::seq(3, 3)).array() * T43.array();                                                        // OpCwiseMul
            g_T44(Eigen::all, Eigen::seq(4, 4)).array() += g_T78(Eigen::all, Eigen::seq(4, 4)).array() * T43.array();                                                        // OpCwiseMul
            g_T67 += g_T119 * T118(0, 0);                                                                                                                                    // OpTimesScalar
            g_T118(0) += (g_T119(Eigen::all, Eigen::seq(0, 0)).array() * T67.array()).sum();                                                                                 // OpTimesScalar
            g_T118(1) += (g_T119(Eigen::all, Eigen::seq(1, 1)).array() * T67.array()).sum();                                                                                 // OpTimesScalar
            g_T118(2) += (g_T119(Eigen::all, Eigen::seq(2, 2)).array() * T67.array()).sum();                                                                                 // OpTimesScalar
            g_T118(3) += (g_T119(Eigen::all, Eigen::seq(3, 3)).array() * T67.array()).sum();                                                                                 // OpTimesScalar
            g_T118(4) += (g_T119(Eigen::all, Eigen::seq(4, 4)).array() * T67.array()).sum();                                                                                 // OpTimesScalar
            g_T0 += g_T67(Eigen::seq(0, 0), Eigen::all);                                                                                                                     // OpMatConcat
            g_T64 += g_T67(Eigen::seq(1, 1), Eigen::all);                                                                                                                    // OpMatConcat
            g_T46 += g_T67(Eigen::seq(2, 2), Eigen::all);                                                                                                                    // OpMatConcat
            g_T47 += g_T67(Eigen::seq(3, 3), Eigen::all);                                                                                                                    // OpMatConcat
            g_T66 += g_T67(Eigen::seq(4, 4), Eigen::all);                                                                                                                    // OpMatConcat
            g_T45 += g_T64;                                                                                                                                                  // OpAdd
            g_T44 += g_T64;                                                                                                                                                  // OpAdd
            g_T39 += g_T66;                                                                                                                                                  // OpAdd
            g_T65 += g_T66;                                                                                                                                                  // OpAdd
            g_T45(Eigen::all, Eigen::seq(0, 0)).array() += g_T65(Eigen::all, Eigen::seq(0, 0)).array() * T44.array();                                                        // OpCwiseMul
            g_T45(Eigen::all, Eigen::seq(1, 1)).array() += g_T65(Eigen::all, Eigen::seq(1, 1)).array() * T44.array();                                                        // OpCwiseMul
            g_T45(Eigen::all, Eigen::seq(2, 2)).array() += g_T65(Eigen::all, Eigen::seq(2, 2)).array() * T44.array();                                                        // OpCwiseMul
            g_T45(Eigen::all, Eigen::seq(3, 3)).array() += g_T65(Eigen::all, Eigen::seq(3, 3)).array() * T44.array();                                                        // OpCwiseMul
            g_T45(Eigen::all, Eigen::seq(4, 4)).array() += g_T65(Eigen::all, Eigen::seq(4, 4)).array() * T44.array();                                                        // OpCwiseMul
            g_T44(Eigen::all, Eigen::seq(0, 0)).array() += g_T65(Eigen::all, Eigen::seq(0, 0)).array() * T45.array();                                                        // OpCwiseMul
            g_T44(Eigen::all, Eigen::seq(1, 1)).array() += g_T65(Eigen::all, Eigen::seq(1, 1)).array() * T45.array();                                                        // OpCwiseMul
            g_T44(Eigen::all, Eigen::seq(2, 2)).array() += g_T65(Eigen::all, Eigen::seq(2, 2)).array() * T45.array();                                                        // OpCwiseMul
            g_T44(Eigen::all, Eigen::seq(3, 3)).array() += g_T65(Eigen::all, Eigen::seq(3, 3)).array() * T45.array();                                                        // OpCwiseMul
            g_T44(Eigen::all, Eigen::seq(4, 4)).array() += g_T65(Eigen::all, Eigen::seq(4, 4)).array() * T45.array();                                                        // OpCwiseMul
            g_T87(Eigen::all, Eigen::seq(0, 0)).array() += g_T118(Eigen::all, Eigen::seq(0, 0)).array() * T58.array();                                                       // OpCwiseMul
            g_T87(Eigen::all, Eigen::seq(1, 1)).array() += g_T118(Eigen::all, Eigen::seq(1, 1)).array() * T58.array();                                                       // OpCwiseMul
            g_T87(Eigen::all, Eigen::seq(2, 2)).array() += g_T118(Eigen::all, Eigen::seq(2, 2)).array() * T58.array();                                                       // OpCwiseMul
            g_T87(Eigen::all, Eigen::seq(3, 3)).array() += g_T118(Eigen::all, Eigen::seq(3, 3)).array() * T58.array();                                                       // OpCwiseMul
            g_T87(Eigen::all, Eigen::seq(4, 4)).array() += g_T118(Eigen::all, Eigen::seq(4, 4)).array() * T58.array();                                                       // OpCwiseMul
            g_T58(Eigen::all, Eigen::seq(0, 0)).array() += g_T118(Eigen::all, Eigen::seq(0, 0)).array() * T87.array();                                                       // OpCwiseMul
            g_T58(Eigen::all, Eigen::seq(1, 1)).array() += g_T118(Eigen::all, Eigen::seq(1, 1)).array() * T87.array();                                                       // OpCwiseMul
            g_T58(Eigen::all, Eigen::seq(2, 2)).array() += g_T118(Eigen::all, Eigen::seq(2, 2)).array() * T87.array();                                                       // OpCwiseMul
            g_T58(Eigen::all, Eigen::seq(3, 3)).array() += g_T118(Eigen::all, Eigen::seq(3, 3)).array() * T87.array();                                                       // OpCwiseMul
            g_T58(Eigen::all, Eigen::seq(4, 4)).array() += g_T118(Eigen::all, Eigen::seq(4, 4)).array() * T87.array();                                                       // OpCwiseMul
            g_T52(Eigen::all, Eigen::seq(0, 0)).array() += g_T58(Eigen::all, Eigen::seq(0, 0)).array() * (1 + (T52.array() - T58.array()) * T52.array().sign() / T56(0, 0)); // OpHYFixExp
            g_T52(Eigen::all, Eigen::seq(1, 1)).array() += g_T58(Eigen::all, Eigen::seq(1, 1)).array() * (1 + (T52.array() - T58.array()) * T52.array().sign() / T56(0, 0)); // OpHYFixExp
            g_T52(Eigen::all, Eigen::seq(2, 2)).array() += g_T58(Eigen::all, Eigen::seq(2, 2)).array() * (1 + (T52.array() - T58.array()) * T52.array().sign() / T56(0, 0)); // OpHYFixExp
            g_T52(Eigen::all, Eigen::seq(3, 3)).array() += g_T58(Eigen::all, Eigen::seq(3, 3)).array() * (1 + (T52.array() - T58.array()) * T52.array().sign() / T56(0, 0)); // OpHYFixExp
            g_T52(Eigen::all, Eigen::seq(4, 4)).array() += g_T58(Eigen::all, Eigen::seq(4, 4)).array() * (1 + (T52.array() - T58.array()) * T52.array().sign() / T56(0, 0)); // OpHYFixExp
            g_T56(0) += (g_T58(Eigen::all, Eigen::seq(0, 0)).array() * (T58.array() - T52.array()) * (T52.array().abs() + T56(0, 0)) / (T56(0, 0) * T56(0, 0))).sum();       // OpHYFixExp
            g_T56(1) += (g_T58(Eigen::all, Eigen::seq(1, 1)).array() * (T58.array() - T52.array()) * (T52.array().abs() + T56(0, 0)) / (T56(0, 0) * T56(0, 0))).sum();       // OpHYFixExp
            g_T56(2) += (g_T58(Eigen::all, Eigen::seq(2, 2)).array() * (T58.array() - T52.array()) * (T52.array().abs() + T56(0, 0)) / (T56(0, 0) * T56(0, 0))).sum();       // OpHYFixExp
            g_T56(3) += (g_T58(Eigen::all, Eigen::seq(3, 3)).array() * (T58.array() - T52.array()) * (T52.array().abs() + T56(0, 0)) / (T56(0, 0) * T56(0, 0))).sum();       // OpHYFixExp
            g_T56(4) += (g_T58(Eigen::all, Eigen::seq(4, 4)).array() * (T58.array() - T52.array()) * (T52.array().abs() + T56(0, 0)) / (T56(0, 0) * T56(0, 0))).sum();       // OpHYFixExp
            g_T51(Eigen::all, Eigen::seq(0, 0)).array() += g_T52(Eigen::all, Eigen::seq(0, 0)).array() * T51.array().sign();                                                 // OpAbs
            g_T51(Eigen::all, Eigen::seq(1, 1)).array() += g_T52(Eigen::all, Eigen::seq(1, 1)).array() * T51.array().sign();                                                 // OpAbs
            g_T51(Eigen::all, Eigen::seq(2, 2)).array() += g_T52(Eigen::all, Eigen::seq(2, 2)).array() * T51.array().sign();                                                 // OpAbs
            g_T51(Eigen::all, Eigen::seq(3, 3)).array() += g_T52(Eigen::all, Eigen::seq(3, 3)).array() * T51.array().sign();                                                 // OpAbs
            g_T51(Eigen::all, Eigen::seq(4, 4)).array() += g_T52(Eigen::all, Eigen::seq(4, 4)).array() * T51.array().sign();                                                 // OpAbs
            g_T45 += g_T51;                                                                                                                                                  // OpAdd
            g_T44 += g_T51;                                                                                                                                                  // OpAdd
            g_T54(Eigen::all, Eigen::seq(0, 0)).array() += g_T56(Eigen::all, Eigen::seq(0, 0)).array() * T55.array();                                                        // OpCwiseMul
            g_T54(Eigen::all, Eigen::seq(1, 1)).array() += g_T56(Eigen::all, Eigen::seq(1, 1)).array() * T55.array();                                                        // OpCwiseMul
            g_T54(Eigen::all, Eigen::seq(2, 2)).array() += g_T56(Eigen::all, Eigen::seq(2, 2)).array() * T55.array();                                                        // OpCwiseMul
            g_T54(Eigen::all, Eigen::seq(3, 3)).array() += g_T56(Eigen::all, Eigen::seq(3, 3)).array() * T55.array();                                                        // OpCwiseMul
            g_T54(Eigen::all, Eigen::seq(4, 4)).array() += g_T56(Eigen::all, Eigen::seq(4, 4)).array() * T55.array();                                                        // OpCwiseMul
            g_T55(Eigen::all, Eigen::seq(0, 0)).array() += g_T56(Eigen::all, Eigen::seq(0, 0)).array() * T54.array();                                                        // OpCwiseMul
            g_T55(Eigen::all, Eigen::seq(1, 1)).array() += g_T56(Eigen::all, Eigen::seq(1, 1)).array() * T54.array();                                                        // OpCwiseMul
            g_T55(Eigen::all, Eigen::seq(2, 2)).array() += g_T56(Eigen::all, Eigen::seq(2, 2)).array() * T54.array();                                                        // OpCwiseMul
            g_T55(Eigen::all, Eigen::seq(3, 3)).array() += g_T56(Eigen::all, Eigen::seq(3, 3)).array() * T54.array();                                                        // OpCwiseMul
            g_T55(Eigen::all, Eigen::seq(4, 4)).array() += g_T56(Eigen::all, Eigen::seq(4, 4)).array() * T54.array();                                                        // OpCwiseMul
            g_T53 += g_T54;                                                                                                                                                  // OpAdd
            g_T44 += g_T54;                                                                                                                                                  // OpAdd
            Eigen::Matrix<double, 1, 1> T1_T53 = 1 / T53.array();                                                                                                            // OpSqrt
            if (T53(0) == 0)
                T1_T53(0) = 0;                                                                                                 // OpSqrt
            g_T35(Eigen::all, Eigen::seq(0, 0)).array() += g_T53(Eigen::all, Eigen::seq(0, 0)).array() * T1_T53.array() * 0.5; // OpSqrt
            g_T35(Eigen::all, Eigen::seq(1, 1)).array() += g_T53(Eigen::all, Eigen::seq(1, 1)).array() * T1_T53.array() * 0.5; // OpSqrt
            g_T35(Eigen::all, Eigen::seq(2, 2)).array() += g_T53(Eigen::all, Eigen::seq(2, 2)).array() * T1_T53.array() * 0.5; // OpSqrt
            g_T35(Eigen::all, Eigen::seq(3, 3)).array() += g_T53(Eigen::all, Eigen::seq(3, 3)).array() * T1_T53.array() * 0.5; // OpSqrt
            g_T35(Eigen::all, Eigen::seq(4, 4)).array() += g_T53(Eigen::all, Eigen::seq(4, 4)).array() * T1_T53.array() * 0.5; // OpSqrt
            g_T3 += g_T55 * 0.5;                                                                                               // OpTimesConstScalar
            // grad end is at g_T3
            g_T86 += g_T87 * 0.5;                                                                                                   // OpTimesConstScalar
            g_T85(Eigen::all, Eigen::seq(0, 0)).array() += g_T86(Eigen::all, Eigen::seq(0, 0)).array() / T44.array();               // OpCwiseDiv
            g_T85(Eigen::all, Eigen::seq(1, 1)).array() += g_T86(Eigen::all, Eigen::seq(1, 1)).array() / T44.array();               // OpCwiseDiv
            g_T85(Eigen::all, Eigen::seq(2, 2)).array() += g_T86(Eigen::all, Eigen::seq(2, 2)).array() / T44.array();               // OpCwiseDiv
            g_T85(Eigen::all, Eigen::seq(3, 3)).array() += g_T86(Eigen::all, Eigen::seq(3, 3)).array() / T44.array();               // OpCwiseDiv
            g_T85(Eigen::all, Eigen::seq(4, 4)).array() += g_T86(Eigen::all, Eigen::seq(4, 4)).array() / T44.array();               // OpCwiseDiv
            g_T44(Eigen::all, Eigen::seq(0, 0)).array() -= g_T86(Eigen::all, Eigen::seq(0, 0)).array() * T86.array() / T44.array(); // OpCwiseDiv
            g_T44(Eigen::all, Eigen::seq(1, 1)).array() -= g_T86(Eigen::all, Eigen::seq(1, 1)).array() * T86.array() / T44.array(); // OpCwiseDiv
            g_T44(Eigen::all, Eigen::seq(2, 2)).array() -= g_T86(Eigen::all, Eigen::seq(2, 2)).array() * T86.array() / T44.array(); // OpCwiseDiv
            g_T44(Eigen::all, Eigen::seq(3, 3)).array() -= g_T86(Eigen::all, Eigen::seq(3, 3)).array() * T86.array() / T44.array(); // OpCwiseDiv
            g_T44(Eigen::all, Eigen::seq(4, 4)).array() -= g_T86(Eigen::all, Eigen::seq(4, 4)).array() * T86.array() / T44.array(); // OpCwiseDiv
            g_T73 += g_T85;                                                                                                         // OpAdd
            g_T84 += g_T85;                                                                                                         // OpAdd
            g_T83(Eigen::all, Eigen::seq(0, 0)).array() += g_T84(Eigen::all, Eigen::seq(0, 0)).array() * T75.array();               // OpCwiseMul
            g_T83(Eigen::all, Eigen::seq(1, 1)).array() += g_T84(Eigen::all, Eigen::seq(1, 1)).array() * T75.array();               // OpCwiseMul
            g_T83(Eigen::all, Eigen::seq(2, 2)).array() += g_T84(Eigen::all, Eigen::seq(2, 2)).array() * T75.array();               // OpCwiseMul
            g_T83(Eigen::all, Eigen::seq(3, 3)).array() += g_T84(Eigen::all, Eigen::seq(3, 3)).array() * T75.array();               // OpCwiseMul
            g_T83(Eigen::all, Eigen::seq(4, 4)).array() += g_T84(Eigen::all, Eigen::seq(4, 4)).array() * T75.array();               // OpCwiseMul
            g_T75(Eigen::all, Eigen::seq(0, 0)).array() += g_T84(Eigen::all, Eigen::seq(0, 0)).array() * T83.array();               // OpCwiseMul
            g_T75(Eigen::all, Eigen::seq(1, 1)).array() += g_T84(Eigen::all, Eigen::seq(1, 1)).array() * T83.array();               // OpCwiseMul
            g_T75(Eigen::all, Eigen::seq(2, 2)).array() += g_T84(Eigen::all, Eigen::seq(2, 2)).array() * T83.array();               // OpCwiseMul
            g_T75(Eigen::all, Eigen::seq(3, 3)).array() += g_T84(Eigen::all, Eigen::seq(3, 3)).array() * T83.array();               // OpCwiseMul
            g_T75(Eigen::all, Eigen::seq(4, 4)).array() += g_T84(Eigen::all, Eigen::seq(4, 4)).array() * T83.array();               // OpCwiseMul
            g_T74(Eigen::all, Eigen::seq(0, 0))({0}, {0}) += g_T75(Eigen::all, Eigen::seq(0, 0));                                   // OpMatBlock
            g_T74(Eigen::all, Eigen::seq(1, 1))({0}, {0}) += g_T75(Eigen::all, Eigen::seq(1, 1));                                   // OpMatBlock
            g_T74(Eigen::all, Eigen::seq(2, 2))({0}, {0}) += g_T75(Eigen::all, Eigen::seq(2, 2));                                   // OpMatBlock
            g_T74(Eigen::all, Eigen::seq(3, 3))({0}, {0}) += g_T75(Eigen::all, Eigen::seq(3, 3));                                   // OpMatBlock
            g_T74(Eigen::all, Eigen::seq(4, 4))({0}, {0}) += g_T75(Eigen::all, Eigen::seq(4, 4));                                   // OpMatBlock
            g_T43(Eigen::all, Eigen::seq(0, 0)).array() += g_T83(Eigen::all, Eigen::seq(0, 0)).array() * T44.array();               // OpCwiseMul
            g_T43(Eigen::all, Eigen::seq(1, 1)).array() += g_T83(Eigen::all, Eigen::seq(1, 1)).array() * T44.array();               // OpCwiseMul
            g_T43(Eigen::all, Eigen::seq(2, 2)).array() += g_T83(Eigen::all, Eigen::seq(2, 2)).array() * T44.array();               // OpCwiseMul
            g_T43(Eigen::all, Eigen::seq(3, 3)).array() += g_T83(Eigen::all, Eigen::seq(3, 3)).array() * T44.array();               // OpCwiseMul
            g_T43(Eigen::all, Eigen::seq(4, 4)).array() += g_T83(Eigen::all, Eigen::seq(4, 4)).array() * T44.array();               // OpCwiseMul
            g_T44(Eigen::all, Eigen::seq(0, 0)).array() += g_T83(Eigen::all, Eigen::seq(0, 0)).array() * T43.array();               // OpCwiseMul
            g_T44(Eigen::all, Eigen::seq(1, 1)).array() += g_T83(Eigen::all, Eigen::seq(1, 1)).array() * T43.array();               // OpCwiseMul
            g_T44(Eigen::all, Eigen::seq(2, 2)).array() += g_T83(Eigen::all, Eigen::seq(2, 2)).array() * T43.array();               // OpCwiseMul
            g_T44(Eigen::all, Eigen::seq(3, 3)).array() += g_T83(Eigen::all, Eigen::seq(3, 3)).array() * T43.array();               // OpCwiseMul
            g_T44(Eigen::all, Eigen::seq(4, 4)).array() += g_T83(Eigen::all, Eigen::seq(4, 4)).array() * T43.array();               // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T1_T44 = 1 / T44.array();                                                                   // OpSqrt
            if (T44(0) == 0)
                T1_T44(0) = 0;                                                                                                      // OpSqrt
            g_T42(Eigen::all, Eigen::seq(0, 0)).array() += g_T44(Eigen::all, Eigen::seq(0, 0)).array() * T1_T44.array() * 0.5;      // OpSqrt
            g_T42(Eigen::all, Eigen::seq(1, 1)).array() += g_T44(Eigen::all, Eigen::seq(1, 1)).array() * T1_T44.array() * 0.5;      // OpSqrt
            g_T42(Eigen::all, Eigen::seq(2, 2)).array() += g_T44(Eigen::all, Eigen::seq(2, 2)).array() * T1_T44.array() * 0.5;      // OpSqrt
            g_T42(Eigen::all, Eigen::seq(3, 3)).array() += g_T44(Eigen::all, Eigen::seq(3, 3)).array() * T1_T44.array() * 0.5;      // OpSqrt
            g_T42(Eigen::all, Eigen::seq(4, 4)).array() += g_T44(Eigen::all, Eigen::seq(4, 4)).array() * T1_T44.array() * 0.5;      // OpSqrt
            g_T69 += g_T122 * T121(0, 0);                                                                                           // OpTimesScalar
            g_T121(0) += (g_T122(Eigen::all, Eigen::seq(0, 0)).array() * T69.array()).sum();                                        // OpTimesScalar
            g_T121(1) += (g_T122(Eigen::all, Eigen::seq(1, 1)).array() * T69.array()).sum();                                        // OpTimesScalar
            g_T121(2) += (g_T122(Eigen::all, Eigen::seq(2, 2)).array() * T69.array()).sum();                                        // OpTimesScalar
            g_T121(3) += (g_T122(Eigen::all, Eigen::seq(3, 3)).array() * T69.array()).sum();                                        // OpTimesScalar
            g_T121(4) += (g_T122(Eigen::all, Eigen::seq(4, 4)).array() * T69.array()).sum();                                        // OpTimesScalar
            g_T0 += g_T69(Eigen::seq(0, 0), Eigen::all);                                                                            // OpMatConcat
            g_T34 += g_T69(Eigen::seq(1, 3), Eigen::all);                                                                           // OpMatConcat
            g_T68 += g_T69(Eigen::seq(4, 4), Eigen::all);                                                                           // OpMatConcat
            g_T35 += g_T68 * 0.5;                                                                                                   // OpTimesConstScalar
            g_T89(Eigen::all, Eigen::seq(0, 0)).array() += g_T121(Eigen::all, Eigen::seq(0, 0)).array() * T50.array();              // OpCwiseMul
            g_T89(Eigen::all, Eigen::seq(1, 1)).array() += g_T121(Eigen::all, Eigen::seq(1, 1)).array() * T50.array();              // OpCwiseMul
            g_T89(Eigen::all, Eigen::seq(2, 2)).array() += g_T121(Eigen::all, Eigen::seq(2, 2)).array() * T50.array();              // OpCwiseMul
            g_T89(Eigen::all, Eigen::seq(3, 3)).array() += g_T121(Eigen::all, Eigen::seq(3, 3)).array() * T50.array();              // OpCwiseMul
            g_T89(Eigen::all, Eigen::seq(4, 4)).array() += g_T121(Eigen::all, Eigen::seq(4, 4)).array() * T50.array();              // OpCwiseMul
            g_T50(Eigen::all, Eigen::seq(0, 0)).array() += g_T121(Eigen::all, Eigen::seq(0, 0)).array() * T89.array();              // OpCwiseMul
            g_T50(Eigen::all, Eigen::seq(1, 1)).array() += g_T121(Eigen::all, Eigen::seq(1, 1)).array() * T89.array();              // OpCwiseMul
            g_T50(Eigen::all, Eigen::seq(2, 2)).array() += g_T121(Eigen::all, Eigen::seq(2, 2)).array() * T89.array();              // OpCwiseMul
            g_T50(Eigen::all, Eigen::seq(3, 3)).array() += g_T121(Eigen::all, Eigen::seq(3, 3)).array() * T89.array();              // OpCwiseMul
            g_T50(Eigen::all, Eigen::seq(4, 4)).array() += g_T121(Eigen::all, Eigen::seq(4, 4)).array() * T89.array();              // OpCwiseMul
            g_T72 += g_T89;                                                                                                         // OpSubs
            g_T88 -= g_T89;                                                                                                         // OpSubs
            g_T7 += g_T72;                                                                                                          // OpSubs
            g_T6 -= g_T72;                                                                                                          // OpSubs
            g_T73(Eigen::all, Eigen::seq(0, 0)).array() += g_T88(Eigen::all, Eigen::seq(0, 0)).array() / T42.array();               // OpCwiseDiv
            g_T73(Eigen::all, Eigen::seq(1, 1)).array() += g_T88(Eigen::all, Eigen::seq(1, 1)).array() / T42.array();               // OpCwiseDiv
            g_T73(Eigen::all, Eigen::seq(2, 2)).array() += g_T88(Eigen::all, Eigen::seq(2, 2)).array() / T42.array();               // OpCwiseDiv
            g_T73(Eigen::all, Eigen::seq(3, 3)).array() += g_T88(Eigen::all, Eigen::seq(3, 3)).array() / T42.array();               // OpCwiseDiv
            g_T73(Eigen::all, Eigen::seq(4, 4)).array() += g_T88(Eigen::all, Eigen::seq(4, 4)).array() / T42.array();               // OpCwiseDiv
            g_T42(Eigen::all, Eigen::seq(0, 0)).array() -= g_T88(Eigen::all, Eigen::seq(0, 0)).array() * T88.array() / T42.array(); // OpCwiseDiv
            g_T42(Eigen::all, Eigen::seq(1, 1)).array() -= g_T88(Eigen::all, Eigen::seq(1, 1)).array() * T88.array() / T42.array(); // OpCwiseDiv
            g_T42(Eigen::all, Eigen::seq(2, 2)).array() -= g_T88(Eigen::all, Eigen::seq(2, 2)).array() * T88.array() / T42.array(); // OpCwiseDiv
            g_T42(Eigen::all, Eigen::seq(3, 3)).array() -= g_T88(Eigen::all, Eigen::seq(3, 3)).array() * T88.array() / T42.array(); // OpCwiseDiv
            g_T42(Eigen::all, Eigen::seq(4, 4)).array() -= g_T88(Eigen::all, Eigen::seq(4, 4)).array() * T88.array() / T42.array(); // OpCwiseDiv
            g_T41(Eigen::all, Eigen::seq(0, 0)).array() += g_T42(Eigen::all, Eigen::seq(0, 0)).array() * T2.array();                // OpCwiseMul
            g_T41(Eigen::all, Eigen::seq(1, 1)).array() += g_T42(Eigen::all, Eigen::seq(1, 1)).array() * T2.array();                // OpCwiseMul
            g_T41(Eigen::all, Eigen::seq(2, 2)).array() += g_T42(Eigen::all, Eigen::seq(2, 2)).array() * T2.array();                // OpCwiseMul
            g_T41(Eigen::all, Eigen::seq(3, 3)).array() += g_T42(Eigen::all, Eigen::seq(3, 3)).array() * T2.array();                // OpCwiseMul
            g_T41(Eigen::all, Eigen::seq(4, 4)).array() += g_T42(Eigen::all, Eigen::seq(4, 4)).array() * T2.array();                // OpCwiseMul
            g_T2(Eigen::all, Eigen::seq(0, 0)).array() += g_T42(Eigen::all, Eigen::seq(0, 0)).array() * T41.array();                // OpCwiseMul
            g_T2(Eigen::all, Eigen::seq(1, 1)).array() += g_T42(Eigen::all, Eigen::seq(1, 1)).array() * T41.array();                // OpCwiseMul
            g_T2(Eigen::all, Eigen::seq(2, 2)).array() += g_T42(Eigen::all, Eigen::seq(2, 2)).array() * T41.array();                // OpCwiseMul
            g_T2(Eigen::all, Eigen::seq(3, 3)).array() += g_T42(Eigen::all, Eigen::seq(3, 3)).array() * T41.array();                // OpCwiseMul
            g_T2(Eigen::all, Eigen::seq(4, 4)).array() += g_T42(Eigen::all, Eigen::seq(4, 4)).array() * T41.array();                // OpCwiseMul
            g_T39 += g_T41;                                                                                                         // OpSubs
            g_T40 -= g_T41;                                                                                                         // OpSubs
            g_T38(Eigen::all, Eigen::seq(0, 0)).array() += g_T39(Eigen::all, Eigen::seq(0, 0)).array() / T30.array();               // OpCwiseDiv
            g_T38(Eigen::all, Eigen::seq(1, 1)).array() += g_T39(Eigen::all, Eigen::seq(1, 1)).array() / T30.array();               // OpCwiseDiv
            g_T38(Eigen::all, Eigen::seq(2, 2)).array() += g_T39(Eigen::all, Eigen::seq(2, 2)).array() / T30.array();               // OpCwiseDiv
            g_T38(Eigen::all, Eigen::seq(3, 3)).array() += g_T39(Eigen::all, Eigen::seq(3, 3)).array() / T30.array();               // OpCwiseDiv
            g_T38(Eigen::all, Eigen::seq(4, 4)).array() += g_T39(Eigen::all, Eigen::seq(4, 4)).array() / T30.array();               // OpCwiseDiv
            g_T30(Eigen::all, Eigen::seq(0, 0)).array() -= g_T39(Eigen::all, Eigen::seq(0, 0)).array() * T39.array() / T30.array(); // OpCwiseDiv
            g_T30(Eigen::all, Eigen::seq(1, 1)).array() -= g_T39(Eigen::all, Eigen::seq(1, 1)).array() * T39.array() / T30.array(); // OpCwiseDiv
            g_T30(Eigen::all, Eigen::seq(2, 2)).array() -= g_T39(Eigen::all, Eigen::seq(2, 2)).array() * T39.array() / T30.array(); // OpCwiseDiv
            g_T30(Eigen::all, Eigen::seq(3, 3)).array() -= g_T39(Eigen::all, Eigen::seq(3, 3)).array() * T39.array() / T30.array(); // OpCwiseDiv
            g_T30(Eigen::all, Eigen::seq(4, 4)).array() -= g_T39(Eigen::all, Eigen::seq(4, 4)).array() * T39.array() / T30.array(); // OpCwiseDiv
            g_T36 += g_T38;                                                                                                         // OpAdd
            g_T37 += g_T38;                                                                                                         // OpAdd
            g_T25(Eigen::all, Eigen::seq(0, 0)).array() += g_T36(Eigen::all, Eigen::seq(0, 0)).array() * T28.array();               // OpCwiseMul
            g_T25(Eigen::all, Eigen::seq(1, 1)).array() += g_T36(Eigen::all, Eigen::seq(1, 1)).array() * T28.array();               // OpCwiseMul
            g_T25(Eigen::all, Eigen::seq(2, 2)).array() += g_T36(Eigen::all, Eigen::seq(2, 2)).array() * T28.array();               // OpCwiseMul
            g_T25(Eigen::all, Eigen::seq(3, 3)).array() += g_T36(Eigen::all, Eigen::seq(3, 3)).array() * T28.array();               // OpCwiseMul
            g_T25(Eigen::all, Eigen::seq(4, 4)).array() += g_T36(Eigen::all, Eigen::seq(4, 4)).array() * T28.array();               // OpCwiseMul
            g_T28(Eigen::all, Eigen::seq(0, 0)).array() += g_T36(Eigen::all, Eigen::seq(0, 0)).array() * T25.array();               // OpCwiseMul
            g_T28(Eigen::all, Eigen::seq(1, 1)).array() += g_T36(Eigen::all, Eigen::seq(1, 1)).array() * T25.array();               // OpCwiseMul
            g_T28(Eigen::all, Eigen::seq(2, 2)).array() += g_T36(Eigen::all, Eigen::seq(2, 2)).array() * T25.array();               // OpCwiseMul
            g_T28(Eigen::all, Eigen::seq(3, 3)).array() += g_T36(Eigen::all, Eigen::seq(3, 3)).array() * T25.array();               // OpCwiseMul
            g_T28(Eigen::all, Eigen::seq(4, 4)).array() += g_T36(Eigen::all, Eigen::seq(4, 4)).array() * T25.array();               // OpCwiseMul
            g_T24(Eigen::all, Eigen::seq(0, 0)).array() += g_T25(Eigen::all, Eigen::seq(0, 0)).array() / T6.array();                // OpCwiseDiv
            g_T24(Eigen::all, Eigen::seq(1, 1)).array() += g_T25(Eigen::all, Eigen::seq(1, 1)).array() / T6.array();                // OpCwiseDiv
            g_T24(Eigen::all, Eigen::seq(2, 2)).array() += g_T25(Eigen::all, Eigen::seq(2, 2)).array() / T6.array();                // OpCwiseDiv
            g_T24(Eigen::all, Eigen::seq(3, 3)).array() += g_T25(Eigen::all, Eigen::seq(3, 3)).array() / T6.array();                // OpCwiseDiv
            g_T24(Eigen::all, Eigen::seq(4, 4)).array() += g_T25(Eigen::all, Eigen::seq(4, 4)).array() / T6.array();                // OpCwiseDiv
            g_T6(Eigen::all, Eigen::seq(0, 0)).array() -= g_T25(Eigen::all, Eigen::seq(0, 0)).array() * T25.array() / T6.array();   // OpCwiseDiv
            g_T6(Eigen::all, Eigen::seq(1, 1)).array() -= g_T25(Eigen::all, Eigen::seq(1, 1)).array() * T25.array() / T6.array();   // OpCwiseDiv
            g_T6(Eigen::all, Eigen::seq(2, 2)).array() -= g_T25(Eigen::all, Eigen::seq(2, 2)).array() * T25.array() / T6.array();   // OpCwiseDiv
            g_T6(Eigen::all, Eigen::seq(3, 3)).array() -= g_T25(Eigen::all, Eigen::seq(3, 3)).array() * T25.array() / T6.array();   // OpCwiseDiv
            g_T6(Eigen::all, Eigen::seq(4, 4)).array() -= g_T25(Eigen::all, Eigen::seq(4, 4)).array() * T25.array() / T6.array();   // OpCwiseDiv
            g_T14 += g_T24;                                                                                                         // OpAdd
            g_T19 += g_T24;                                                                                                         // OpAdd
            g_T27(Eigen::all, Eigen::seq(0, 0)).array() += g_T37(Eigen::all, Eigen::seq(0, 0)).array() * T29.array();               // OpCwiseMul
            g_T27(Eigen::all, Eigen::seq(1, 1)).array() += g_T37(Eigen::all, Eigen::seq(1, 1)).array() * T29.array();               // OpCwiseMul
            g_T27(Eigen::all, Eigen::seq(2, 2)).array() += g_T37(Eigen::all, Eigen::seq(2, 2)).array() * T29.array();               // OpCwiseMul
            g_T27(Eigen::all, Eigen::seq(3, 3)).array() += g_T37(Eigen::all, Eigen::seq(3, 3)).array() * T29.array();               // OpCwiseMul
            g_T27(Eigen::all, Eigen::seq(4, 4)).array() += g_T37(Eigen::all, Eigen::seq(4, 4)).array() * T29.array();               // OpCwiseMul
            g_T29(Eigen::all, Eigen::seq(0, 0)).array() += g_T37(Eigen::all, Eigen::seq(0, 0)).array() * T27.array();               // OpCwiseMul
            g_T29(Eigen::all, Eigen::seq(1, 1)).array() += g_T37(Eigen::all, Eigen::seq(1, 1)).array() * T27.array();               // OpCwiseMul
            g_T29(Eigen::all, Eigen::seq(2, 2)).array() += g_T37(Eigen::all, Eigen::seq(2, 2)).array() * T27.array();               // OpCwiseMul
            g_T29(Eigen::all, Eigen::seq(3, 3)).array() += g_T37(Eigen::all, Eigen::seq(3, 3)).array() * T27.array();               // OpCwiseMul
            g_T29(Eigen::all, Eigen::seq(4, 4)).array() += g_T37(Eigen::all, Eigen::seq(4, 4)).array() * T27.array();               // OpCwiseMul
            g_T26(Eigen::all, Eigen::seq(0, 0)).array() += g_T27(Eigen::all, Eigen::seq(0, 0)).array() / T7.array();                // OpCwiseDiv
            g_T26(Eigen::all, Eigen::seq(1, 1)).array() += g_T27(Eigen::all, Eigen::seq(1, 1)).array() / T7.array();                // OpCwiseDiv
            g_T26(Eigen::all, Eigen::seq(2, 2)).array() += g_T27(Eigen::all, Eigen::seq(2, 2)).array() / T7.array();                // OpCwiseDiv
            g_T26(Eigen::all, Eigen::seq(3, 3)).array() += g_T27(Eigen::all, Eigen::seq(3, 3)).array() / T7.array();                // OpCwiseDiv
            g_T26(Eigen::all, Eigen::seq(4, 4)).array() += g_T27(Eigen::all, Eigen::seq(4, 4)).array() / T7.array();                // OpCwiseDiv
            g_T7(Eigen::all, Eigen::seq(0, 0)).array() -= g_T27(Eigen::all, Eigen::seq(0, 0)).array() * T27.array() / T7.array();   // OpCwiseDiv
            g_T7(Eigen::all, Eigen::seq(1, 1)).array() -= g_T27(Eigen::all, Eigen::seq(1, 1)).array() * T27.array() / T7.array();   // OpCwiseDiv
            g_T7(Eigen::all, Eigen::seq(2, 2)).array() -= g_T27(Eigen::all, Eigen::seq(2, 2)).array() * T27.array() / T7.array();   // OpCwiseDiv
            g_T7(Eigen::all, Eigen::seq(3, 3)).array() -= g_T27(Eigen::all, Eigen::seq(3, 3)).array() * T27.array() / T7.array();   // OpCwiseDiv
            g_T7(Eigen::all, Eigen::seq(4, 4)).array() -= g_T27(Eigen::all, Eigen::seq(4, 4)).array() * T27.array() / T7.array();   // OpCwiseDiv
            g_T15 += g_T26;                                                                                                         // OpAdd
            g_T23 += g_T26;                                                                                                         // OpAdd
            g_T35 += g_T40 * 0.5;                                                                                                   // OpTimesConstScalar
            g_T34(Eigen::all, Eigen::seq(0, 0)) += g_T35(0) * T34;                                                                  // OpMatDot
            g_T34(Eigen::all, Eigen::seq(1, 1)) += g_T35(1) * T34;                                                                  // OpMatDot
            g_T34(Eigen::all, Eigen::seq(2, 2)) += g_T35(2) * T34;                                                                  // OpMatDot
            g_T34(Eigen::all, Eigen::seq(3, 3)) += g_T35(3) * T34;                                                                  // OpMatDot
            g_T34(Eigen::all, Eigen::seq(4, 4)) += g_T35(4) * T34;                                                                  // OpMatDot
            g_T34(Eigen::all, Eigen::seq(0, 0)) += g_T35(0) * T34;                                                                  // OpMatDot
            g_T34(Eigen::all, Eigen::seq(1, 1)) += g_T35(1) * T34;                                                                  // OpMatDot
            g_T34(Eigen::all, Eigen::seq(2, 2)) += g_T35(2) * T34;                                                                  // OpMatDot
            g_T34(Eigen::all, Eigen::seq(3, 3)) += g_T35(3) * T34;                                                                  // OpMatDot
            g_T34(Eigen::all, Eigen::seq(4, 4)) += g_T35(4) * T34;                                                                  // OpMatDot
            g_T23 += g_T73;                                                                                                         // OpSubs
            g_T19 -= g_T73;                                                                                                         // OpSubs
            g_T70 += g_T125 * T124(0, 0);                                                                                           // OpTimesScalar
            g_T124(0) += (g_T125(Eigen::all, Eigen::seq(0, 0)).array() * T70.array()).sum();                                        // OpTimesScalar
            g_T124(1) += (g_T125(Eigen::all, Eigen::seq(1, 1)).array() * T70.array()).sum();                                        // OpTimesScalar
            g_T124(2) += (g_T125(Eigen::all, Eigen::seq(2, 2)).array() * T70.array()).sum();                                        // OpTimesScalar
            g_T124(3) += (g_T125(Eigen::all, Eigen::seq(3, 3)).array() * T70.array()).sum();                                        // OpTimesScalar
            g_T124(4) += (g_T125(Eigen::all, Eigen::seq(4, 4)).array() * T70.array()).sum();                                        // OpTimesScalar
            g_T1 += g_T70(Eigen::seq(0, 0), Eigen::all);                                                                            // OpMatConcat
            g_T1 += g_T70(Eigen::seq(1, 1), Eigen::all);                                                                            // OpMatConcat
            g_T0 += g_T70(Eigen::seq(2, 2), Eigen::all);                                                                            // OpMatConcat
            g_T1 += g_T70(Eigen::seq(3, 3), Eigen::all);                                                                            // OpMatConcat
            g_T46 += g_T70(Eigen::seq(4, 4), Eigen::all);                                                                           // OpMatConcat
            g_T34(Eigen::all, Eigen::seq(0, 0))({1}, {0}) += g_T46(Eigen::all, Eigen::seq(0, 0));                                   // OpMatBlock
            g_T34(Eigen::all, Eigen::seq(1, 1))({1}, {0}) += g_T46(Eigen::all, Eigen::seq(1, 1));                                   // OpMatBlock
            g_T34(Eigen::all, Eigen::seq(2, 2))({1}, {0}) += g_T46(Eigen::all, Eigen::seq(2, 2));                                   // OpMatBlock
            g_T34(Eigen::all, Eigen::seq(3, 3))({1}, {0}) += g_T46(Eigen::all, Eigen::seq(3, 3));                                   // OpMatBlock
            g_T34(Eigen::all, Eigen::seq(4, 4))({1}, {0}) += g_T46(Eigen::all, Eigen::seq(4, 4));                                   // OpMatBlock
            g_T90(Eigen::all, Eigen::seq(0, 0)).array() += g_T124(Eigen::all, Eigen::seq(0, 0)).array() * T50.array();              // OpCwiseMul
            g_T90(Eigen::all, Eigen::seq(1, 1)).array() += g_T124(Eigen::all, Eigen::seq(1, 1)).array() * T50.array();              // OpCwiseMul
            g_T90(Eigen::all, Eigen::seq(2, 2)).array() += g_T124(Eigen::all, Eigen::seq(2, 2)).array() * T50.array();              // OpCwiseMul
            g_T90(Eigen::all, Eigen::seq(3, 3)).array() += g_T124(Eigen::all, Eigen::seq(3, 3)).array() * T50.array();              // OpCwiseMul
            g_T90(Eigen::all, Eigen::seq(4, 4)).array() += g_T124(Eigen::all, Eigen::seq(4, 4)).array() * T50.array();              // OpCwiseMul
            g_T50(Eigen::all, Eigen::seq(0, 0)).array() += g_T124(Eigen::all, Eigen::seq(0, 0)).array() * T90.array();              // OpCwiseMul
            g_T50(Eigen::all, Eigen::seq(1, 1)).array() += g_T124(Eigen::all, Eigen::seq(1, 1)).array() * T90.array();              // OpCwiseMul
            g_T50(Eigen::all, Eigen::seq(2, 2)).array() += g_T124(Eigen::all, Eigen::seq(2, 2)).array() * T90.array();              // OpCwiseMul
            g_T50(Eigen::all, Eigen::seq(3, 3)).array() += g_T124(Eigen::all, Eigen::seq(3, 3)).array() * T90.array();              // OpCwiseMul
            g_T50(Eigen::all, Eigen::seq(4, 4)).array() += g_T124(Eigen::all, Eigen::seq(4, 4)).array() * T90.array();              // OpCwiseMul
            g_T43(Eigen::all, Eigen::seq(0, 0)).array() += g_T90(Eigen::all, Eigen::seq(0, 0)).array() * T76.array();               // OpCwiseMul
            g_T43(Eigen::all, Eigen::seq(1, 1)).array() += g_T90(Eigen::all, Eigen::seq(1, 1)).array() * T76.array();               // OpCwiseMul
            g_T43(Eigen::all, Eigen::seq(2, 2)).array() += g_T90(Eigen::all, Eigen::seq(2, 2)).array() * T76.array();               // OpCwiseMul
            g_T43(Eigen::all, Eigen::seq(3, 3)).array() += g_T90(Eigen::all, Eigen::seq(3, 3)).array() * T76.array();               // OpCwiseMul
            g_T43(Eigen::all, Eigen::seq(4, 4)).array() += g_T90(Eigen::all, Eigen::seq(4, 4)).array() * T76.array();               // OpCwiseMul
            g_T76(Eigen::all, Eigen::seq(0, 0)).array() += g_T90(Eigen::all, Eigen::seq(0, 0)).array() * T43.array();               // OpCwiseMul
            g_T76(Eigen::all, Eigen::seq(1, 1)).array() += g_T90(Eigen::all, Eigen::seq(1, 1)).array() * T43.array();               // OpCwiseMul
            g_T76(Eigen::all, Eigen::seq(2, 2)).array() += g_T90(Eigen::all, Eigen::seq(2, 2)).array() * T43.array();               // OpCwiseMul
            g_T76(Eigen::all, Eigen::seq(3, 3)).array() += g_T90(Eigen::all, Eigen::seq(3, 3)).array() * T43.array();               // OpCwiseMul
            g_T76(Eigen::all, Eigen::seq(4, 4)).array() += g_T90(Eigen::all, Eigen::seq(4, 4)).array() * T43.array();               // OpCwiseMul
            g_T74(Eigen::all, Eigen::seq(0, 0))({1}, {0}) += g_T76(Eigen::all, Eigen::seq(0, 0));                                   // OpMatBlock
            g_T74(Eigen::all, Eigen::seq(1, 1))({1}, {0}) += g_T76(Eigen::all, Eigen::seq(1, 1));                                   // OpMatBlock
            g_T74(Eigen::all, Eigen::seq(2, 2))({1}, {0}) += g_T76(Eigen::all, Eigen::seq(2, 2));                                   // OpMatBlock
            g_T74(Eigen::all, Eigen::seq(3, 3))({1}, {0}) += g_T76(Eigen::all, Eigen::seq(3, 3));                                   // OpMatBlock
            g_T74(Eigen::all, Eigen::seq(4, 4))({1}, {0}) += g_T76(Eigen::all, Eigen::seq(4, 4));                                   // OpMatBlock
            g_T71 += g_T128 * T127(0, 0);                                                                                           // OpTimesScalar
            g_T127(0) += (g_T128(Eigen::all, Eigen::seq(0, 0)).array() * T71.array()).sum();                                        // OpTimesScalar
            g_T127(1) += (g_T128(Eigen::all, Eigen::seq(1, 1)).array() * T71.array()).sum();                                        // OpTimesScalar
            g_T127(2) += (g_T128(Eigen::all, Eigen::seq(2, 2)).array() * T71.array()).sum();                                        // OpTimesScalar
            g_T127(3) += (g_T128(Eigen::all, Eigen::seq(3, 3)).array() * T71.array()).sum();                                        // OpTimesScalar
            g_T127(4) += (g_T128(Eigen::all, Eigen::seq(4, 4)).array() * T71.array()).sum();                                        // OpTimesScalar
            g_T1 += g_T71(Eigen::seq(0, 0), Eigen::all);                                                                            // OpMatConcat
            g_T1 += g_T71(Eigen::seq(1, 1), Eigen::all);                                                                            // OpMatConcat
            g_T1 += g_T71(Eigen::seq(2, 2), Eigen::all);                                                                            // OpMatConcat
            g_T0 += g_T71(Eigen::seq(3, 3), Eigen::all);                                                                            // OpMatConcat
            g_T47 += g_T71(Eigen::seq(4, 4), Eigen::all);                                                                           // OpMatConcat
            // grad end is at g_T0
            // grad end is at g_T1
            g_T34(Eigen::all, Eigen::seq(0, 0))({2}, {0}) += g_T47(Eigen::all, Eigen::seq(0, 0));                             // OpMatBlock
            g_T34(Eigen::all, Eigen::seq(1, 1))({2}, {0}) += g_T47(Eigen::all, Eigen::seq(1, 1));                             // OpMatBlock
            g_T34(Eigen::all, Eigen::seq(2, 2))({2}, {0}) += g_T47(Eigen::all, Eigen::seq(2, 2));                             // OpMatBlock
            g_T34(Eigen::all, Eigen::seq(3, 3))({2}, {0}) += g_T47(Eigen::all, Eigen::seq(3, 3));                             // OpMatBlock
            g_T34(Eigen::all, Eigen::seq(4, 4))({2}, {0}) += g_T47(Eigen::all, Eigen::seq(4, 4));                             // OpMatBlock
            g_T91(Eigen::all, Eigen::seq(0, 0)).array() += g_T127(Eigen::all, Eigen::seq(0, 0)).array() * T50.array();        // OpCwiseMul
            g_T91(Eigen::all, Eigen::seq(1, 1)).array() += g_T127(Eigen::all, Eigen::seq(1, 1)).array() * T50.array();        // OpCwiseMul
            g_T91(Eigen::all, Eigen::seq(2, 2)).array() += g_T127(Eigen::all, Eigen::seq(2, 2)).array() * T50.array();        // OpCwiseMul
            g_T91(Eigen::all, Eigen::seq(3, 3)).array() += g_T127(Eigen::all, Eigen::seq(3, 3)).array() * T50.array();        // OpCwiseMul
            g_T91(Eigen::all, Eigen::seq(4, 4)).array() += g_T127(Eigen::all, Eigen::seq(4, 4)).array() * T50.array();        // OpCwiseMul
            g_T50(Eigen::all, Eigen::seq(0, 0)).array() += g_T127(Eigen::all, Eigen::seq(0, 0)).array() * T91.array();        // OpCwiseMul
            g_T50(Eigen::all, Eigen::seq(1, 1)).array() += g_T127(Eigen::all, Eigen::seq(1, 1)).array() * T91.array();        // OpCwiseMul
            g_T50(Eigen::all, Eigen::seq(2, 2)).array() += g_T127(Eigen::all, Eigen::seq(2, 2)).array() * T91.array();        // OpCwiseMul
            g_T50(Eigen::all, Eigen::seq(3, 3)).array() += g_T127(Eigen::all, Eigen::seq(3, 3)).array() * T91.array();        // OpCwiseMul
            g_T50(Eigen::all, Eigen::seq(4, 4)).array() += g_T127(Eigen::all, Eigen::seq(4, 4)).array() * T91.array();        // OpCwiseMul
            g_T45(Eigen::all, Eigen::seq(0, 0)).array() += g_T50(Eigen::all, Eigen::seq(0, 0)).array() * T45.array().sign();  // OpAbs
            g_T45(Eigen::all, Eigen::seq(1, 1)).array() += g_T50(Eigen::all, Eigen::seq(1, 1)).array() * T45.array().sign();  // OpAbs
            g_T45(Eigen::all, Eigen::seq(2, 2)).array() += g_T50(Eigen::all, Eigen::seq(2, 2)).array() * T45.array().sign();  // OpAbs
            g_T45(Eigen::all, Eigen::seq(3, 3)).array() += g_T50(Eigen::all, Eigen::seq(3, 3)).array() * T45.array().sign();  // OpAbs
            g_T45(Eigen::all, Eigen::seq(4, 4)).array() += g_T50(Eigen::all, Eigen::seq(4, 4)).array() * T45.array().sign();  // OpAbs
            g_T34(Eigen::all, Eigen::seq(0, 0))({0}, {0}) += g_T45(Eigen::all, Eigen::seq(0, 0));                             // OpMatBlock
            g_T34(Eigen::all, Eigen::seq(1, 1))({0}, {0}) += g_T45(Eigen::all, Eigen::seq(1, 1));                             // OpMatBlock
            g_T34(Eigen::all, Eigen::seq(2, 2))({0}, {0}) += g_T45(Eigen::all, Eigen::seq(2, 2));                             // OpMatBlock
            g_T34(Eigen::all, Eigen::seq(3, 3))({0}, {0}) += g_T45(Eigen::all, Eigen::seq(3, 3));                             // OpMatBlock
            g_T34(Eigen::all, Eigen::seq(4, 4))({0}, {0}) += g_T45(Eigen::all, Eigen::seq(4, 4));                             // OpMatBlock
            g_T33 += g_T34 / T30(0, 0);                                                                                       // OpDivideScalar
            g_T30(0) += (g_T34(Eigen::all, Eigen::seq(0, 0)).array() * T33.array()).sum() * (-1.0 / (T30(0, 0) * T30(0, 0))); // OpDivideScalar
            g_T30(1) += (g_T34(Eigen::all, Eigen::seq(1, 1)).array() * T33.array()).sum() * (-1.0 / (T30(0, 0) * T30(0, 0))); // OpDivideScalar
            g_T30(2) += (g_T34(Eigen::all, Eigen::seq(2, 2)).array() * T33.array()).sum() * (-1.0 / (T30(0, 0) * T30(0, 0))); // OpDivideScalar
            g_T30(3) += (g_T34(Eigen::all, Eigen::seq(3, 3)).array() * T33.array()).sum() * (-1.0 / (T30(0, 0) * T30(0, 0))); // OpDivideScalar
            g_T30(4) += (g_T34(Eigen::all, Eigen::seq(4, 4)).array() * T33.array()).sum() * (-1.0 / (T30(0, 0) * T30(0, 0))); // OpDivideScalar
            g_T28 += g_T30;                                                                                                   // OpAdd
            g_T29 += g_T30;                                                                                                   // OpAdd
            g_T31 += g_T33;                                                                                                   // OpAdd
            g_T32 += g_T33;                                                                                                   // OpAdd
            g_T9 += g_T31 * T28(0, 0);                                                                                        // OpTimesScalar
            g_T28(0) += (g_T31(Eigen::all, Eigen::seq(0, 0)).array() * T9.array()).sum();                                     // OpTimesScalar
            g_T28(1) += (g_T31(Eigen::all, Eigen::seq(1, 1)).array() * T9.array()).sum();                                     // OpTimesScalar
            g_T28(2) += (g_T31(Eigen::all, Eigen::seq(2, 2)).array() * T9.array()).sum();                                     // OpTimesScalar
            g_T28(3) += (g_T31(Eigen::all, Eigen::seq(3, 3)).array() * T9.array()).sum();                                     // OpTimesScalar
            g_T28(4) += (g_T31(Eigen::all, Eigen::seq(4, 4)).array() * T9.array()).sum();                                     // OpTimesScalar
            g_T11 += g_T32 * T29(0, 0);                                                                                       // OpTimesScalar
            g_T29(0) += (g_T32(Eigen::all, Eigen::seq(0, 0)).array() * T11.array()).sum();                                    // OpTimesScalar
            g_T29(1) += (g_T32(Eigen::all, Eigen::seq(1, 1)).array() * T11.array()).sum();                                    // OpTimesScalar
            g_T29(2) += (g_T32(Eigen::all, Eigen::seq(2, 2)).array() * T11.array()).sum();                                    // OpTimesScalar
            g_T29(3) += (g_T32(Eigen::all, Eigen::seq(3, 3)).array() * T11.array()).sum();                                    // OpTimesScalar
            g_T29(4) += (g_T32(Eigen::all, Eigen::seq(4, 4)).array() * T11.array()).sum();                                    // OpTimesScalar
            g_T43(Eigen::all, Eigen::seq(0, 0)).array() += g_T91(Eigen::all, Eigen::seq(0, 0)).array() * T77.array();         // OpCwiseMul
            g_T43(Eigen::all, Eigen::seq(1, 1)).array() += g_T91(Eigen::all, Eigen::seq(1, 1)).array() * T77.array();         // OpCwiseMul
            g_T43(Eigen::all, Eigen::seq(2, 2)).array() += g_T91(Eigen::all, Eigen::seq(2, 2)).array() * T77.array();         // OpCwiseMul
            g_T43(Eigen::all, Eigen::seq(3, 3)).array() += g_T91(Eigen::all, Eigen::seq(3, 3)).array() * T77.array();         // OpCwiseMul
            g_T43(Eigen::all, Eigen::seq(4, 4)).array() += g_T91(Eigen::all, Eigen::seq(4, 4)).array() * T77.array();         // OpCwiseMul
            g_T77(Eigen::all, Eigen::seq(0, 0)).array() += g_T91(Eigen::all, Eigen::seq(0, 0)).array() * T43.array();         // OpCwiseMul
            g_T77(Eigen::all, Eigen::seq(1, 1)).array() += g_T91(Eigen::all, Eigen::seq(1, 1)).array() * T43.array();         // OpCwiseMul
            g_T77(Eigen::all, Eigen::seq(2, 2)).array() += g_T91(Eigen::all, Eigen::seq(2, 2)).array() * T43.array();         // OpCwiseMul
            g_T77(Eigen::all, Eigen::seq(3, 3)).array() += g_T91(Eigen::all, Eigen::seq(3, 3)).array() * T43.array();         // OpCwiseMul
            g_T77(Eigen::all, Eigen::seq(4, 4)).array() += g_T91(Eigen::all, Eigen::seq(4, 4)).array() * T43.array();         // OpCwiseMul
            g_T28(Eigen::all, Eigen::seq(0, 0)).array() += g_T43(Eigen::all, Eigen::seq(0, 0)).array() * T29.array();         // OpCwiseMul
            g_T28(Eigen::all, Eigen::seq(1, 1)).array() += g_T43(Eigen::all, Eigen::seq(1, 1)).array() * T29.array();         // OpCwiseMul
            g_T28(Eigen::all, Eigen::seq(2, 2)).array() += g_T43(Eigen::all, Eigen::seq(2, 2)).array() * T29.array();         // OpCwiseMul
            g_T28(Eigen::all, Eigen::seq(3, 3)).array() += g_T43(Eigen::all, Eigen::seq(3, 3)).array() * T29.array();         // OpCwiseMul
            g_T28(Eigen::all, Eigen::seq(4, 4)).array() += g_T43(Eigen::all, Eigen::seq(4, 4)).array() * T29.array();         // OpCwiseMul
            g_T29(Eigen::all, Eigen::seq(0, 0)).array() += g_T43(Eigen::all, Eigen::seq(0, 0)).array() * T28.array();         // OpCwiseMul
            g_T29(Eigen::all, Eigen::seq(1, 1)).array() += g_T43(Eigen::all, Eigen::seq(1, 1)).array() * T28.array();         // OpCwiseMul
            g_T29(Eigen::all, Eigen::seq(2, 2)).array() += g_T43(Eigen::all, Eigen::seq(2, 2)).array() * T28.array();         // OpCwiseMul
            g_T29(Eigen::all, Eigen::seq(3, 3)).array() += g_T43(Eigen::all, Eigen::seq(3, 3)).array() * T28.array();         // OpCwiseMul
            g_T29(Eigen::all, Eigen::seq(4, 4)).array() += g_T43(Eigen::all, Eigen::seq(4, 4)).array() * T28.array();         // OpCwiseMul
            Eigen::Matrix<double, 1, 1> T1_T28 = 1 / T28.array();                                                             // OpSqrt
            if (T28(0) == 0)
                T1_T28(0) = 0;                                                                                                // OpSqrt
            g_T6(Eigen::all, Eigen::seq(0, 0)).array() += g_T28(Eigen::all, Eigen::seq(0, 0)).array() * T1_T28.array() * 0.5; // OpSqrt
            g_T6(Eigen::all, Eigen::seq(1, 1)).array() += g_T28(Eigen::all, Eigen::seq(1, 1)).array() * T1_T28.array() * 0.5; // OpSqrt
            g_T6(Eigen::all, Eigen::seq(2, 2)).array() += g_T28(Eigen::all, Eigen::seq(2, 2)).array() * T1_T28.array() * 0.5; // OpSqrt
            g_T6(Eigen::all, Eigen::seq(3, 3)).array() += g_T28(Eigen::all, Eigen::seq(3, 3)).array() * T1_T28.array() * 0.5; // OpSqrt
            g_T6(Eigen::all, Eigen::seq(4, 4)).array() += g_T28(Eigen::all, Eigen::seq(4, 4)).array() * T1_T28.array() * 0.5; // OpSqrt
            Eigen::Matrix<double, 1, 1> T1_T29 = 1 / T29.array();                                                             // OpSqrt
            if (T29(0) == 0)
                T1_T29(0) = 0;                                                                                                // OpSqrt
            g_T7(Eigen::all, Eigen::seq(0, 0)).array() += g_T29(Eigen::all, Eigen::seq(0, 0)).array() * T1_T29.array() * 0.5; // OpSqrt
            g_T7(Eigen::all, Eigen::seq(1, 1)).array() += g_T29(Eigen::all, Eigen::seq(1, 1)).array() * T1_T29.array() * 0.5; // OpSqrt
            g_T7(Eigen::all, Eigen::seq(2, 2)).array() += g_T29(Eigen::all, Eigen::seq(2, 2)).array() * T1_T29.array() * 0.5; // OpSqrt
            g_T7(Eigen::all, Eigen::seq(3, 3)).array() += g_T29(Eigen::all, Eigen::seq(3, 3)).array() * T1_T29.array() * 0.5; // OpSqrt
            g_T7(Eigen::all, Eigen::seq(4, 4)).array() += g_T29(Eigen::all, Eigen::seq(4, 4)).array() * T1_T29.array() * 0.5; // OpSqrt
            g_T74(Eigen::all, Eigen::seq(0, 0))({2}, {0}) += g_T77(Eigen::all, Eigen::seq(0, 0));                             // OpMatBlock
            g_T74(Eigen::all, Eigen::seq(1, 1))({2}, {0}) += g_T77(Eigen::all, Eigen::seq(1, 1));                             // OpMatBlock
            g_T74(Eigen::all, Eigen::seq(2, 2))({2}, {0}) += g_T77(Eigen::all, Eigen::seq(2, 2));                             // OpMatBlock
            g_T74(Eigen::all, Eigen::seq(3, 3))({2}, {0}) += g_T77(Eigen::all, Eigen::seq(3, 3));                             // OpMatBlock
            g_T74(Eigen::all, Eigen::seq(4, 4))({2}, {0}) += g_T77(Eigen::all, Eigen::seq(4, 4));                             // OpMatBlock
            g_T11 += g_T74;                                                                                                   // OpSubs
            g_T9 -= g_T74;                                                                                                    // OpSubs
            g_T104 += g_T130;                                                                                                 // OpAdd
            g_T115 += g_T130;                                                                                                 // OpAdd
            g_T94 += g_T104(Eigen::seq(0, 0), Eigen::all);                                                                    // OpMatConcat
            g_T97 += g_T104(Eigen::seq(1, 1), Eigen::all);                                                                    // OpMatConcat
            g_T99 += g_T104(Eigen::seq(2, 2), Eigen::all);                                                                    // OpMatConcat
            g_T101 += g_T104(Eigen::seq(3, 3), Eigen::all);                                                                   // OpMatConcat
            g_T103 += g_T104(Eigen::seq(4, 4), Eigen::all);                                                                   // OpMatConcat
            g_T4(Eigen::all, Eigen::seq(0, 0))({1}, {0}) += g_T94(Eigen::all, Eigen::seq(0, 0));                              // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(1, 1))({1}, {0}) += g_T94(Eigen::all, Eigen::seq(1, 1));                              // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(2, 2))({1}, {0}) += g_T94(Eigen::all, Eigen::seq(2, 2));                              // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(3, 3))({1}, {0}) += g_T94(Eigen::all, Eigen::seq(3, 3));                              // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(4, 4))({1}, {0}) += g_T94(Eigen::all, Eigen::seq(4, 4));                              // OpMatBlock
            g_T96 += g_T97;                                                                                                   // OpAdd
            g_T19 += g_T97;                                                                                                   // OpAdd
            g_T95(Eigen::all, Eigen::seq(0, 0)).array() += g_T96(Eigen::all, Eigen::seq(0, 0)).array() * T92.array();         // OpCwiseMul
            g_T95(Eigen::all, Eigen::seq(1, 1)).array() += g_T96(Eigen::all, Eigen::seq(1, 1)).array() * T92.array();         // OpCwiseMul
            g_T95(Eigen::all, Eigen::seq(2, 2)).array() += g_T96(Eigen::all, Eigen::seq(2, 2)).array() * T92.array();         // OpCwiseMul
            g_T95(Eigen::all, Eigen::seq(3, 3)).array() += g_T96(Eigen::all, Eigen::seq(3, 3)).array() * T92.array();         // OpCwiseMul
            g_T95(Eigen::all, Eigen::seq(4, 4)).array() += g_T96(Eigen::all, Eigen::seq(4, 4)).array() * T92.array();         // OpCwiseMul
            g_T92(Eigen::all, Eigen::seq(0, 0)).array() += g_T96(Eigen::all, Eigen::seq(0, 0)).array() * T95.array();         // OpCwiseMul
            g_T92(Eigen::all, Eigen::seq(1, 1)).array() += g_T96(Eigen::all, Eigen::seq(1, 1)).array() * T95.array();         // OpCwiseMul
            g_T92(Eigen::all, Eigen::seq(2, 2)).array() += g_T96(Eigen::all, Eigen::seq(2, 2)).array() * T95.array();         // OpCwiseMul
            g_T92(Eigen::all, Eigen::seq(3, 3)).array() += g_T96(Eigen::all, Eigen::seq(3, 3)).array() * T95.array();         // OpCwiseMul
            g_T92(Eigen::all, Eigen::seq(4, 4)).array() += g_T96(Eigen::all, Eigen::seq(4, 4)).array() * T95.array();         // OpCwiseMul
            g_T4(Eigen::all, Eigen::seq(0, 0))({1}, {0}) += g_T95(Eigen::all, Eigen::seq(0, 0));                              // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(1, 1))({1}, {0}) += g_T95(Eigen::all, Eigen::seq(1, 1));                              // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(2, 2))({1}, {0}) += g_T95(Eigen::all, Eigen::seq(2, 2));                              // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(3, 3))({1}, {0}) += g_T95(Eigen::all, Eigen::seq(3, 3));                              // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(4, 4))({1}, {0}) += g_T95(Eigen::all, Eigen::seq(4, 4));                              // OpMatBlock
            g_T98(Eigen::all, Eigen::seq(0, 0)).array() += g_T99(Eigen::all, Eigen::seq(0, 0)).array() * T92.array();         // OpCwiseMul
            g_T98(Eigen::all, Eigen::seq(1, 1)).array() += g_T99(Eigen::all, Eigen::seq(1, 1)).array() * T92.array();         // OpCwiseMul
            g_T98(Eigen::all, Eigen::seq(2, 2)).array() += g_T99(Eigen::all, Eigen::seq(2, 2)).array() * T92.array();         // OpCwiseMul
            g_T98(Eigen::all, Eigen::seq(3, 3)).array() += g_T99(Eigen::all, Eigen::seq(3, 3)).array() * T92.array();         // OpCwiseMul
            g_T98(Eigen::all, Eigen::seq(4, 4)).array() += g_T99(Eigen::all, Eigen::seq(4, 4)).array() * T92.array();         // OpCwiseMul
            g_T92(Eigen::all, Eigen::seq(0, 0)).array() += g_T99(Eigen::all, Eigen::seq(0, 0)).array() * T98.array();         // OpCwiseMul
            g_T92(Eigen::all, Eigen::seq(1, 1)).array() += g_T99(Eigen::all, Eigen::seq(1, 1)).array() * T98.array();         // OpCwiseMul
            g_T92(Eigen::all, Eigen::seq(2, 2)).array() += g_T99(Eigen::all, Eigen::seq(2, 2)).array() * T98.array();         // OpCwiseMul
            g_T92(Eigen::all, Eigen::seq(3, 3)).array() += g_T99(Eigen::all, Eigen::seq(3, 3)).array() * T98.array();         // OpCwiseMul
            g_T92(Eigen::all, Eigen::seq(4, 4)).array() += g_T99(Eigen::all, Eigen::seq(4, 4)).array() * T98.array();         // OpCwiseMul
            g_T4(Eigen::all, Eigen::seq(0, 0))({2}, {0}) += g_T98(Eigen::all, Eigen::seq(0, 0));                              // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(1, 1))({2}, {0}) += g_T98(Eigen::all, Eigen::seq(1, 1));                              // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(2, 2))({2}, {0}) += g_T98(Eigen::all, Eigen::seq(2, 2));                              // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(3, 3))({2}, {0}) += g_T98(Eigen::all, Eigen::seq(3, 3));                              // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(4, 4))({2}, {0}) += g_T98(Eigen::all, Eigen::seq(4, 4));                              // OpMatBlock
            g_T100(Eigen::all, Eigen::seq(0, 0)).array() += g_T101(Eigen::all, Eigen::seq(0, 0)).array() * T92.array();       // OpCwiseMul
            g_T100(Eigen::all, Eigen::seq(1, 1)).array() += g_T101(Eigen::all, Eigen::seq(1, 1)).array() * T92.array();       // OpCwiseMul
            g_T100(Eigen::all, Eigen::seq(2, 2)).array() += g_T101(Eigen::all, Eigen::seq(2, 2)).array() * T92.array();       // OpCwiseMul
            g_T100(Eigen::all, Eigen::seq(3, 3)).array() += g_T101(Eigen::all, Eigen::seq(3, 3)).array() * T92.array();       // OpCwiseMul
            g_T100(Eigen::all, Eigen::seq(4, 4)).array() += g_T101(Eigen::all, Eigen::seq(4, 4)).array() * T92.array();       // OpCwiseMul
            g_T92(Eigen::all, Eigen::seq(0, 0)).array() += g_T101(Eigen::all, Eigen::seq(0, 0)).array() * T100.array();       // OpCwiseMul
            g_T92(Eigen::all, Eigen::seq(1, 1)).array() += g_T101(Eigen::all, Eigen::seq(1, 1)).array() * T100.array();       // OpCwiseMul
            g_T92(Eigen::all, Eigen::seq(2, 2)).array() += g_T101(Eigen::all, Eigen::seq(2, 2)).array() * T100.array();       // OpCwiseMul
            g_T92(Eigen::all, Eigen::seq(3, 3)).array() += g_T101(Eigen::all, Eigen::seq(3, 3)).array() * T100.array();       // OpCwiseMul
            g_T92(Eigen::all, Eigen::seq(4, 4)).array() += g_T101(Eigen::all, Eigen::seq(4, 4)).array() * T100.array();       // OpCwiseMul
            g_T4(Eigen::all, Eigen::seq(0, 0))({3}, {0}) += g_T100(Eigen::all, Eigen::seq(0, 0));                             // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(1, 1))({3}, {0}) += g_T100(Eigen::all, Eigen::seq(1, 1));                             // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(2, 2))({3}, {0}) += g_T100(Eigen::all, Eigen::seq(2, 2));                             // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(3, 3))({3}, {0}) += g_T100(Eigen::all, Eigen::seq(3, 3));                             // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(4, 4))({3}, {0}) += g_T100(Eigen::all, Eigen::seq(4, 4));                             // OpMatBlock
            g_T102(Eigen::all, Eigen::seq(0, 0)).array() += g_T103(Eigen::all, Eigen::seq(0, 0)).array() * T92.array();       // OpCwiseMul
            g_T102(Eigen::all, Eigen::seq(1, 1)).array() += g_T103(Eigen::all, Eigen::seq(1, 1)).array() * T92.array();       // OpCwiseMul
            g_T102(Eigen::all, Eigen::seq(2, 2)).array() += g_T103(Eigen::all, Eigen::seq(2, 2)).array() * T92.array();       // OpCwiseMul
            g_T102(Eigen::all, Eigen::seq(3, 3)).array() += g_T103(Eigen::all, Eigen::seq(3, 3)).array() * T92.array();       // OpCwiseMul
            g_T102(Eigen::all, Eigen::seq(4, 4)).array() += g_T103(Eigen::all, Eigen::seq(4, 4)).array() * T92.array();       // OpCwiseMul
            g_T92(Eigen::all, Eigen::seq(0, 0)).array() += g_T103(Eigen::all, Eigen::seq(0, 0)).array() * T102.array();       // OpCwiseMul
            g_T92(Eigen::all, Eigen::seq(1, 1)).array() += g_T103(Eigen::all, Eigen::seq(1, 1)).array() * T102.array();       // OpCwiseMul
            g_T92(Eigen::all, Eigen::seq(2, 2)).array() += g_T103(Eigen::all, Eigen::seq(2, 2)).array() * T102.array();       // OpCwiseMul
            g_T92(Eigen::all, Eigen::seq(3, 3)).array() += g_T103(Eigen::all, Eigen::seq(3, 3)).array() * T102.array();       // OpCwiseMul
            g_T92(Eigen::all, Eigen::seq(4, 4)).array() += g_T103(Eigen::all, Eigen::seq(4, 4)).array() * T102.array();       // OpCwiseMul
            g_T9(Eigen::all, Eigen::seq(0, 0))({0}, {0}) += g_T92(Eigen::all, Eigen::seq(0, 0));                              // OpMatBlock
            g_T9(Eigen::all, Eigen::seq(1, 1))({0}, {0}) += g_T92(Eigen::all, Eigen::seq(1, 1));                              // OpMatBlock
            g_T9(Eigen::all, Eigen::seq(2, 2))({0}, {0}) += g_T92(Eigen::all, Eigen::seq(2, 2));                              // OpMatBlock
            g_T9(Eigen::all, Eigen::seq(3, 3))({0}, {0}) += g_T92(Eigen::all, Eigen::seq(3, 3));                              // OpMatBlock
            g_T9(Eigen::all, Eigen::seq(4, 4))({0}, {0}) += g_T92(Eigen::all, Eigen::seq(4, 4));                              // OpMatBlock
            g_T14 += g_T102;                                                                                                  // OpAdd
            g_T19 += g_T102;                                                                                                  // OpAdd
            g_T18(Eigen::all, Eigen::seq(0, 0)).array() += g_T19(Eigen::all, Eigen::seq(0, 0)).array() * T2.array();          // OpCwiseMul
            g_T18(Eigen::all, Eigen::seq(1, 1)).array() += g_T19(Eigen::all, Eigen::seq(1, 1)).array() * T2.array();          // OpCwiseMul
            g_T18(Eigen::all, Eigen::seq(2, 2)).array() += g_T19(Eigen::all, Eigen::seq(2, 2)).array() * T2.array();          // OpCwiseMul
            g_T18(Eigen::all, Eigen::seq(3, 3)).array() += g_T19(Eigen::all, Eigen::seq(3, 3)).array() * T2.array();          // OpCwiseMul
            g_T18(Eigen::all, Eigen::seq(4, 4)).array() += g_T19(Eigen::all, Eigen::seq(4, 4)).array() * T2.array();          // OpCwiseMul
            g_T2(Eigen::all, Eigen::seq(0, 0)).array() += g_T19(Eigen::all, Eigen::seq(0, 0)).array() * T18.array();          // OpCwiseMul
            g_T2(Eigen::all, Eigen::seq(1, 1)).array() += g_T19(Eigen::all, Eigen::seq(1, 1)).array() * T18.array();          // OpCwiseMul
            g_T2(Eigen::all, Eigen::seq(2, 2)).array() += g_T19(Eigen::all, Eigen::seq(2, 2)).array() * T18.array();          // OpCwiseMul
            g_T2(Eigen::all, Eigen::seq(3, 3)).array() += g_T19(Eigen::all, Eigen::seq(3, 3)).array() * T18.array();          // OpCwiseMul
            g_T2(Eigen::all, Eigen::seq(4, 4)).array() += g_T19(Eigen::all, Eigen::seq(4, 4)).array() * T18.array();          // OpCwiseMul
            g_T14 += g_T18;                                                                                                   // OpSubs
            g_T17 -= g_T18;                                                                                                   // OpSubs
            g_T4(Eigen::all, Eigen::seq(0, 0))({4}, {0}) += g_T14(Eigen::all, Eigen::seq(0, 0));                              // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(1, 1))({4}, {0}) += g_T14(Eigen::all, Eigen::seq(1, 1));                              // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(2, 2))({4}, {0}) += g_T14(Eigen::all, Eigen::seq(2, 2));                              // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(3, 3))({4}, {0}) += g_T14(Eigen::all, Eigen::seq(3, 3));                              // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(4, 4))({4}, {0}) += g_T14(Eigen::all, Eigen::seq(4, 4));                              // OpMatBlock
            g_T16 += g_T17 * 0.5;                                                                                             // OpTimesConstScalar
            g_T6(Eigen::all, Eigen::seq(0, 0)).array() += g_T16(Eigen::all, Eigen::seq(0, 0)).array() * T12.array();          // OpCwiseMul
            g_T6(Eigen::all, Eigen::seq(1, 1)).array() += g_T16(Eigen::all, Eigen::seq(1, 1)).array() * T12.array();          // OpCwiseMul
            g_T6(Eigen::all, Eigen::seq(2, 2)).array() += g_T16(Eigen::all, Eigen::seq(2, 2)).array() * T12.array();          // OpCwiseMul
            g_T6(Eigen::all, Eigen::seq(3, 3)).array() += g_T16(Eigen::all, Eigen::seq(3, 3)).array() * T12.array();          // OpCwiseMul
            g_T6(Eigen::all, Eigen::seq(4, 4)).array() += g_T16(Eigen::all, Eigen::seq(4, 4)).array() * T12.array();          // OpCwiseMul
            g_T12(Eigen::all, Eigen::seq(0, 0)).array() += g_T16(Eigen::all, Eigen::seq(0, 0)).array() * T6.array();          // OpCwiseMul
            g_T12(Eigen::all, Eigen::seq(1, 1)).array() += g_T16(Eigen::all, Eigen::seq(1, 1)).array() * T6.array();          // OpCwiseMul
            g_T12(Eigen::all, Eigen::seq(2, 2)).array() += g_T16(Eigen::all, Eigen::seq(2, 2)).array() * T6.array();          // OpCwiseMul
            g_T12(Eigen::all, Eigen::seq(3, 3)).array() += g_T16(Eigen::all, Eigen::seq(3, 3)).array() * T6.array();          // OpCwiseMul
            g_T12(Eigen::all, Eigen::seq(4, 4)).array() += g_T16(Eigen::all, Eigen::seq(4, 4)).array() * T6.array();          // OpCwiseMul
            g_T9(Eigen::all, Eigen::seq(0, 0)) += g_T12(0) * T9;                                                              // OpMatDot
            g_T9(Eigen::all, Eigen::seq(1, 1)) += g_T12(1) * T9;                                                              // OpMatDot
            g_T9(Eigen::all, Eigen::seq(2, 2)) += g_T12(2) * T9;                                                              // OpMatDot
            g_T9(Eigen::all, Eigen::seq(3, 3)) += g_T12(3) * T9;                                                              // OpMatDot
            g_T9(Eigen::all, Eigen::seq(4, 4)) += g_T12(4) * T9;                                                              // OpMatDot
            g_T9(Eigen::all, Eigen::seq(0, 0)) += g_T12(0) * T9;                                                              // OpMatDot
            g_T9(Eigen::all, Eigen::seq(1, 1)) += g_T12(1) * T9;                                                              // OpMatDot
            g_T9(Eigen::all, Eigen::seq(2, 2)) += g_T12(2) * T9;                                                              // OpMatDot
            g_T9(Eigen::all, Eigen::seq(3, 3)) += g_T12(3) * T9;                                                              // OpMatDot
            g_T9(Eigen::all, Eigen::seq(4, 4)) += g_T12(4) * T9;                                                              // OpMatDot
            g_T8 += g_T9 / T6(0, 0);                                                                                          // OpDivideScalar
            g_T6(0) += (g_T9(Eigen::all, Eigen::seq(0, 0)).array() * T8.array()).sum() * (-1.0 / (T6(0, 0) * T6(0, 0)));      // OpDivideScalar
            g_T6(1) += (g_T9(Eigen::all, Eigen::seq(1, 1)).array() * T8.array()).sum() * (-1.0 / (T6(0, 0) * T6(0, 0)));      // OpDivideScalar
            g_T6(2) += (g_T9(Eigen::all, Eigen::seq(2, 2)).array() * T8.array()).sum() * (-1.0 / (T6(0, 0) * T6(0, 0)));      // OpDivideScalar
            g_T6(3) += (g_T9(Eigen::all, Eigen::seq(3, 3)).array() * T8.array()).sum() * (-1.0 / (T6(0, 0) * T6(0, 0)));      // OpDivideScalar
            g_T6(4) += (g_T9(Eigen::all, Eigen::seq(4, 4)).array() * T8.array()).sum() * (-1.0 / (T6(0, 0) * T6(0, 0)));      // OpDivideScalar
            g_T4(Eigen::all, Eigen::seq(0, 0))({0}, {0}) += g_T6(Eigen::all, Eigen::seq(0, 0));                               // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(1, 1))({0}, {0}) += g_T6(Eigen::all, Eigen::seq(1, 1));                               // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(2, 2))({0}, {0}) += g_T6(Eigen::all, Eigen::seq(2, 2));                               // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(3, 3))({0}, {0}) += g_T6(Eigen::all, Eigen::seq(3, 3));                               // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(4, 4))({0}, {0}) += g_T6(Eigen::all, Eigen::seq(4, 4));                               // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(0, 0))({1, 2, 3}, {0}) += g_T8(Eigen::all, Eigen::seq(0, 0));                         // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(1, 1))({1, 2, 3}, {0}) += g_T8(Eigen::all, Eigen::seq(1, 1));                         // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(2, 2))({1, 2, 3}, {0}) += g_T8(Eigen::all, Eigen::seq(2, 2));                         // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(3, 3))({1, 2, 3}, {0}) += g_T8(Eigen::all, Eigen::seq(3, 3));                         // OpMatBlock
            g_T4(Eigen::all, Eigen::seq(4, 4))({1, 2, 3}, {0}) += g_T8(Eigen::all, Eigen::seq(4, 4));                         // OpMatBlock
            // grad end is at g_T4
            g_T105 += g_T115(Eigen::seq(0, 0), Eigen::all);                                                             // OpMatConcat
            g_T108 += g_T115(Eigen::seq(1, 1), Eigen::all);                                                             // OpMatConcat
            g_T110 += g_T115(Eigen::seq(2, 2), Eigen::all);                                                             // OpMatConcat
            g_T112 += g_T115(Eigen::seq(3, 3), Eigen::all);                                                             // OpMatConcat
            g_T114 += g_T115(Eigen::seq(4, 4), Eigen::all);                                                             // OpMatConcat
            g_T5(Eigen::all, Eigen::seq(0, 0))({1}, {0}) += g_T105(Eigen::all, Eigen::seq(0, 0));                       // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(1, 1))({1}, {0}) += g_T105(Eigen::all, Eigen::seq(1, 1));                       // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(2, 2))({1}, {0}) += g_T105(Eigen::all, Eigen::seq(2, 2));                       // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(3, 3))({1}, {0}) += g_T105(Eigen::all, Eigen::seq(3, 3));                       // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(4, 4))({1}, {0}) += g_T105(Eigen::all, Eigen::seq(4, 4));                       // OpMatBlock
            g_T107 += g_T108;                                                                                           // OpAdd
            g_T23 += g_T108;                                                                                            // OpAdd
            g_T106(Eigen::all, Eigen::seq(0, 0)).array() += g_T107(Eigen::all, Eigen::seq(0, 0)).array() * T93.array(); // OpCwiseMul
            g_T106(Eigen::all, Eigen::seq(1, 1)).array() += g_T107(Eigen::all, Eigen::seq(1, 1)).array() * T93.array(); // OpCwiseMul
            g_T106(Eigen::all, Eigen::seq(2, 2)).array() += g_T107(Eigen::all, Eigen::seq(2, 2)).array() * T93.array(); // OpCwiseMul
            g_T106(Eigen::all, Eigen::seq(3, 3)).array() += g_T107(Eigen::all, Eigen::seq(3, 3)).array() * T93.array(); // OpCwiseMul
            g_T106(Eigen::all, Eigen::seq(4, 4)).array() += g_T107(Eigen::all, Eigen::seq(4, 4)).array() * T93.array(); // OpCwiseMul
            g_T93(Eigen::all, Eigen::seq(0, 0)).array() += g_T107(Eigen::all, Eigen::seq(0, 0)).array() * T106.array(); // OpCwiseMul
            g_T93(Eigen::all, Eigen::seq(1, 1)).array() += g_T107(Eigen::all, Eigen::seq(1, 1)).array() * T106.array(); // OpCwiseMul
            g_T93(Eigen::all, Eigen::seq(2, 2)).array() += g_T107(Eigen::all, Eigen::seq(2, 2)).array() * T106.array(); // OpCwiseMul
            g_T93(Eigen::all, Eigen::seq(3, 3)).array() += g_T107(Eigen::all, Eigen::seq(3, 3)).array() * T106.array(); // OpCwiseMul
            g_T93(Eigen::all, Eigen::seq(4, 4)).array() += g_T107(Eigen::all, Eigen::seq(4, 4)).array() * T106.array(); // OpCwiseMul
            g_T5(Eigen::all, Eigen::seq(0, 0))({1}, {0}) += g_T106(Eigen::all, Eigen::seq(0, 0));                       // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(1, 1))({1}, {0}) += g_T106(Eigen::all, Eigen::seq(1, 1));                       // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(2, 2))({1}, {0}) += g_T106(Eigen::all, Eigen::seq(2, 2));                       // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(3, 3))({1}, {0}) += g_T106(Eigen::all, Eigen::seq(3, 3));                       // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(4, 4))({1}, {0}) += g_T106(Eigen::all, Eigen::seq(4, 4));                       // OpMatBlock
            g_T109(Eigen::all, Eigen::seq(0, 0)).array() += g_T110(Eigen::all, Eigen::seq(0, 0)).array() * T93.array(); // OpCwiseMul
            g_T109(Eigen::all, Eigen::seq(1, 1)).array() += g_T110(Eigen::all, Eigen::seq(1, 1)).array() * T93.array(); // OpCwiseMul
            g_T109(Eigen::all, Eigen::seq(2, 2)).array() += g_T110(Eigen::all, Eigen::seq(2, 2)).array() * T93.array(); // OpCwiseMul
            g_T109(Eigen::all, Eigen::seq(3, 3)).array() += g_T110(Eigen::all, Eigen::seq(3, 3)).array() * T93.array(); // OpCwiseMul
            g_T109(Eigen::all, Eigen::seq(4, 4)).array() += g_T110(Eigen::all, Eigen::seq(4, 4)).array() * T93.array(); // OpCwiseMul
            g_T93(Eigen::all, Eigen::seq(0, 0)).array() += g_T110(Eigen::all, Eigen::seq(0, 0)).array() * T109.array(); // OpCwiseMul
            g_T93(Eigen::all, Eigen::seq(1, 1)).array() += g_T110(Eigen::all, Eigen::seq(1, 1)).array() * T109.array(); // OpCwiseMul
            g_T93(Eigen::all, Eigen::seq(2, 2)).array() += g_T110(Eigen::all, Eigen::seq(2, 2)).array() * T109.array(); // OpCwiseMul
            g_T93(Eigen::all, Eigen::seq(3, 3)).array() += g_T110(Eigen::all, Eigen::seq(3, 3)).array() * T109.array(); // OpCwiseMul
            g_T93(Eigen::all, Eigen::seq(4, 4)).array() += g_T110(Eigen::all, Eigen::seq(4, 4)).array() * T109.array(); // OpCwiseMul
            g_T5(Eigen::all, Eigen::seq(0, 0))({2}, {0}) += g_T109(Eigen::all, Eigen::seq(0, 0));                       // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(1, 1))({2}, {0}) += g_T109(Eigen::all, Eigen::seq(1, 1));                       // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(2, 2))({2}, {0}) += g_T109(Eigen::all, Eigen::seq(2, 2));                       // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(3, 3))({2}, {0}) += g_T109(Eigen::all, Eigen::seq(3, 3));                       // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(4, 4))({2}, {0}) += g_T109(Eigen::all, Eigen::seq(4, 4));                       // OpMatBlock
            g_T111(Eigen::all, Eigen::seq(0, 0)).array() += g_T112(Eigen::all, Eigen::seq(0, 0)).array() * T93.array(); // OpCwiseMul
            g_T111(Eigen::all, Eigen::seq(1, 1)).array() += g_T112(Eigen::all, Eigen::seq(1, 1)).array() * T93.array(); // OpCwiseMul
            g_T111(Eigen::all, Eigen::seq(2, 2)).array() += g_T112(Eigen::all, Eigen::seq(2, 2)).array() * T93.array(); // OpCwiseMul
            g_T111(Eigen::all, Eigen::seq(3, 3)).array() += g_T112(Eigen::all, Eigen::seq(3, 3)).array() * T93.array(); // OpCwiseMul
            g_T111(Eigen::all, Eigen::seq(4, 4)).array() += g_T112(Eigen::all, Eigen::seq(4, 4)).array() * T93.array(); // OpCwiseMul
            g_T93(Eigen::all, Eigen::seq(0, 0)).array() += g_T112(Eigen::all, Eigen::seq(0, 0)).array() * T111.array(); // OpCwiseMul
            g_T93(Eigen::all, Eigen::seq(1, 1)).array() += g_T112(Eigen::all, Eigen::seq(1, 1)).array() * T111.array(); // OpCwiseMul
            g_T93(Eigen::all, Eigen::seq(2, 2)).array() += g_T112(Eigen::all, Eigen::seq(2, 2)).array() * T111.array(); // OpCwiseMul
            g_T93(Eigen::all, Eigen::seq(3, 3)).array() += g_T112(Eigen::all, Eigen::seq(3, 3)).array() * T111.array(); // OpCwiseMul
            g_T93(Eigen::all, Eigen::seq(4, 4)).array() += g_T112(Eigen::all, Eigen::seq(4, 4)).array() * T111.array(); // OpCwiseMul
            g_T5(Eigen::all, Eigen::seq(0, 0))({3}, {0}) += g_T111(Eigen::all, Eigen::seq(0, 0));                       // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(1, 1))({3}, {0}) += g_T111(Eigen::all, Eigen::seq(1, 1));                       // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(2, 2))({3}, {0}) += g_T111(Eigen::all, Eigen::seq(2, 2));                       // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(3, 3))({3}, {0}) += g_T111(Eigen::all, Eigen::seq(3, 3));                       // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(4, 4))({3}, {0}) += g_T111(Eigen::all, Eigen::seq(4, 4));                       // OpMatBlock
            g_T113(Eigen::all, Eigen::seq(0, 0)).array() += g_T114(Eigen::all, Eigen::seq(0, 0)).array() * T93.array(); // OpCwiseMul
            g_T113(Eigen::all, Eigen::seq(1, 1)).array() += g_T114(Eigen::all, Eigen::seq(1, 1)).array() * T93.array(); // OpCwiseMul
            g_T113(Eigen::all, Eigen::seq(2, 2)).array() += g_T114(Eigen::all, Eigen::seq(2, 2)).array() * T93.array(); // OpCwiseMul
            g_T113(Eigen::all, Eigen::seq(3, 3)).array() += g_T114(Eigen::all, Eigen::seq(3, 3)).array() * T93.array(); // OpCwiseMul
            g_T113(Eigen::all, Eigen::seq(4, 4)).array() += g_T114(Eigen::all, Eigen::seq(4, 4)).array() * T93.array(); // OpCwiseMul
            g_T93(Eigen::all, Eigen::seq(0, 0)).array() += g_T114(Eigen::all, Eigen::seq(0, 0)).array() * T113.array(); // OpCwiseMul
            g_T93(Eigen::all, Eigen::seq(1, 1)).array() += g_T114(Eigen::all, Eigen::seq(1, 1)).array() * T113.array(); // OpCwiseMul
            g_T93(Eigen::all, Eigen::seq(2, 2)).array() += g_T114(Eigen::all, Eigen::seq(2, 2)).array() * T113.array(); // OpCwiseMul
            g_T93(Eigen::all, Eigen::seq(3, 3)).array() += g_T114(Eigen::all, Eigen::seq(3, 3)).array() * T113.array(); // OpCwiseMul
            g_T93(Eigen::all, Eigen::seq(4, 4)).array() += g_T114(Eigen::all, Eigen::seq(4, 4)).array() * T113.array(); // OpCwiseMul
            g_T11(Eigen::all, Eigen::seq(0, 0))({0}, {0}) += g_T93(Eigen::all, Eigen::seq(0, 0));                       // OpMatBlock
            g_T11(Eigen::all, Eigen::seq(1, 1))({0}, {0}) += g_T93(Eigen::all, Eigen::seq(1, 1));                       // OpMatBlock
            g_T11(Eigen::all, Eigen::seq(2, 2))({0}, {0}) += g_T93(Eigen::all, Eigen::seq(2, 2));                       // OpMatBlock
            g_T11(Eigen::all, Eigen::seq(3, 3))({0}, {0}) += g_T93(Eigen::all, Eigen::seq(3, 3));                       // OpMatBlock
            g_T11(Eigen::all, Eigen::seq(4, 4))({0}, {0}) += g_T93(Eigen::all, Eigen::seq(4, 4));                       // OpMatBlock
            g_T15 += g_T113;                                                                                            // OpAdd
            g_T23 += g_T113;                                                                                            // OpAdd
            g_T22(Eigen::all, Eigen::seq(0, 0)).array() += g_T23(Eigen::all, Eigen::seq(0, 0)).array() * T2.array();    // OpCwiseMul
            g_T22(Eigen::all, Eigen::seq(1, 1)).array() += g_T23(Eigen::all, Eigen::seq(1, 1)).array() * T2.array();    // OpCwiseMul
            g_T22(Eigen::all, Eigen::seq(2, 2)).array() += g_T23(Eigen::all, Eigen::seq(2, 2)).array() * T2.array();    // OpCwiseMul
            g_T22(Eigen::all, Eigen::seq(3, 3)).array() += g_T23(Eigen::all, Eigen::seq(3, 3)).array() * T2.array();    // OpCwiseMul
            g_T22(Eigen::all, Eigen::seq(4, 4)).array() += g_T23(Eigen::all, Eigen::seq(4, 4)).array() * T2.array();    // OpCwiseMul
            g_T2(Eigen::all, Eigen::seq(0, 0)).array() += g_T23(Eigen::all, Eigen::seq(0, 0)).array() * T22.array();    // OpCwiseMul
            g_T2(Eigen::all, Eigen::seq(1, 1)).array() += g_T23(Eigen::all, Eigen::seq(1, 1)).array() * T22.array();    // OpCwiseMul
            g_T2(Eigen::all, Eigen::seq(2, 2)).array() += g_T23(Eigen::all, Eigen::seq(2, 2)).array() * T22.array();    // OpCwiseMul
            g_T2(Eigen::all, Eigen::seq(3, 3)).array() += g_T23(Eigen::all, Eigen::seq(3, 3)).array() * T22.array();    // OpCwiseMul
            g_T2(Eigen::all, Eigen::seq(4, 4)).array() += g_T23(Eigen::all, Eigen::seq(4, 4)).array() * T22.array();    // OpCwiseMul
            // grad end is at g_T2
            g_T15 += g_T22;                                                                                                // OpSubs
            g_T21 -= g_T22;                                                                                                // OpSubs
            g_T5(Eigen::all, Eigen::seq(0, 0))({4}, {0}) += g_T15(Eigen::all, Eigen::seq(0, 0));                           // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(1, 1))({4}, {0}) += g_T15(Eigen::all, Eigen::seq(1, 1));                           // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(2, 2))({4}, {0}) += g_T15(Eigen::all, Eigen::seq(2, 2));                           // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(3, 3))({4}, {0}) += g_T15(Eigen::all, Eigen::seq(3, 3));                           // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(4, 4))({4}, {0}) += g_T15(Eigen::all, Eigen::seq(4, 4));                           // OpMatBlock
            g_T20 += g_T21 * 0.5;                                                                                          // OpTimesConstScalar
            g_T7(Eigen::all, Eigen::seq(0, 0)).array() += g_T20(Eigen::all, Eigen::seq(0, 0)).array() * T13.array();       // OpCwiseMul
            g_T7(Eigen::all, Eigen::seq(1, 1)).array() += g_T20(Eigen::all, Eigen::seq(1, 1)).array() * T13.array();       // OpCwiseMul
            g_T7(Eigen::all, Eigen::seq(2, 2)).array() += g_T20(Eigen::all, Eigen::seq(2, 2)).array() * T13.array();       // OpCwiseMul
            g_T7(Eigen::all, Eigen::seq(3, 3)).array() += g_T20(Eigen::all, Eigen::seq(3, 3)).array() * T13.array();       // OpCwiseMul
            g_T7(Eigen::all, Eigen::seq(4, 4)).array() += g_T20(Eigen::all, Eigen::seq(4, 4)).array() * T13.array();       // OpCwiseMul
            g_T13(Eigen::all, Eigen::seq(0, 0)).array() += g_T20(Eigen::all, Eigen::seq(0, 0)).array() * T7.array();       // OpCwiseMul
            g_T13(Eigen::all, Eigen::seq(1, 1)).array() += g_T20(Eigen::all, Eigen::seq(1, 1)).array() * T7.array();       // OpCwiseMul
            g_T13(Eigen::all, Eigen::seq(2, 2)).array() += g_T20(Eigen::all, Eigen::seq(2, 2)).array() * T7.array();       // OpCwiseMul
            g_T13(Eigen::all, Eigen::seq(3, 3)).array() += g_T20(Eigen::all, Eigen::seq(3, 3)).array() * T7.array();       // OpCwiseMul
            g_T13(Eigen::all, Eigen::seq(4, 4)).array() += g_T20(Eigen::all, Eigen::seq(4, 4)).array() * T7.array();       // OpCwiseMul
            g_T11(Eigen::all, Eigen::seq(0, 0)) += g_T13(0) * T11;                                                         // OpMatDot
            g_T11(Eigen::all, Eigen::seq(1, 1)) += g_T13(1) * T11;                                                         // OpMatDot
            g_T11(Eigen::all, Eigen::seq(2, 2)) += g_T13(2) * T11;                                                         // OpMatDot
            g_T11(Eigen::all, Eigen::seq(3, 3)) += g_T13(3) * T11;                                                         // OpMatDot
            g_T11(Eigen::all, Eigen::seq(4, 4)) += g_T13(4) * T11;                                                         // OpMatDot
            g_T11(Eigen::all, Eigen::seq(0, 0)) += g_T13(0) * T11;                                                         // OpMatDot
            g_T11(Eigen::all, Eigen::seq(1, 1)) += g_T13(1) * T11;                                                         // OpMatDot
            g_T11(Eigen::all, Eigen::seq(2, 2)) += g_T13(2) * T11;                                                         // OpMatDot
            g_T11(Eigen::all, Eigen::seq(3, 3)) += g_T13(3) * T11;                                                         // OpMatDot
            g_T11(Eigen::all, Eigen::seq(4, 4)) += g_T13(4) * T11;                                                         // OpMatDot
            g_T10 += g_T11 / T7(0, 0);                                                                                     // OpDivideScalar
            g_T7(0) += (g_T11(Eigen::all, Eigen::seq(0, 0)).array() * T10.array()).sum() * (-1.0 / (T7(0, 0) * T7(0, 0))); // OpDivideScalar
            g_T7(1) += (g_T11(Eigen::all, Eigen::seq(1, 1)).array() * T10.array()).sum() * (-1.0 / (T7(0, 0) * T7(0, 0))); // OpDivideScalar
            g_T7(2) += (g_T11(Eigen::all, Eigen::seq(2, 2)).array() * T10.array()).sum() * (-1.0 / (T7(0, 0) * T7(0, 0))); // OpDivideScalar
            g_T7(3) += (g_T11(Eigen::all, Eigen::seq(3, 3)).array() * T10.array()).sum() * (-1.0 / (T7(0, 0) * T7(0, 0))); // OpDivideScalar
            g_T7(4) += (g_T11(Eigen::all, Eigen::seq(4, 4)).array() * T10.array()).sum() * (-1.0 / (T7(0, 0) * T7(0, 0))); // OpDivideScalar
            g_T5(Eigen::all, Eigen::seq(0, 0))({0}, {0}) += g_T7(Eigen::all, Eigen::seq(0, 0));                            // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(1, 1))({0}, {0}) += g_T7(Eigen::all, Eigen::seq(1, 1));                            // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(2, 2))({0}, {0}) += g_T7(Eigen::all, Eigen::seq(2, 2));                            // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(3, 3))({0}, {0}) += g_T7(Eigen::all, Eigen::seq(3, 3));                            // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(4, 4))({0}, {0}) += g_T7(Eigen::all, Eigen::seq(4, 4));                            // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(0, 0))({1, 2, 3}, {0}) += g_T10(Eigen::all, Eigen::seq(0, 0));                     // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(1, 1))({1, 2, 3}, {0}) += g_T10(Eigen::all, Eigen::seq(1, 1));                     // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(2, 2))({1, 2, 3}, {0}) += g_T10(Eigen::all, Eigen::seq(2, 2));                     // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(3, 3))({1, 2, 3}, {0}) += g_T10(Eigen::all, Eigen::seq(3, 3));                     // OpMatBlock
            g_T5(Eigen::all, Eigen::seq(4, 4))({1, 2, 3}, {0}) += g_T10(Eigen::all, Eigen::seq(4, 4));                     // OpMatBlock
            // grad end is at g_T5

            //
            //
            //
            //

            F = T132;
            //

            dFdUL = g_T4;
            // std::cout << g_T4 << std::endl
            //           << std::endl;
            dFdUR = g_T5;
            // std::cout << g_T5 << std::endl
            //           << std::endl;
        }

        template <typename TU, typename TGradU, typename TFlux, typename TNorm, typename TGU, typename TGGradU>
        void ViscousFlux_IdealGas_N_AutoDiffGen(
            const TU &U, const TGradU &GradU, TNorm norm, bool adiabatic, real gamma, real mu, real k, real Cp, TFlux &Flux,
            TGU &GU, TGGradU &GGradU)
        {
            Eigen::Matrix<double, 3, 3> T0 = Eigen::Matrix3d::Identity();                    // OpIn
            Eigen::Matrix<double, 1, 1> T1 = Eigen::Matrix<real, 1, 1>{gamma - 1};           // OpIn
            Eigen::Matrix<double, 1, 1> T2 = Eigen::Matrix<real, 1, 1>{gamma / (gamma - 1)}; // OpIn
            Eigen::Matrix<double, 3, 1> T3 = norm;                                           // OpIn
            Eigen::Matrix<double, 1, 1> T4 {{mu}};                                             // OpIn
            Eigen::Matrix<double, 1, 1> T5  {{k}};                                              // OpIn
            Eigen::Matrix<double, 1, 1> T6  {{Cp}};                                             // OpIn
            Eigen::Matrix<double, 5, 1> T7  =U;                                              // OpIn
            Eigen::Matrix<double, 3, 5> T8  =GradU;                                          // OpIn

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

            Flux({0}, {0}).setZero();
            Flux({1, 2, 3, 4}, {0}) = T54;
            //

            GU({0, 1, 2, 3, 4}, 0).setZero();
            GU({0, 1, 2, 3, 4}, {1, 2, 3, 4}) = g_T7;
            // std::cout << g_T4 << std::endl
            //           << std::endl;
            GGradU = g_T8;
            // std::cout << g_T5 << std::endl
            //           << std::endl;
        }
    }
}