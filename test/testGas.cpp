#include "../DNDS_Gas.hpp"

#include "time.h"

int main()
{
    using namespace DNDS;
    Eigen::Vector<real, 5> U1{1.0000, 0.5916, 0, 0, 2.6750};
    Eigen::Vector<real, 5> U2{1.0000, 0, 0, 0, 2.5};
    Eigen::Vector<real, 5> U3{1.0000, 1, 1, 0.5, 6};

    auto LeV3 = Gas::IdealGas_EulerGasLeftEigenVector(U3, 1.4);
    auto ReV3 = Gas::IdealGas_EulerGasRightEigenVector(U3, 1.4);

    std::cout << "LeV\n"
              << LeV3 << std::endl;
    std::cout << "ReV\n"
              << ReV3 << std::endl;
    std::cout << "LeV * ReV\n"
              << LeV3 * ReV3 << std::endl;

    Eigen::Vector<real, 5> F, F1, F2;
    Gas::RoeFlux_IdealGas_HartenYee(
        Eigen::Vector<real, 5>{1.0000, 0.5916, 0, 0, 2.6750},
        Eigen::Vector<real, 5>{1.0000, 0, 0, 0, 2.5},
        1.4, F,
        [&]() {});
    real eps = 1e-5;

    int N = 10000;
    auto t0 = clock();
    for (int i = 0; i < N; i++)
        Gas::RoeFlux_IdealGas_HartenYee(
            Eigen::Vector<real, 5>{1.0000 + 1e-6 * (N - i), 0.5916 + 1e-6 * (N - i - 1), 0 + 1e-6 * (N - i - 1), 0 + 1e-6 * (N - i - 1), 2.6750 + 1e-6 * (N - i - 1)},
            Eigen::Vector<real, 5>{1.0000 + 1e-6 * (N - i - 1), 0 + 1e-6 * (N - i - 1), 0 + 1e-6 * (N - i - 1), 0 + 1e-6 * (N - i - 1), 2.5 + 1e-6 * (N - i - 1)},
            1.4, F1,
            [&]() {});
    std::cout << "\n=== TIME: " << double(clock() - t0) / CLOCKS_PER_SEC << std::endl;
    Eigen::Vector<real, 5> dFdUL0 = (F1 - F) / 1e-6;

    std::cout << "F\n"
              << F.transpose() << std::endl;

    Eigen::Matrix<real, 5, 5> gL;
    Eigen::Matrix<real, 5, 5> gR;
    gL.setZero(), gR.setZero();
    t0 = clock();
    for (int i = 0; i < N; i++)
        Gas::RoeFlux_IdealGas_HartenYee_AutoDiff(
            Eigen::Vector<real, 5>{1.0000, 0.5916, 0, 0, 2.6750},
            Eigen::Vector<real, 5>{1.0000, 0, 0, 0, 2.5},
            1.4, F, gL, gR,
            [&]() {});
    std::cout << "\n=== TIME: " << double(clock() - t0) / CLOCKS_PER_SEC << std::endl;
    std::cout << "Done" << std::endl;
    std::cout << "F\n"
              << F.transpose() << std::endl
              << "GL\n"
              << gL << std::endl
              << "GR\n"
              << gR << std::endl;
    std::cout << "dFdUL0\n"
              << dFdUL0.transpose() << std::endl;

    // 0.369291683630373 1.54891136111009 0 1.45489148963839

    return 0;
}