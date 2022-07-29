#include "../../DNDS_Gas.hpp"

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

    Eigen::Vector<real, 5> F;
    Gas::RoeFlux_IdealGas_HartenYee(Eigen::Vector<real, 5>{1.0000, 0.5916, 0, 0, 2.6750}, Eigen::Vector<real, 5>{1.0000, 0, 0, 0, 2.5}, 1.4, F);

    std::cout << "F\n"
              << F.transpose() << std::endl;
    // 0.369291683630373 1.54891136111009 0 1.45489148963839

        return 0;
}