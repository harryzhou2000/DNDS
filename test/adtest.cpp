#include "../DNDS_AutoDiff.hpp"
#include <iostream>

int main()
{
    using namespace DNDS;

    Eigen::MatrixXd uIn{{1, 2, 3, 4}};
    AutoDiff::ADEigenMat u0, u1, u2, u3, u4;
    u0 = uIn.transpose();
    u1 = u0 * u0;
    u2 = u1 * u0;
    u3 = u2 * u0;
    u4 = u3 * u0;

    u4 = u4 + u4;
    u3 = u2 * u1;
    u1 = (u2 + u3) * u4 - u1;

    u4.back();

    std::cout << "u4 = 2u0^5" << std::endl;
    std::cout << u4 << std::endl;
    std::cout << u0 << std::endl;

    u3.back();
    std::cout << "u3 = u0^5" << std::endl;
    std::cout << u3 << std::endl;
    std::cout << u0 << std::endl;

    u1.back();
    std::cout << "u1 = 2u0^10 + 2u0^8 - u0^2" << std::endl;
    std::cout << u1 << std::endl;
    std::cout << u0 << std::endl;

    //  34           0           0           0
    //   0       12284           0           0
    //   0           0      428646           0
    //   0           0           0 5.50502e+06

    u0 = uIn.transpose();
    u1 = u0 * u0;
    u1 = u1.sqrt();

    u1.back();
    std::cout << "u1 = u0" << std::endl;
    std::cout << u1 << std::endl;
    std::cout << u0 << std::endl;

    u2 = u1({1, 2}, {0});

    u2.back();
    std::cout << "u2 = u0({1, 2}, {0})" << std::endl;
    std::cout << u2 << std::endl;
    std::cout << u0 << std::endl;

    u3 = u2.clone();
    u3.back();
    std::cout << "u3 = u0({1, 2}, {0})" << std::endl;
    std::cout << u3 << std::endl;
    std::cout << u0 << std::endl;

    u4 = AutoDiff::ADEigenMat::concat0({u2, u3});
    u4.back();
    std::cout << "u4 = u0({1, 2}, {0})u0({1, 2}, {0}) " << std::endl;
    std::cout << u4 << std::endl;
    std::cout << u0 << std::endl;

    return 0;
}