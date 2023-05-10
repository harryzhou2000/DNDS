#include <iostream>
#include "../../Eigen/Dense"
#include <vector>
#include "../../DNDS_Defines.h"

/**
 * @brief input vector<Eigen::Array-like>
 */
template <typename TinOthers, typename Tout>
static inline void FWBAP_L2_Multiway_Polynomial2D(const TinOthers &uOthers, int Nother, Tout &uOut)
{
    using namespace DNDS;
    static const int p = 4;
    static const real verySmallReal_pDiP = std::pow(verySmallReal, 1.0 / p);

    Eigen::ArrayXXd uUp; //* copy!
    uUp.resizeLike(uOthers[0]);
    uUp.setZero();
    Eigen::ArrayXd uDown;
    uDown.resize(uOthers[0].cols());
    uDown.setZero();
    Eigen::ArrayXXd uMax = uUp + verySmallReal;
    for (int iOther = 0; iOther < Nother; iOther++)
        uMax = uMax.max(uOthers[iOther].abs());
    uMax.rowwise() = uMax.colwise().maxCoeff();
    uOut = uMax;

    for (int iOther = 0; iOther < Nother; iOther++)
    {
        Eigen::ArrayXd thetaNorm;
        Eigen::ArrayXXd theta = uOthers[iOther] / uMax;
        switch (theta.rows())
        {
        case 2:
            thetaNorm =
                theta(0, Eigen::all).pow(2) +
                theta(1, Eigen::all).pow(2);
            break;
        case 3:
            thetaNorm =
                theta(0, Eigen::all).pow(2) +
                theta(1, Eigen::all).pow(2) * 0.5 +
                theta(2, Eigen::all).pow(2);
            break;
        case 4:
            thetaNorm =
                theta(0, Eigen::all).pow(2) +
                theta(1, Eigen::all).pow(2) * (1. / 3.) +
                theta(2, Eigen::all).pow(2) * (1. / 3.) +
                theta(3, Eigen::all).pow(2);
            break;

        default:
            DNDS_assert(false);
            break;
        }
        thetaNorm += verySmallReal_pDiP;

        uDown += thetaNorm.pow(-p);
        uUp += theta.rowwise() * thetaNorm.pow(-p).transpose();
    }

    // std::cout << uUp << std::endl;
    // std::cout << uDown << std::endl;
    uOut *= uUp.rowwise() / (uDown.transpose() + verySmallReal);

    for (int iOther = 0; iOther < Nother; iOther++)
    {
        Eigen::ArrayXd uDotuOut;
        switch (uOut.rows())
        {
        case 2:
            uDotuOut =
                uOthers[iOther](0, Eigen::all) * uOut(0, Eigen::all) +
                uOthers[iOther](1, Eigen::all) * uOut(1, Eigen::all);
            break;
        case 3:
            uDotuOut =
                uOthers[iOther](0, Eigen::all) * uOut(0, Eigen::all) +
                uOthers[iOther](1, Eigen::all) * uOut(1, Eigen::all) * 0.5 +
                uOthers[iOther](2, Eigen::all) * uOut(2, Eigen::all);
            break;
        case 4:
            uDotuOut =
                uOthers[iOther](0, Eigen::all) * uOut(0, Eigen::all) +
                uOthers[iOther](1, Eigen::all) * uOut(1, Eigen::all) * (1. / 3.) +
                uOthers[iOther](2, Eigen::all) * uOut(2, Eigen::all) * (1. / 3.) +
                uOthers[iOther](3, Eigen::all) * uOut(3, Eigen::all);
            break;

        default:
            DNDS_assert(false);
            break;
        }

        uOut.rowwise() *= 0.5 * (uDotuOut.sign().transpose() + 1);
    }

    if (uOut.hasNaN())
    {
        std::cout << "Limiter FWBAP_L2_Multiway Failed" << std::endl;
        std::cout << uMax.transpose() << std::endl;
        std::cout << uUp.transpose() << std::endl;
        std::cout << uDown.transpose() << std::endl;
        abort();
    }
}

int main()
{
    Eigen::ArrayXXd A{
        {1, 0.5, 1, 7, 0},
        {0, 1, 2, 4, 0},
        {0, 0.5, 3, 2, 0}};

    Eigen::ArrayXXd B{
        {1, 0.5, 1, 7, 0},
        {0, 1, 2, 0, 0},
        {0, 0.5, 3, 0, 0}};

    Eigen::ArrayXXd C{
        {-1, 0, 1, 1, 0},
        {0, -2, 2, 2, 0},
        {1, 0, 3, 0, 0}};
    std::vector<Eigen::ArrayXXd> Uothers{A, B, C};
    Eigen::ArrayXXd uout;
    FWBAP_L2_Multiway_Polynomial2D(Uothers, Uothers.size(), uout);

    std::cout << uout << std::endl;

    return 0;
}