#include <iostream>

#define DNDS_AUTODIFF_DEBUG_PRINTTOPO
#define DNDS_AUTODIFF_GENERATE_CODE
#include "../../DNDS_AutoDiff.hpp"
#include "../../DNDS_Gas.hpp"

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

    Eigen::Matrix<real, 3, 5> F, F1;

    Gas::ViscousFlux_IdealGas(U, GU, n, false, 1.4, 1e-3, 1e-3, 1e3, F);
    std::cout << n.transpose() * F << std::endl;
    Gas::ViscousFlux_IdealGas(U1, GU1, n, false, 1.4, 1e-3, 1e-3, 1e3, F1);
    std::cout << n.transpose() * (F1 - F) / eps << std::endl;

    {
        using namespace AutoDiff;

        // ADEigenMat _One{Eigen::MatrixXd{{1}}}, _Zero{Eigen::MatrixXd{{0}}};
        ADEigenMat _I(Eigen::MatrixXd::Identity(3, 3));
        ADEigenMat gammaM1(gammaD - Eigen::MatrixXd{{1}});
        ADEigenMat gammaDivGammaM1(Eigen::MatrixXd{{gammaD(0) / (gammaD(0) - 1)}});
        // ADEigenMat scaleHartenYee(scaleHY);
        ADEigenMat nad(n);
        ADEigenMat _mu(mu), _k(k), _cp(cp);

        ADEigenMat Uad(U), GUad(GU);
        ADEigenMat rhoV = Uad({1, 2, 3}, {0});
        ADEigenMat rho = Uad({0}, {0});
        ADEigenMat E = Uad({4}, {0});
        ADEigenMat velo = rhoV / rho;
        ADEigenMat vsqr = velo.dot(velo);
        ADEigenMat GradRho = GUad({0, 1, 2}, {0});
        ADEigenMat GradRhoV = GUad({0, 1, 2}, {1, 2, 3});
        ADEigenMat GradE = GUad({0, 1, 2}, {4});
        ADEigenMat veloT = velo.transpose();
        // ADEigenMat rhoVT = rhoV.transpose();

        ADEigenMat strainRate = (GradRhoV - GradRho.matmul(veloT)) / rho;

        ADEigenMat GradP = (GradE - (GradRhoV.matmul(velo) +
                                     strainRate.matmul(rhoV)) *
                                        0.5) *
                           gammaM1;

        ADEigenMat p = (E - rho * vsqr * 0.5) * gammaM1;
        ADEigenMat rhoSqr = rho*rho;
        ADEigenMat GradT = (GradP * rho - GradRho * p) *
                           (gammaDivGammaM1 / (_cp * rhoSqr));
        ADEigenMat strainRateSym = strainRate + strainRate.transpose();
        ADEigenMat strainRateTrace = strainRate({0}, {0}) +
                                        strainRate({1}, {1}) +
                                        strainRate({2}, {2});

        ADEigenMat tau = (strainRateSym + _I * (strainRateTrace * (-2. / 3.))) * _mu;
        ADEigenMat taut = tau.matmul(velo)+ GradT * _k;

        ADEigenMat Ff = tau.matmul(nad); // tau is symmetric
        ADEigenMat Fe = taut.dot(nad);

        ADEigenMat F4 = ADEigenMat::concat0({Ff, Fe});

        F4.back();

        std::cout << std::endl << objID[F4.ptr()] << std::endl
                  << F4 << std::endl
                  << std::endl;
        std::cout << std::endl
                  << objID[Uad.ptr()] << std::endl
                  << Uad.g() << std::endl
                  << std::endl;
        std::cout << objID[GUad.ptr()] << std::endl
                  << GUad.g() << std::endl
                  << std::endl;
    }
    return 0;
}