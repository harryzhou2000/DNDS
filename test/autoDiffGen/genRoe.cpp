#include <iostream>

#define DNDS_AUTODIFF_DEBUG_PRINTTOPO
#define DNDS_AUTODIFF_GENERATE_CODE
#include "../../DNDS_AutoDiff.hpp"

int main()
{
    using namespace DNDS;

    auto UL = Eigen::Vector<real, 5>{1.0000, 0.5916, 0, 0, 2.6750};
    auto UR = Eigen::Vector<real, 5>{1.0000, -0.5916, 0, 0, 2.6750};
    auto gammaD = Eigen::MatrixXd{{1.4}};
    auto scaleHY = Eigen::MatrixXd{{0.01}};

    {
        using namespace AutoDiff;

        if (!(UL(0) > 0 && UR(0) > 0))
        {
            DNDS_assert(UL(0) > 0 && UR(0) > 0);
        }

        ADEigenMat _One{Eigen::MatrixXd{{1}}}, _Zero{Eigen::MatrixXd{{0}}};
        ADEigenMat gammaM1(gammaD - Eigen::MatrixXd{{1}});
        ADEigenMat scaleHartenYee(scaleHY);

        ADEigenMat ULad(UL), URad(UR);
        ADEigenMat rhoL = ULad({0}, {0});
        ADEigenMat rhoR = URad({0}, {0});
        ADEigenMat veloL = ULad({1, 2, 3}, {0}) / rhoL;
        ADEigenMat veloR = URad({1, 2, 3}, {0}) / rhoR;
        ADEigenMat vsqrL = veloL.dot(veloL);
        ADEigenMat vsqrR = veloR.dot(veloR);
        ADEigenMat EL = ULad({4}, {0});
        ADEigenMat ER = URad({4}, {0});
        ADEigenMat pL = (EL - rhoL * vsqrL * 0.5) * (gammaM1);
        ADEigenMat pR = (ER - rhoR * vsqrR * 0.5) * (gammaM1);
        ADEigenMat HL = (EL + pL) / rhoL;
        ADEigenMat HR = (ER + pR) / rhoR;
        ADEigenMat sqrtRhoL = rhoL.sqrt();
        ADEigenMat sqrtRhoR = rhoR.sqrt();
        ADEigenMat sqrtRhoLpR = sqrtRhoL + sqrtRhoR;

        ADEigenMat veloRoe = (veloL * sqrtRhoL + veloR * sqrtRhoR) / sqrtRhoLpR;
        ADEigenMat vsqrRoe = veloRoe.dot(veloRoe);
        ADEigenMat HRoe = (HL * sqrtRhoL + HR * sqrtRhoR) / sqrtRhoLpR;
        ADEigenMat asqrRoe = (HRoe - vsqrRoe * 0.5) * (gammaM1);
        ADEigenMat rhoRoe = sqrtRhoL * sqrtRhoR;

        // std::cout << asqrRoe.d() << std::endl;
        if (!(asqrRoe.d()(0, 0) > 0))
        {
            DNDS_assert(false);
        }

        ADEigenMat aRoe = asqrRoe.sqrt();

        ADEigenMat veloRoe0 = veloRoe({0}, {0});
        ADEigenMat veloRoe1 = veloRoe({1}, {0});
        ADEigenMat veloRoe2 = veloRoe({2}, {0});
        ADEigenMat lam0 = (veloRoe0 - aRoe).abs();
        ADEigenMat lam123 = (veloRoe0).abs();
        ADEigenMat lam4 = (veloRoe0 + aRoe).abs();

        //*HY
        ADEigenMat thresholdHartenYee = (vsqrRoe.sqrt() + aRoe) * (scaleHartenYee * 0.5);
        lam0 = lam0.hyFixExp(thresholdHartenYee);
        lam4 = lam4.hyFixExp(thresholdHartenYee);
        //*HY

        ADEigenMat K0 =
            ADEigenMat::concat0({_One,
                                 veloRoe0 - aRoe,
                                 veloRoe1,
                                 veloRoe({2}, {0}),
                                 HRoe - veloRoe0 * aRoe});
        ADEigenMat K4 =
            ADEigenMat::concat0({_One,
                                 veloRoe0 + aRoe,
                                 veloRoe1,
                                 veloRoe2,
                                 HRoe + veloRoe0 * aRoe});
        ADEigenMat K1 =
            ADEigenMat::concat0({_One,
                                 veloRoe,
                                 vsqrRoe * 0.5});
        ADEigenMat K2 =
            ADEigenMat::concat0({_Zero,
                                 _Zero,
                                 _One,
                                 _Zero,
                                 veloRoe1});
        ADEigenMat K3 =
            ADEigenMat::concat0({_Zero,
                                 _Zero,
                                 _Zero,
                                 _One,
                                 veloRoe2});

        ADEigenMat incRho = rhoR - rhoL;
        ADEigenMat incP = pR - pL;
        ADEigenMat incVelo = veloR - veloL;
        ADEigenMat incVelo0 = incVelo({0}, {0});
        ADEigenMat incVelo1 = incVelo({1}, {0});
        ADEigenMat incVelo2 = incVelo({2}, {0});

        ADEigenMat alpha0 = (incP - rhoRoe * aRoe * incVelo0) / aRoe * 0.5;
        ADEigenMat alpha4 = (incP + rhoRoe * aRoe * incVelo0) / aRoe * 0.5;
        ADEigenMat alpha1 = incRho - incP / asqrRoe;
        ADEigenMat alpha2 = rhoRoe * incVelo1;
        ADEigenMat alpha3 = rhoRoe * incVelo2;

        ADEigenMat veloL0 = veloL({0}, {0});
        ADEigenMat veloR0 = veloR({0}, {0});

        ADEigenMat FL =
            ADEigenMat::concat0({ULad({1}, {0}),
                                 ULad({1}, {0}) * veloL0 + pL,
                                 ULad({2}, {0}) * veloL0,
                                 ULad({3}, {0}) * veloL0,
                                 (EL + pL) * veloL0});
        ADEigenMat FR =
            ADEigenMat::concat0({URad({1}, {0}),
                                 URad({1}, {0}) * veloR0 + pR,
                                 URad({2}, {0}) * veloR0,
                                 URad({3}, {0}) * veloR0,
                                 (ER + pR) * veloR0});
        ADEigenMat FInc = K0 * (alpha0 * lam0) +
                          K4 * (alpha4 * lam4) +
                          K1 * (alpha1 * lam123) +
                          K2 * (alpha2 * lam123) +
                          K3 * (alpha3 * lam123);
        ADEigenMat FOut = (FL + FR - FInc) * 0.5;
        FOut.back();

        std::cout << std::endl
                  << ULad.g() << std::endl
                  << std::endl;
        std::cout << URad.g() << std::endl
                  << std::endl;
    }
    return 0;
}