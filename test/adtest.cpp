#include "../DNDS_AutoDiff.hpp"
#include <iostream>

int main()
{
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

        u1 = u1.transpose().matmul(u1);
        u1.back();
        std::cout << "u1 = u0 dot u0" << std::endl;
        std::cout << u1 << std::endl;
        std::cout << u0 << std::endl;

        u1 = u0.matmul(u0.transpose());
        u1.back();
        std::cout << "u1 = u0 tensor u0" << std::endl;
        std::cout << u1 << std::endl;
        std::cout << u0 << std::endl;
        // 2 2 3 4 2 0 0 0 3 0 0 0 4 0 0 0
        // 0 1 0 0 1 4 3 4 0 3 0 0 0 4 0 0
        // 0 0 1 0 0 0 2 0 1 2 6 4 0 0 4 0
        // 0 0 0 1 0 0 0 2 0 0 0 3 1 2 3 8

        u2 = u0.dot(u0);
        u2.back();
        std::cout << "u2 = u0 dot u0" << std::endl;
        std::cout << u2 << std::endl;
        std::cout << u0 << std::endl;

        u0 = Eigen::MatrixXd{{2}};
        u1 = uIn.transpose();

        u2 = u1 / u0 * u0 * u0;
        u2.back();
        std::cout << "u2 = u1 * u0" << std::endl;
        std::cout << u2 << std::endl;
        std::cout << u0 << std::endl;
        std::cout << u1 << std::endl;

        u2 = u1 / u1 / u1 * u1 * u1 * 2;
        u2.back();
        std::cout << "u2 = 2  u1" << std::endl;
        std::cout << u2 << std::endl;
        std::cout << u1 << std::endl;

        u1 = Eigen::MatrixXd{{-2}, {-1}, {1}, {2}};
        u3 = u1.hyFixExp(u0).abs();
        u3.back();
        std::cout << "u3 = u1.hyFixExp(u0).abs()" << std::endl;
        std::cout << u3 << std::endl;
        std::cout << u0 << std::endl;
        std::cout << u1 << std::endl;
    }

    {
        using namespace DNDS;
        using namespace AutoDiff;
        real gamma = 1.4;

        static real scaleHartenYee = 0.01;
        auto UL = Eigen::Vector<real, 5>{1.0000, 0.5916, 0, 0, 2.6750};
        auto UR = Eigen::Vector<real, 5>{1.0000, 0, 0, 0, 2.5};

        if (!(UL(0) > 0 && UR(0) > 0))
        {
        }
        assert(UL(0) > 0 && UR(0) > 0);
        ADEigenMat ULad(UL), URad(UR);
        std::cout << ULad << std::endl
                  << URad << std::endl;
        ADEigenMat veloL = ULad({1, 2, 3}, {0}) / ULad({0}, {0});
        ADEigenMat veloR = URad({1, 2, 3}, {0}) / URad({0}, {0});
        ADEigenMat vsqrL = veloL.dot(veloL);
        ADEigenMat vsqrR = veloR.dot(veloR);
        
        ADEigenMat pL = (ULad({4}, {0}) - ULad({0}, {0}) * vsqrL * 0.5) * (gamma - 1);
        ADEigenMat pR = (URad({4}, {0}) - URad({0}, {0}) * vsqrR * 0.5) * (gamma - 1);
        ADEigenMat HL = (ULad({4}, {0}) + pL) / ULad({0}, {0});
        ADEigenMat HR = (URad({4}, {0}) + pR) / URad({0}, {0});
        ADEigenMat sqrtRhoL = ULad({0}, {0}).sqrt();
        ADEigenMat sqrtRhoR = URad({0}, {0}).sqrt();
        ADEigenMat sqrtRhoLpR = sqrtRhoL + sqrtRhoR;
        std::cout << pL << std::endl
                  << pR << std::endl;

        ADEigenMat veloRoe = (veloL * sqrtRhoL + veloR * sqrtRhoR) / sqrtRhoLpR;
        ADEigenMat vsqrRoe = veloRoe.dot(veloRoe);
        ADEigenMat HRoe = (HL * sqrtRhoL + HR * sqrtRhoR) / sqrtRhoLpR;
        ADEigenMat asqrRoe = (HRoe - vsqrRoe * 0.5) * (gamma - 1);
        ADEigenMat rhoRoe = sqrtRhoL * sqrtRhoR;

        std::cout << asqrRoe.d() << std::endl;
        if (!(asqrRoe.d()(0, 0) > 0))
        {
            assert(false);
        }

        ADEigenMat aRoe = asqrRoe.sqrt();

        ADEigenMat lam0 = (veloRoe({0}, {0}) - aRoe).abs();
        ADEigenMat lam123 = (veloRoe({0}, {0})).abs();
        ADEigenMat lam4 = (veloRoe({0}, {0}) + aRoe).abs();

        //*HY
        ADEigenMat thresholdHartenYee = (vsqrRoe.sqrt() + aRoe) * (scaleHartenYee * 0.5);
        lam0 = lam0.hyFixExp(thresholdHartenYee);
        lam4 = lam4.hyFixExp(thresholdHartenYee);
        //*HY

        ADEigenMat _One{Eigen::MatrixXd{{1}}}, _Zero{Eigen::MatrixXd{{0}}};

        ADEigenMat K0 =
            ADEigenMat::concat0({_One,
                                 veloRoe({0}, {0}) - aRoe,
                                 veloRoe({1}, {0}),
                                 veloRoe({2}, {0}),
                                 HRoe - veloRoe({0}, {0}) * aRoe});
        ADEigenMat K4 =
            ADEigenMat::concat0({_One,
                                 veloRoe({0}, {0}) + aRoe,
                                 veloRoe({1}, {0}),
                                 veloRoe({2}, {0}),
                                 HRoe + veloRoe({0}, {0}) * aRoe});
        ADEigenMat K1 =
            ADEigenMat::concat0({_One,
                                 veloRoe,
                                 vsqrRoe * 0.5});
        ADEigenMat K2 =
            ADEigenMat::concat0({_Zero,
                                 _Zero,
                                 _One,
                                 _Zero,
                                 veloRoe({1}, {0})});
        ADEigenMat K3 =
            ADEigenMat::concat0({_Zero,
                                 _Zero,
                                 _Zero,
                                 _One,
                                 veloRoe({2}, {0})});

        ADEigenMat incRho = URad({0}, {0}) - ULad({0}, {0});
        ADEigenMat incP = pR - pL;
        ADEigenMat incVelo = veloR - veloL;

        ADEigenMat alpha0 = (incP - rhoRoe * aRoe * incVelo({0}, {0})) / aRoe * 0.5;
        ADEigenMat alpha4 = (incP + rhoRoe * aRoe * incVelo({0}, {0})) / aRoe * 0.5;
        ADEigenMat alpha1 = incRho - incP / asqrRoe;
        ADEigenMat alpha2 = rhoRoe * incVelo({1}, {0});
        ADEigenMat alpha3 = rhoRoe * incVelo({2}, {0});

        ADEigenMat FL =
            ADEigenMat::concat0({ULad({1}, {0}),
                                 ULad({1}, {0}) * veloL({0}, {0}) + pL,
                                 ULad({2}, {0}) * veloL({0}, {0}),
                                 ULad({3}, {0}) * veloL({0}, {0}),
                                 (ULad({4}, {0}) + pL) * veloL({0}, {0})});
        ADEigenMat FR =
            ADEigenMat::concat0({URad({1}, {0}),
                                 URad({1}, {0}) * veloR({0}, {0}) + pR,
                                 URad({2}, {0}) * veloR({0}, {0}),
                                 URad({3}, {0}) * veloR({0}, {0}),
                                 (URad({4}, {0}) + pR) * veloR({0}, {0})});
        ADEigenMat FInc = K0 * (alpha0 * lam0) +
                          K4 * (alpha4 * lam4) +
                          K1 * (alpha1 * lam123) +
                          K2 * (alpha2 * lam123) +
                          K3 * (alpha3 * lam123);
        ADEigenMat FOut = FL;
        FOut.back();

        // F = FOut.d();
        // dFdUL = ULad.g();
        // dFdUR = URad.g();
        // std::cout << F.transpose() << std::endl;
        // std::cout << dFdUL << std::endl;
        // std::cout << dFdUR << std::endl;
    }

    return 0;
}