#include "DNDS_HardEigen.h"
#include <iostream>

namespace DNDS
{
    namespace HardEigen
    {

// #define EIGEN_USE_LAPACKE
        void EigenLeastSquareInverse(const Eigen::MatrixXd &A, Eigen::MatrixXd &AI)
        {
            static const double sVmin = 1e-12;
            auto SVDResult = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
            // auto SVDResult = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

            AI = SVDResult.solve(Eigen::MatrixXd::Identity(A.rows(), A.rows()));
            // auto sVs = SVDResult.singularValues();
            // auto sVsMax = SVDResult.singularValues().array().abs().maxCoeff();
            // for (auto &i : sVs)
            //     if (std::fabs(i) > sVmin) //! note this filtering!
            //         i = 1. / i;
            //     else
            //         i = 0.;
            // AI = SVDResult.matrixV() * sVs.asDiagonal() * SVDResult.matrixU().transpose();

            // std::cout << AI * A << std::endl;
        }

        void EigenLeastSquareInverse_Filtered(const Eigen::MatrixXd &A, Eigen::MatrixXd &AI)
        {
            static const double sVmin = 1e-12;
            auto SVDResult = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
            // auto SVDResult = A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

            // AI = SVDResult.solve(Eigen::MatrixXd::Identity(A.rows(), A.rows()));
            auto sVs = SVDResult.singularValues();
            auto sVsMax = SVDResult.singularValues().array().abs().maxCoeff();
            for (auto &i : sVs)
                if (std::fabs(i) > sVmin) //! note this filtering!
                    i = 1. / i;
                else
                    i = 0.;
            AI = SVDResult.matrixV() * sVs.asDiagonal() * SVDResult.matrixU().transpose();

            // std::cout << AI * A << std::endl;
        }
    }
}