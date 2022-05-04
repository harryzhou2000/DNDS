#pragma once
#include "Eigen/Dense"

namespace DNDS
{
    namespace HardEigen
    {
        void EigenLeastSquareInverse(const Eigen::MatrixXd &A, Eigen::MatrixXd &AI);
        void EigenLeastSquareInverse_Filtered(const Eigen::MatrixXd &A, Eigen::MatrixXd &AI);

        Eigen::Matrix3d Eigen3x3RealSymEigenDecomposition(const Eigen::Matrix3d &A);
    }
}