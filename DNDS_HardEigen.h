#pragma once
#include "Eigen/Dense"

namespace DNDS
{
    namespace HardEigen
    {
        void EigenLeastSquareInverse(const Eigen::MatrixXd &A, Eigen::MatrixXd &AI);
    }
}