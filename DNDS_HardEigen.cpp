#include "DNDS_HardEigen.h"

namespace DNDS
{
    namespace HardEigen
    {
        void EigenLeastSquareInverse(const Eigen::MatrixXd &A, Eigen::MatrixXd &AI)
        {
            auto SVDResult = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
            AI = SVDResult.solve(Eigen::MatrixXd::Identity(A.rows(), A.rows()));
        }
    }
}