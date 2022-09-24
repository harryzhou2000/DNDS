#include <iostream>
#include "../DNDS_Linear.hpp"
#include "../Eigen/Dense"

int main()
{
    int N = 20;
    using namespace DNDS;
    class Vec : public Eigen::VectorXd
    {
    public:
        // += *= = ok
        using Eigen::VectorXd::VectorXd;
        void addTo(const Eigen::VectorXd &R, DNDS::real a)
        {
            (*this) += R * a;
        }

        DNDS::real norm2()
        {
            return (*this).norm();
        }
    } x, b;
    x.resize(N), b.resize(N);
    Eigen::MatrixXd A;
    A.resize(N, N);
    A.setZero();
    for (int i = 0; i < N; i++)
    {
        if (i > 0)
            A(i, i - 1) = 14;
        A(i, i) = 2 * (i + 1);
        if (i < (N - 1))
            A(i, i + 1) = 1.5;
    } // condition number~2e3
    // A *= 1;
    // A.setRandom();
    // A += Eigen::MatrixXd::Identity(N,N);
    b = A * Eigen::VectorXd::Ones(N);
    x.setZero();
    std::cout << A << std::endl
              << std::endl;
    std::cout << b.transpose() << std::endl
              << std::endl;
    std::cout << x.transpose() << std::endl
              << std::endl;
    

    Linear::GMRES_LeftPreconditioned<Vec> gmres(3, [&](Vec &v)
                                                { v.resize(N); });
    gmres.solve([&](Vec &x, Vec &Ax)
                { Ax = A * x; },
                [&](Vec &x, Vec &MLx)
                {
                    MLx = x;
                    for (int i = 0; i < MLx.size(); i++) // ! preconditioning by diagonal
                        MLx(i) /= A(i,i); 
                },
                b, x, 10,
                [&](uint32_t iRestart, DNDS::real res, DNDS::real resB)
                {
                    std::cout << iRestart << " Res: " << res / resB << std::endl;
                    if (res / resB < 1e-14)
                        return true;
                    return false;
                });

    std::cout << x.transpose() << std::endl;

    return 0;
}