#include <iostream>
#include "../../Eigen/Dense"

// g++ -o testEigen.exe testEigen.cpp
void testF(Eigen::Ref<Eigen::MatrixXd> a)
{
    std::cout << "Ref Receive\n"
              << a << std::endl;
}

template <int a = 1, class T1, class T2>
void testTempF(T1 m, T2 n)
{
    std::cout << a << std::endl;
}

int main()
{
    Eigen::MatrixXd a{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto b = (a.array() * -2).matrix();
    auto d = a.array().max(b.array());
    std::cout << "a = \n"
              << a << std::endl;
    std::cout << "a.array().inverse() = \n"
              << a.array().max(b.array()) << std::endl;
    // auto aa = Eigen::MatrixXd::Random(3,3);
    // testF(aa);

    Eigen::MatrixXd Mres{{1, 0, -1}};
    std::cout << Mres.array().pow(1.0) << std::endl;

    std::string t = "test";
    std::cout << "t size = " << t.size() << std::endl;
    for (auto c : t)
        std::cout << int(c) << std::endl;

    auto v = Eigen::Vector<double, 10>::LinSpaced(0, 9);
    std::cout << "reshape col " << std::endl
              << v.reshaped<Eigen::ColMajor>(2, 5) << std::endl;

    Eigen::Matrix<double, 3, 1> A(Eigen::Index(3), Eigen::Index(1));
    std::cout << A << std::endl;
    A.setZero();
    std::cout << A << std::endl;

    testTempF(1, 2);

    testTempF<2>(1., 2);

    testTempF<3>(1, 3.);

    
    a.diagonal().setConstant(123);
    std::cout << a << std::endl;

    return 0;
}