#include <iostream>
#include "../../Eigen/Dense"

void testF(Eigen::Ref<Eigen::MatrixXd> a)
{
    std::cout << "Ref Receive\n"
              << a << std::endl;
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

    return 0;
}