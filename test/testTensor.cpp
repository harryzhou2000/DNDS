#include "../SmallTensor.hpp"
#include "../Eigen/Dense"

int main()
{
    using namespace SmallTensor;
    Tensor<double, 1, 2, 3> A;
    for (int i = 0; i < A.order(); i++)
        std::cout << A.dim(i) << "\t";
    std::cout << std::endl;
    std::cout << A.size() << std::endl;

    Eigen::Tensor<double, 3> e1(3, 3, 3);
    e1.setZero();
    e1(0, 1, 2) = 1;
    e1(1, 2, 0) = 1;
    e1(2, 0, 1) = 1;
    e1(1, 0, 2) = -1;
    e1(2, 1, 0) = -1;
    e1(0, 2, 1) = -1;
    Eigen::Tensor<double, 2> e2(3, 3);
    e2.setValues({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    Eigen::array<Eigen::IndexPair<int>, 1> transposed_product_dims = {Eigen::IndexPair<int>(0, 1)};
    // Eigen::Tensor<double, 3> e3 = e1.contract(e2, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 1)});

    // Eigen::Vector3d v1{1, 0, 0};
    // Eigen::Vector3d v2{0, 1, 0};

    Eigen::Tensor<double, 1> v1e(3);
    v1e.setValues({0.5, 0, 1.2});
    Eigen::Tensor<double, 1> v2e(3);
    v2e.setValues({0.76, 1.3, 0.2});

    std::cout << std::endl
              << e1.contract(v1e, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 0)})
                     .contract(v2e, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 0)})
              << std::endl;

    std::cout << std::endl
              << "SIZE " << sizeof(Eigen::TensorFixedSize<double, Eigen::Sizes<1, 2, 3, 4>>)
              << std::endl;
    return 0;
}