#include "../DNDS_Elements.hpp"

int main()
{
    DNDS::Elem::ElementManager::InitNBuffer();
    auto &nb = DNDS::Elem::ElementManager::NBuffer;

    auto e = DNDS::Elem::ElementManager(DNDS::Elem::ElemType::Tri6, DNDS::Elem::INT_SCHEME_TRI_7);
    Eigen::MatrixXd coords(3, 6);
    coords << 0, 1, 0.5, 0.5, 0.75, 0.25,
        0, 0, std::sqrt(3) * 0.5, 0, std::sqrt(3) * 0.25, std::sqrt(3) * 0.25,
        0, 0, 0, 0, 0, 0;
    std::cout << "COORDS\n"
              << coords << std::endl;
    double v = 0.0;
    e.Integration(v, [&](double &vinc, int m, DNDS::Elem::tPoint &p, DNDS::Elem::tDiFj &DiNj) -> void
                  { 
                    vinc = DNDS::Elem::DiNj2Jacobi(DiNj, coords)({0,1},{0,1}).determinant();
                    std::cout << DNDS::Elem::DiNj2Jacobi(DiNj, coords)({0, 1}, {0, 1}).inverse() << std::endl; });
    std::cout << "VOL = " << v << std::endl;
    return 0;
}