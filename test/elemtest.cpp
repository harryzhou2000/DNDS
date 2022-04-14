#include "../DNDS_Elements.hpp"

int main()
{
    DNDS::Elem::ElementManager::InitNBuffer();
    auto &nb = DNDS::Elem::ElementManager::NBuffer;

    auto etri6 = DNDS::Elem::ElementManager(DNDS::Elem::ElemType::Tri6, DNDS::Elem::INT_SCHEME_TRI_7);
    auto etri3 = DNDS::Elem::ElementManager(DNDS::Elem::ElemType::Tri3, DNDS::Elem::INT_SCHEME_TRI_7);
    auto equad4 = DNDS::Elem::ElementManager(DNDS::Elem::ElemType::Quad4, DNDS::Elem::INT_SCHEME_QUAD_16);
    auto equad9 = DNDS::Elem::ElementManager(DNDS::Elem::ElemType::Quad9, DNDS::Elem::INT_SCHEME_QUAD_16);
    Eigen::MatrixXd coordsTri6(3, 6), coordsQuad9(3, 9);
    coordsTri6 << 0, 1, 0.5, 0.5, 0.75, 0.25,
        0, 0, std::sqrt(3) * 0.5, 0, std::sqrt(3) * 0.25, std::sqrt(3) * 0.25,
        0, 0, 0, 0, 0, 0;
    coordsQuad9 << 0, 1, 1, 0, 0.5, 1, 0.5, 0, 0.5,
        0, 0, 1, 1, 0, 0.5, 1, 0.5, 0.5,
        0, 0, 0, 0, 0, 0, 0, 0, 0;
    double r = 1, pi = std::acos(-1);
    // coordsTri6 << std::cos(pi * 0. / 3.), std::cos(pi * 2. / 3.), std::cos(pi * 4. / 3.), std::cos(pi * 1. / 3.), std::cos(pi * 3. / 3.), std::cos(pi * 5. / 3.),
    //     std::sin(pi * 0. / 3.), std::sin(pi * 2. / 3.), std::sin(pi * 4. / 3.), std::sin(pi * 1. / 3.), std::sin(pi * 3. / 3.), std::sin(pi * 5. / 3.),
    //     0, 0, 0, 0, 0, 0;
    coordsQuad9 << std::cos(pi * 0. / 4.), std::cos(pi * 2. / 4.), std::cos(pi * 4. / 4.), std::cos(pi * 6. / 4.),
        std::cos(pi * 1. / 4.), std::cos(pi * 3. / 4.), std::cos(pi * 5. / 4.), std::cos(pi * 7. / 4.), 0,
        std::sin(pi * 0. / 4.), std::sin(pi * 2. / 4.), std::sin(pi * 4. / 4.), std::sin(pi * 6. / 4.),
        std::sin(pi * 1. / 4.), std::sin(pi * 3. / 4.), std::sin(pi * 5. / 4.), std::sin(pi * 7. / 4.), 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0;
    std::cout << "COORDS\n"
              << coordsTri6 << std::endl;

    Eigen::MatrixXd coordsTri3 = coordsTri6(Eigen::all, {0, 1, 2});
    Eigen::MatrixXd coordsQuad4 = coordsQuad9(Eigen::all, {0, 1, 2, 3});

    {
        double v = 0.0;
        auto &coords = coordsTri6;
        etri6.Integration(v, [&](double &vinc, int m, DNDS::Elem::tPoint &p, DNDS::Elem::tDiFj &DiNj) -> void
                          { 
                    vinc = DNDS::Elem::DiNj2Jacobi(DiNj, coords)({0,1},{0,1}).determinant();
                    std::cout << DNDS::Elem::DiNj2Jacobi(DiNj, coords)({0, 1}, {0, 1}).inverse() << std::endl; });
        std::cout << "VOL = " << v << std::endl;
    }
    {
        double v = 0.0;
        auto &coords = coordsTri3;
        etri3.Integration(v, [&](double &vinc, int m, DNDS::Elem::tPoint &p, DNDS::Elem::tDiFj &DiNj) -> void
                          { vinc = DNDS::Elem::DiNj2Jacobi(DiNj, coords)({0, 1}, {0, 1}).determinant(); });
        std::cout << "VOL = " << v << std::endl;
    }
    {
        double v = 0.0;
        auto &coords = coordsQuad9;
        equad9.Integration(v, [&](double &vinc, int m, DNDS::Elem::tPoint &p, DNDS::Elem::tDiFj &DiNj) -> void
                           { vinc = DNDS::Elem::DiNj2Jacobi(DiNj, coords)({0, 1}, {0, 1}).determinant(); });
        std::cout << "VOL = " << v << std::endl;
    }
    {
        double v = 0.0;
        auto &coords = coordsQuad4;
        equad4.Integration(v, [&](double &vinc, int m, DNDS::Elem::tPoint &p, DNDS::Elem::tDiFj &DiNj) -> void
                           { vinc = DNDS::Elem::DiNj2Jacobi(DiNj, coords)({0, 1}, {0, 1}).determinant(); });
        std::cout << "VOL = " << v << std::endl;
    }

    return 0;
}