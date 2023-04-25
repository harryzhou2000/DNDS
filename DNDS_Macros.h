#pragma once
#include <string>

// #define EIGEN_USE_BLAS
// #define EIGEN_USE_LAPACKE_STRICT

static const std::string DNDS_Macros_State = std::string("DNDS_Macros ")
#ifdef EIGEN_USE_BLAS
                                             + " EIGEN_USE_BLAS "
#endif
#ifdef EIGEN_USE_LAPACKE_STRICT
                                             + " EIGEN_USE_LAPACKE_STRICT "
#endif
    ;
