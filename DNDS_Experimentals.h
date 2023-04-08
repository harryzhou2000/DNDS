#pragma once
#include <string>



// #define USE_NORM_FUNCTIONAL
// #define USE_LOCAL_COORD_CURVILINEAR
// #define PRINT_EVERY_VR_JACOBI_ITER_INCREMENT

static const std::string DNDS_Experimentals_State = std::string("DNDS_Experimentals ")
#ifdef USE_NORM_FUNCTIONAL
                                                    + " USE_NORM_FUNCTIONAL "
#endif
#ifdef USE_LOCAL_COORD_CURVILINEAR
                                                    + " USE_LOCAL_COORD_CURVILINEAR "
#endif
#ifdef PRINT_EVERY_VR_JACOBI_ITER_INCREMENT
                                                    + " PRINT_EVERY_VR_JACOBI_ITER_INCREMENT "
#endif
    ;