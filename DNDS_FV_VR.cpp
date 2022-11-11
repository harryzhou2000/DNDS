#include "DNDS_FV_VR.hpp"

void DNDS::VRFiniteVolume2D::Initialization()
{
    SOR_InitRedBlack();
    initIntScheme();
#ifdef USE_LOCAL_COORD_CURVILINEAR
    initUcurve(); // needed before using FDiffBase
#endif
    initMoment();
    initBaseDiffCache();
    initReconstructionMatVec();
}
void DNDS::VRFiniteVolume2D::Initialization_RenewBase()
{
    initMoment();
    initBaseDiffCache();
    initReconstructionMatVec();
}

