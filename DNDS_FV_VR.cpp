#include "DNDS_FV_VR.hpp"

void DNDS::VRFiniteVolume2D::Initialization()
{
    initIntScheme();
    initMoment();
    initBaseDiffCache();
    initReconstructionMatVec();
}

void DNDS::VRFiniteVolume2D::ReconstructionJacobiStep(ArrayLocal<VecStaticBatch<1>> &u,
                                                      ArrayLocal<SemiVarMatrix<1>> &uRec,
                                                      ArrayLocal<SemiVarMatrix<1>> &uRecNewBuf)
{
    ReconstructionJacobiStep<1>(u, uRec, uRecNewBuf);
    // std::cout << __FUNCTION__ << std::endl;
}
