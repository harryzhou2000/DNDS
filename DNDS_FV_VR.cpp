#include "DNDS_FV_VR.hpp"

void DNDS::VRFiniteVolume2D::Initialization()
{
    initIntScheme();
    initMoment();
    initBaseDiffCache();
    initReconstructionMatVec();
}

void DNDS::VRFiniteVolume2D::ReconstructionJacobiStep(ArrayCascadeLocal<VecStaticBatch<1>> &u,
                                                      ArrayCascadeLocal<SemiVarMatrix<1>> &uRec,
                                                      ArrayCascadeLocal<SemiVarMatrix<1>> &uRecNewBuf)
{
    ReconstructionJacobiStep<1>(u, uRec, uRecNewBuf);
    // std::cout << __FUNCTION__ << std::endl;
}
