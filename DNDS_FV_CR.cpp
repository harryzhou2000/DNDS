#include "DNDS_FV_CR.hpp"

void DNDS::CRFiniteVolume2D::Reconstruction(ArrayCascadeLocal<VecStaticBatch<1>> &u,
                                            ArrayCascadeLocal<SemiVarMatrix<1>> &uRec, ArrayCascadeLocal<SemiVarMatrix<1>> &uRecCR)
{
    Reconstruction<1>(u, uRec, uRecCR);
}


void DNDS::CRFiniteVolume2D::Initialization()
{
    initReconstructionMatVec();
}