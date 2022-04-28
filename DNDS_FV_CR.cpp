#include "DNDS_FV_CR.hpp"

void DNDS::CRFiniteVolume2D::Reconstruction(ArrayLocal<VecStaticBatch<1>> &u,
                                            ArrayLocal<SemiVarMatrix<1>> &uRec, ArrayLocal<SemiVarMatrix<1>> &uRecCR)
{
    Reconstruction<1>(u, uRec, uRecCR);
}


void DNDS::CRFiniteVolume2D::Initialization()
{
    initReconstructionMatVec();
}