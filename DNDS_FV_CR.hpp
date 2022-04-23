#pragma once

#include "DNDS_FV_VR.hpp"

namespace DNDS
{
    class CRFiniteVolume2D : public VRFiniteVolume2D
    {
    public:
        using VRFiniteVolume2D::VRFiniteVolume2D;
        // CompactFacedMeshSerialRW *mesh = nullptr; // a mere reference to mesh, user responsible for keeping it valid
        // ImplicitFiniteVolume2D *FV = nullptr;
        VRFiniteVolume2D *VFV = nullptr;

        CRFiniteVolume2D(VRFiniteVolume2D &r) : VRFiniteVolume2D(r)
        {
            VFV = &r;
        }

        void setPtrs(CompactFacedMeshSerialRW *nMesh, ImplicitFiniteVolume2D *nFV, VRFiniteVolume2D *nVFV)
        {
            mesh = nMesh, FV = nFV, VFV = nVFV;
        }

        // TODO: FDiffBaseValue FFaceFunctional initIntScheme : overwrite

        // currently use the same set of basis
        void initReconstructionMatVec()
        {

            // InsertCheck(mpi, "initReconstructionMatVec _ CR");

            vectorInvAb = std::make_shared<decltype(vectorInvAb)::element_type>(mesh->cell2faceLocal.dist->size());
            // matrixInvAB = std::make_shared<decltype(matrixInvAB)::element_type>(mesh->cell2faceLocal.dist->size());
            // only needs one vector

            // for each inner cell (ghost cell no need)
            forEachInArray(
                *mesh->cell2faceLocal.dist,
                [&](tAdjArrayCascade::tComponent &c2f, index iCell)
                {
                    auto &cellAttribute = mesh->cellAtrLocal[iCell][0];
                    auto &cellRecAttribute = cellRecAtrLocal[iCell][0];
                    auto eCell = Elem::ElementManager(cellAttribute.type, cellAttribute.intScheme);
                    assert(c2f.size() == eCell.getNFace());
                    (*vectorInvAb)[iCell].resize(cellRecAttribute.NDOF - 1, cellRecAttribute.NDOF - 1);
                    (*vectorInvAb)[iCell].setZero(); //! which stores AiiCR
                    // InsertCheck(mpi, "initReconstructionMatVec _ CR B2");
                    // get Aii
                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++) // for each face of cell
                    {
                        index iFace = c2f[ic2f];
                        auto f2c = (*mesh->face2cellPair)[iFace];
                        index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                        index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                        auto &faceAttribute = mesh->faceAtrLocal[iFace][0];
                        auto &faceRecAttribute = faceRecAtrLocal[iFace][0];
                        Elem::ElementManager eFace(faceAttribute.type, faceRecAttribute.intScheme);
                        // InsertCheck(mpi, "initReconstructionMatVec _ CR B3");
                        eFace.Integration(
                            (*vectorInvAb)[iCell],
                            [&](Eigen::MatrixXd &incA, int ig, Elem::tPoint &ip, Elem::tDiFj &iDiNj)
                            {
                                Eigen::MatrixXd &diffsI = faceDiBjGaussCache[iFace][ig * 2 + iCellAtFace];
                                Eigen::MatrixXd incAFull;
                                FFaceFunctional(diffsI.topRows(1), diffsI.topRows(1), (*faceWeights)[iFace].topRows(1), incAFull);
                                assert(incAFull(Eigen::all, 0).norm() + incAFull(0, Eigen::all).norm() == 0);
                                incA = incAFull.bottomRightCorner(incAFull.rows() - 1, incAFull.cols() - 1);
                                incA *= faceNorms[iFace][ig].norm(); // note: don't forget the Jacobi!!!
                                // std::cout << "DI " << std::endl;
                                // std::cout << faceDiBjGaussCache[iFace][ig * 2 + iCellFace] << std::endl;
                            });
                    }
                    Eigen::MatrixXd Ainv;
                    HardEigen::EigenLeastSquareInverse((*vectorInvAb)[iCell], Ainv);
                    (*vectorInvAb)[iCell] = Ainv;
                });
            // InsertCheck(mpi, "initReconstructionMatVec _ CR END");
        }

        template <uint32_t vsize>
        // static const int vsize = 1;
        void Reconstruction(ArrayCascadeLocal<VecStaticBatch<vsize>> &u,
                            ArrayCascadeLocal<SemiVarMatrix<vsize>> &uRec, ArrayCascadeLocal<SemiVarMatrix<vsize>> &uRecCR)
        {
            // InsertCheck(mpi, "ReconstructionJacobiStep Start");
            // forEachInArray(
            //     *uRec.dist,
            //     [&](typename decltype(uRec.dist)::element_type::tComponent &uRecE, index iCell)
            //     {
            for (index iCell = 0; iCell < uRec.dist->size(); iCell++)
            {
                auto &uRecE = uRec[iCell];
                auto &cellRA = cellRecAtrLocal[iCell][0];
                real relax = cellRA.relax;
                auto c2f = mesh->cell2faceLocal[iCell];
                int NDOFCell = cellRA.NDOF;
                Eigen::VectorXd bi(NDOFCell - 1);
                bi.setZero();

                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    // this is a repeated code block START
                    index iFace = c2f[ic2f];
                    auto f2c = (*mesh->face2cellPair)[iFace];
                    index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                    index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                    auto &faceAttribute = mesh->faceAtrLocal[iFace][0];
                    auto &faceRecAttribute = faceRecAtrLocal[iFace][0];
                    Elem::ElementManager eFace(faceAttribute.type, faceRecAttribute.intScheme);
                    Eigen::MatrixXd fcoords;
                    mesh->LoadCoords(mesh->face2nodeLocal[iFace], fcoords);
                    // Elem::tPoint faceL = fcoords(Eigen::all, 1) - fcoords(Eigen::all, 0);
                    // Elem::tPoint faceN{faceL(1), -faceL(0), 0.0}; // 2-d specific //pointing from left to right
                    Elem::tPoint faceN = faceNormCenter[iFace];

                    Elem::tPoint gradL{0, 0, 0};
                    gradL({0, 1}) = iCellAtFace ? faceDiBjCenterCache[iFace].second({1, 2}, Eigen::all).rightCols(NDOFCell - 1) * uRec[iCell].m()
                                                : faceDiBjCenterCache[iFace].first({1, 2}, Eigen::all).rightCols(NDOFCell - 1) * uRec[iCell].m();

                    if (iCellOther != FACE_2_VOL_EMPTY)
                    {
                        auto &cellRAOther = cellRecAtrLocal[iCellOther][0];

                        Elem::tPoint gradR{0, 0, 0}; // ! 2d here!!
                        gradR({0, 1}) = iCellAtFace ? faceDiBjCenterCache[iFace].first({1, 2}, Eigen::all).rightCols(cellRAOther.NDOF - 1) * uRec[iCellOther].m()
                                                    : faceDiBjCenterCache[iFace].second({1, 2}, Eigen::all).rightCols(cellRAOther.NDOF - 1) * uRec[iCellOther].m();
                        Elem::tPoint convVelocity = 0.5 * (gradL + gradR);
                        real signLR = ((convVelocity.dot(faceN) > 0) == (iCellAtFace == 0)) ? 1.0 : -1.0;
                        eFace.Integration(
                            bi,
                            [&](Eigen::VectorXd &biInc, int ig, Elem::tPoint pParam, Elem::tDiFj &DiNj)
                            {
                                Eigen::MatrixXd FFace;
                                if (signLR > 0)
                                    FFace = faceDiBjGaussCache[iFace][ig * 2 + iCellAtFace].row(0).rightCols(NDOFCell - 1) * uRec[iCell].m();
                                else
                                    FFace = faceDiBjGaussCache[iFace][ig * 2 + 1 - iCellAtFace].row(0).rightCols(NDOFCell - 1) * uRec[iCellOther].m() + u[iCellOther].p() - u[iCell].p();
                                //! don't forget the mean value between them
                                biInc = (faceDiBjGaussCache[iFace][ig * 2 + iCellAtFace].row(0).rightCols(NDOFCell - 1).transpose()) *
                                        FFace * std::pow((*faceWeights)[iFace][0], 2);
                                biInc *= faceNorms[iFace][ig].norm();
                            });
                    }
                    else
                    {
                        eFace.Integration(
                            bi,
                            [&](Eigen::VectorXd &biInc, int ig, Elem::tPoint pParam, Elem::tDiFj &DiNj)
                            {
                                Eigen::MatrixXd FFace;
                                if (faceAttribute.iPhy == BoundaryType::Wall)
                                {
                                    FFace.resizeLike(faceDiBjGaussCache[iFace][ig * 2 + iCellAtFace].row(0).rightCols(cellRA.NDOF - 1) * uRec[iCell].m());
                                    FFace.setZero();
                                }
                                else if (faceAttribute.iPhy == BoundaryType::Farfield)
                                {
                                    FFace = faceDiBjGaussCache[iFace][ig * 2 + iCellAtFace].row(0).rightCols(cellRA.NDOF - 1) * uRec[iCell].m();
                                }
                                else
                                {
                                    assert(false);
                                }
                                biInc = (faceDiBjGaussCache[iFace][ig * 2 + iCellAtFace].row(0).rightCols(cellRA.NDOF - 1).transpose()) *
                                        FFace * std::pow((*faceWeights)[iFace][0], 2);
                                biInc *= faceNorms[iFace][ig].norm();
                            });
                    }
                }

                uRecCR[iCell].m() = (*vectorInvAb)[iCell] * bi;
                // std::cout << "DIFF\n"
                //           << uRecNewBuf[iCell].m() << std::endl;
            }
            // );
        }

        void Initialization();

        // static const int vsize = 1;
        void Reconstruction(ArrayCascadeLocal<VecStaticBatch<1>> &u,
                            ArrayCascadeLocal<SemiVarMatrix<1>> &uRec, ArrayCascadeLocal<SemiVarMatrix<1>> &uRecCR);
    };
}