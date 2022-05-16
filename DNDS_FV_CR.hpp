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

        CRFiniteVolume2D(VRFiniteVolume2D &r) : VRFiniteVolume2D(r) // calls the copy ctor of vfv, copying everyting already in
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

            // allocate for vector and matrix batch

            auto fGetMatrixSize = [&](int &nmats, std::vector<int> &matSizes, index iCell)
            {
                auto c2f = mesh->cell2faceLocal[iCell];
                auto &cellAttribute = mesh->cellAtrLocal[iCell][0];
                auto &cellRecAttribute = cellRecAtrLocal[iCell][0];
                auto &cellVRRecAttribute = VFV->cellRecAtrLocal[iCell][0];
                auto eCell = Elem::ElementManager(cellAttribute.type, cellAttribute.intScheme);
                nmats = eCell.getNFace() * 2 + 1;
                matSizes.resize(nmats * 2);
                matSizes[0 * 2 + 0] = cellRecAttribute.NDOF - 1;
                matSizes[0 * 2 + 1] = cellRecAttribute.NDOF - 1;
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++) // for each face of cell
                {
                    index iFace = c2f[ic2f];
                    auto f2c = mesh->face2cellLocal[iFace];
                    index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                    auto &faceAttribute = mesh->faceAtrLocal[iFace][0];

                    matSizes[(1 + ic2f * 2 + 0) * 2 + 0] = cellVRRecAttribute.NDOF - 1;
                    matSizes[(1 + ic2f * 2 + 0) * 2 + 1] = cellRecAttribute.NDOF - 1;

                    if (iCellOther != FACE_2_VOL_EMPTY)
                    {
                        auto &cellVRRecAttributeOther = VFV->cellRecAtrLocal[iCellOther][0];
                        matSizes[(1 + ic2f * 2 + 1) * 2 + 0] = cellVRRecAttributeOther.NDOF - 1;
                        matSizes[(1 + ic2f * 2 + 1) * 2 + 1] = cellRecAttribute.NDOF - 1;
                    }
                    else if (faceAttribute.iPhy == BoundaryType::Wall || faceAttribute.iPhy == BoundaryType::Farfield)
                    {
                        matSizes[(1 + ic2f * 2 + 1) * 2 + 0] = cellRecAttribute.NDOF - 1;
                        matSizes[(1 + ic2f * 2 + 1) * 2 + 1] = 1;
                    }
                    else
                    {
                        assert(false);
                    }
                }
            };

            auto fGetVectorSize = [&](int &nmats, std::vector<int> &matSizes, index iCell)
            {
                auto c2f = mesh->cell2faceLocal[iCell];
                auto &cellAttribute = mesh->cellAtrLocal[iCell][0];
                auto &cellRecAttribute = cellRecAtrLocal[iCell][0];
                auto eCell = Elem::ElementManager(cellAttribute.type, cellAttribute.intScheme);
                nmats = 1;
                matSizes.resize(nmats * 2);
                matSizes[0 * 2 + 0] = cellRecAttribute.NDOF - 1;
                matSizes[0 * 2 + 1] = eCell.getNFace();
            };

            matrixBatch = std::make_shared<decltype(matrixBatch)::element_type>(
                decltype(matrixBatch)::element_type::tContext(
                    [&](index i) -> rowsize
                    {
                        int nmats = -123;
                        std::vector<int> matSizes;
                        fGetMatrixSize(nmats, matSizes, i);
                        assert(matSizes.size() == nmats * 2);
                        return DNDS::SmallMatricesBatch::predictSize(nmats, matSizes);
                    },
                    [&](uint8_t *data, index siz, index i)
                    {
                        int nmats = -123;
                        std::vector<int> matSizes;
                        fGetMatrixSize(nmats, matSizes, i);
                        assert(matSizes.size() == nmats * 2);
                        DNDS::SmallMatricesBatch::initializeData(data, nmats, matSizes);
                    },
                    mesh->cell2faceLocal.dist->size()),
                mpi);

            vectorBatch = std::make_shared<decltype(vectorBatch)::element_type>(
                decltype(vectorBatch)::element_type::tContext(
                    [&](index i) -> rowsize
                    {
                        int nmats = -123;
                        std::vector<int> matSizes;
                        fGetVectorSize(nmats, matSizes, i);
                        assert(matSizes.size() == nmats * 2);
                        return DNDS::SmallMatricesBatch::predictSize(nmats, matSizes);
                    },
                    [&](uint8_t *data, index siz, index i)
                    {
                        int nmats = -123;
                        std::vector<int> matSizes;
                        fGetVectorSize(nmats, matSizes, i);
                        assert(matSizes.size() == nmats * 2);
                        DNDS::SmallMatricesBatch::initializeData(data, nmats, matSizes);
                    },
                    mesh->cell2faceLocal.dist->size()),
                mpi);

            // only needs one mat for rec but two for each face's b

            // for each inner cell (ghost cell no need)
            forEachInArray(
                *mesh->cell2faceLocal.dist,
                [&](tAdjArray::tComponent &c2f, index iCell)
                {
                    auto &cellAttribute = mesh->cellAtrLocal[iCell][0];
                    auto &cellRecAttribute = cellRecAtrLocal[iCell][0];
                    auto &cellVRRecAttribute = VFV->cellRecAtrLocal[iCell][0];
                    auto eCell = Elem::ElementManager(cellAttribute.type, cellAttribute.intScheme);
                    assert(c2f.size() == eCell.getNFace());

                    auto matrixBatchElem = (*matrixBatch)[iCell];
                    auto vectorBatchElem = (*vectorBatch)[iCell];

                    // InsertCheck(mpi, "initReconstructionMatVec _ CR B2");
                    // get Aii
                    Eigen::MatrixXd A;
                    A.resizeLike(matrixBatchElem.m(0));
                    A.setZero();
                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++) // for each face of cell
                    {
                        index iFace = c2f[ic2f];
                        auto f2c = mesh->face2cellLocal[iFace];
                        index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                        index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                        auto &faceAttribute = mesh->faceAtrLocal[iFace][0];
                        auto &faceRecAttribute = faceRecAtrLocal[iFace][0];
                        auto faceDiBjGaussBatchElem = (*faceDiBjGaussBatch)[iFace]; // ! which is CR basis, not necessarily identical with VR
                        auto faceDiBjGaussBatchElemVR = (*VFV->faceDiBjGaussBatch)[iFace];

                        Elem::ElementManager eFace(faceAttribute.type, faceRecAttribute.intScheme);

                        eFace.Integration(
                            A,
                            [&](Eigen::MatrixXd &incA, int ig, Elem::tPoint &ip, Elem::tDiFj &iDiNj)
                            {
                                Eigen::MatrixXd diffsI = faceDiBjGaussBatchElem.m(ig * 2 + iCellAtFace);
                                Eigen::MatrixXd incAFull;
                                FFaceFunctional(iFace, ig, diffsI.topRows(1), diffsI.topRows(1), (*faceWeights)[iFace].topRows(1), incAFull);
                                assert(incAFull(Eigen::all, 0).norm() + incAFull(0, Eigen::all).norm() == 0);
                                incA = incAFull.bottomRightCorner(incAFull.rows() - 1, incAFull.cols() - 1);
                                incA *= faceNorms[iFace][ig].norm(); // note: don't forget the Jacobi!!!
                            });

                        Eigen::MatrixXd B;
                        B.resizeLike(matrixBatchElem.m(1 + ic2f * 2 + 0));
                        B.setZero();
                        eFace.Integration(
                            B,
                            [&](Eigen::MatrixXd &incA, int ig, Elem::tPoint &ip, Elem::tDiFj &iDiNj)
                            {
                                auto diffsI = faceDiBjGaussBatchElem.m(ig * 2 + iCellAtFace);
                                auto diffsVRI = faceDiBjGaussBatchElemVR.m(ig * 2 + iCellAtFace);
                                Eigen::MatrixXd incAFull;
                                FFaceFunctional(iFace, ig, diffsVRI.topRows(1), diffsI.topRows(1), (*faceWeights)[iFace].topRows(1), incAFull);
                                assert(incAFull(Eigen::all, 0).norm() + incAFull(0, Eigen::all).norm() == 0);
                                incA = incAFull.bottomRightCorner(incAFull.rows() - 1, incAFull.cols() - 1);
                                incA *= faceNorms[iFace][ig].norm(); // note: don't forget the Jacobi!!!
                            });
                        matrixBatchElem.m(1 + ic2f * 2 + 0) = B;

                        Eigen::MatrixXd col(cellRecAttribute.NDOF - 1, 1);
                        col.setZero();
                        eFace.Integration(
                            col,
                            [&](Eigen::MatrixXd &incA, int ig, Elem::tPoint &ip, Elem::tDiFj &iDiNj)
                            {
                                auto diffsI = faceDiBjGaussBatchElem.m(ig * 2 + iCellAtFace);
                                Eigen::MatrixXd incAFull;
                                FFaceFunctional(iFace, ig, diffsI.topRows(1), Eigen::MatrixXd::Ones(1, 1), (*faceWeights)[iFace].topRows(1), incAFull);
                                assert(incAFull(0) == 0);
                                incA = incAFull.bottomRows(incAFull.rows() - 1);
                                incA *= faceNorms[iFace][ig].norm(); // note: don't forget the Jacobi!!!
                            });
                        vectorBatchElem.m(0)(Eigen::all, ic2f) = col;

                        if (iCellOther != FACE_2_VOL_EMPTY)
                        {
                            auto &cellVRRecAttributeOther = VFV->cellRecAtrLocal[iCellOther][0];
                            Eigen::MatrixXd B;
                            B.resizeLike(matrixBatchElem.m(1 + ic2f * 2 + 1));
                            B.setZero();
                            eFace.Integration(
                                B,
                                [&](Eigen::MatrixXd &incA, int ig, Elem::tPoint &ip, Elem::tDiFj &iDiNj)
                                {
                                    auto diffsI = faceDiBjGaussBatchElem.m(ig * 2 + iCellAtFace);
                                    auto diffsVRI = faceDiBjGaussBatchElemVR.m(ig * 2 + 1 - iCellAtFace);
                                    Eigen::MatrixXd incAFull;
                                    FFaceFunctional(iFace, ig, diffsI.topRows(1), diffsVRI.topRows(1), (*faceWeights)[iFace].topRows(1), incAFull);
                                    assert(incAFull(Eigen::all, 0).norm() + incAFull(0, Eigen::all).norm() == 0);
                                    incA = incAFull.bottomRightCorner(incAFull.rows() - 1, incAFull.cols() - 1);
                                    incA *= faceNorms[iFace][ig].norm(); // note: don't forget the Jacobi!!!
                                });
                            matrixBatchElem.m(1 + ic2f * 2 + 1) = B;
                        }
                        else if (faceAttribute.iPhy == BoundaryType::Wall || faceAttribute.iPhy == BoundaryType::Farfield)
                        {
                            Eigen::MatrixXd B;
                            B.resizeLike(matrixBatchElem.m(1 + ic2f * 2 + 1));
                            B.setZero();
                            eFace.Integration(
                                B,
                                [&](Eigen::MatrixXd &incA, int ig, Elem::tPoint &ip, Elem::tDiFj &iDiNj)
                                {
                                    auto diffsI = faceDiBjGaussBatchElem.m(ig * 2 + iCellAtFace);
                                    Eigen::MatrixXd incAFull;
                                    FFaceFunctional(iFace, ig, diffsI.topRows(1), Eigen::MatrixXd::Ones(1, 1), (*faceWeights)[iFace].topRows(1), incAFull);
                                    assert(incAFull(0) == 0);
                                    incA = incAFull.bottomRows(incAFull.rows() - 1);
                                    incA *= faceNorms[iFace][ig].norm(); // note: don't forget the Jacobi!!!
                                });
                            matrixBatchElem.m(1 + ic2f * 2 + 1) = B;
                        }
                        else
                        {
                            assert(false);
                        }
                    }
                    Eigen::MatrixXd Ainv, AinvFilterd;
                    // HardEigen::EigenLeastSquareInverse(A, Ainv);
                    HardEigen::EigenLeastSquareInverse_Filtered(A, Ainv);

                    matrixBatchElem.m(0) = Ainv;
                    // exit(0);
                });
            // InsertCheck(mpi, "initReconstructionMatVec _ CR END");
        }

        template <uint32_t vsize>
        // static const int vsize = 1;
        void Reconstruction(ArrayLocal<VecStaticBatch<vsize>> &u,
                            ArrayLocal<SemiVarMatrix<vsize>> &uRec, ArrayLocal<SemiVarMatrix<vsize>> &uRecCR)
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
                auto &cellVRRA = VFV->cellRecAtrLocal[iCell][0];
                real relax = cellRA.relax;
                auto &c2f = mesh->cell2faceLocal[iCell];
                int NDOFCell = cellRA.NDOF;
                Eigen::VectorXd bi(NDOFCell - 1);
                bi.setZero();
                auto matrixBatchElem = (*matrixBatch)[iCell];
                auto vectorBatchElem = (*vectorBatch)[iCell];

                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    // this is a repeated code block START
                    index iFace = c2f[ic2f];
                    auto &f2c = (*mesh->face2cellPair)[iFace];
                    index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                    index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                    auto &faceAttribute = mesh->faceAtrLocal[iFace][0];
                    auto &faceRecAttribute = faceRecAtrLocal[iFace][0];
                    auto faceDiBjCenterBatchElemVR = (*VFV->faceDiBjCenterBatch)[iFace];
                    auto faceDiBjGaussBatchElemVR = (*VFV->faceDiBjGaussBatch)[iFace];

                    Elem::ElementManager eFace(faceAttribute.type, faceRecAttribute.intScheme);
                    Eigen::MatrixXd fcoords;
                    mesh->LoadCoords(mesh->face2nodeLocal[iFace], fcoords);
                    // Elem::tPoint faceL = fcoords(Eigen::all, 1) - fcoords(Eigen::all, 0);
                    // Elem::tPoint faceN{faceL(1), -faceL(0), 0.0}; // 2-d specific //pointing from left to right
                    Elem::tPoint faceN = faceNormCenter[iFace];

                    Elem::tPoint gradL{0, 0, 0};
                    // gradL({0, 1}) = iCellAtFace ? VFV->faceDiBjCenterCache[iFace].second({1, 2}, Eigen::all).rightCols(cellVRRA.NDOF - 1) * uRec[iCell].m()
                    //                             : VFV->faceDiBjCenterCache[iFace].first({1, 2}, Eigen::all).rightCols(cellVRRA.NDOF - 1) * uRec[iCell].m();
                    gradL({0, 1}) = faceDiBjCenterBatchElemVR.m(iCellAtFace)({1, 2}, Eigen::all).rightCols(cellVRRA.NDOF - 1) * uRec[iCell].m();

                    if (iCellOther != FACE_2_VOL_EMPTY)
                    {
                        auto &cellRAOther = cellRecAtrLocal[iCellOther][0];
                        auto &cellVRRAOther = VFV->cellRecAtrLocal[iCellOther][0];

                        Elem::tPoint gradR{0, 0, 0}; // ! 2d here!!
                        // gradR({0, 1}) = iCellAtFace ? VFV->faceDiBjCenterCache[iFace].first({1, 2}, Eigen::all).rightCols(cellVRRAOther.NDOF - 1) * uRec[iCellOther].m()
                        //                             : VFV->faceDiBjCenterCache[iFace].second({1, 2}, Eigen::all).rightCols(cellVRRAOther.NDOF - 1) * uRec[iCellOther].m();
                        gradR({0, 1}) = faceDiBjCenterBatchElemVR.m(1 - iCellAtFace)({1, 2}, Eigen::all).rightCols(cellVRRAOther.NDOF - 1) * uRec[iCellOther].m();

                        Elem::tPoint convVelocity = 0.5 * (gradL + gradR);
                        real signLR = ((convVelocity.dot(faceN) > 0) == (iCellAtFace == 0)) ? 1.0 : -1.0;
                        // eFace.Integration(
                        //     bi,
                        //     [&](Eigen::VectorXd &biInc, int ig, Elem::tPoint pParam, Elem::tDiFj &DiNj)
                        //     {
                        //         Eigen::MatrixXd FFace;
                        //         if (signLR > 0)
                        //             FFace = VFV->faceDiBjGaussCache[iFace][ig * 2 + iCellAtFace].row(0).rightCols(cellVRRA.NDOF - 1) * uRec[iCell].m();
                        //         else
                        //             FFace = VFV->faceDiBjGaussCache[iFace][ig * 2 + 1 - iCellAtFace].row(0).rightCols(cellVRRAOther.NDOF - 1) * uRec[iCellOther].m() + u[iCellOther].p() - u[iCell].p();
                        //         //! don't forget the mean value between them
                        //         biInc = (faceDiBjGaussCache[iFace][ig * 2 + iCellAtFace].row(0).rightCols(cellRA.NDOF - 1).transpose()) *
                        //                 FFace * std::pow((*faceWeights)[iFace][0], 2);
                        //         biInc *= faceNorms[iFace][ig].norm();
                        //     });
                        if (signLR > 0)
                            bi += matrixBatchElem.m(1 + ic2f * 2 + 0) * uRec[iCell].m();
                        else
                            bi += matrixBatchElem.m(1 + ic2f * 2 + 1) * uRec[iCellOther].m() + vectorBatchElem.m(0)(Eigen::all, ic2f) * (u[iCellOther].p() - u[iCell].p());
                        //! don't forget the mean value between them
                    }
                    else
                    {
                        // eFace.Integration(
                        //     bi,
                        //     [&](Eigen::VectorXd &biInc, int ig, Elem::tPoint pParam, Elem::tDiFj &DiNj)
                        //     {
                        //         Eigen::MatrixXd FFace;
                        //         if (faceAttribute.iPhy == BoundaryType::Wall)
                        //         {
                        //             FFace.resizeLike(VFV->faceDiBjGaussCache[iFace][ig * 2 + iCellAtFace].row(0).rightCols(cellVRRA.NDOF - 1) * uRec[iCell].m());
                        //             FFace.setZero();
                        //         }
                        //         else if (faceAttribute.iPhy == BoundaryType::Farfield)
                        //         {
                        //             FFace = VFV->faceDiBjGaussCache[iFace][ig * 2 + iCellAtFace].row(0).rightCols(cellVRRA.NDOF - 1) * uRec[iCell].m();
                        //         }
                        //         else
                        //         {
                        //             assert(false);
                        //         }
                        //         biInc = (faceDiBjGaussCache[iFace][ig * 2 + iCellAtFace].row(0).rightCols(cellRA.NDOF - 1).transpose()) *
                        //                 FFace * std::pow((*faceWeights)[iFace][0], 2);
                        //         biInc *= faceNorms[iFace][ig].norm();
                        //     });
                        if (faceAttribute.iPhy == BoundaryType::Wall)
                        {
                            Eigen::MatrixXd bcval;
                            bcval.resizeLike(u[iCell].p());
                            bcval.setZero();
                            // adds zero here for zero-valued VR value
                            bi += matrixBatchElem.m(1 + ic2f * 2 + 1) * (bcval - u[iCell].p());
                        }
                        else if (faceAttribute.iPhy == BoundaryType::Farfield)
                        {
                            bi += matrixBatchElem.m(1 + ic2f * 2 + 1) * u[iCell].p() * 0;
                        }
                        else
                        {
                            assert(false);
                        }
                    }
                }

                uRecCR[iCell].m() = matrixBatchElem.m(0) * bi;
                // std::cout << "DIFF\n"
                //           << uRecNewBuf[iCell].m() << std::endl;
            }
            // );
        }

        void Initialization();

        // static const int vsize = 1;
        void Reconstruction(ArrayLocal<VecStaticBatch<1>> &u,
                            ArrayLocal<SemiVarMatrix<1>> &uRec, ArrayLocal<SemiVarMatrix<1>> &uRecCR);
    };
}