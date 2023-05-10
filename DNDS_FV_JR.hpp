#pragma once
#include "DNDS_FV_VR.hpp"

namespace DNDS
{
    class JRFiniteVolume2D : public VRFiniteVolume2D
    {
    public:
        using VRFiniteVolume2D::VRFiniteVolume2D;

        void initReconstructionMatVec()
        {
            setting.tangWeight = 1;
            // std::cout << "IN JR";
            auto fGetMatrixSize = [&](int &nmats, std::vector<int> &matSizes, index iCell)
            {
                // auto c2f = mesh->cell2faceLocal[iCell];
                // auto &cellAttribute = mesh->cellAtrLocal[iCell][0];
                auto &cellRecAttribute = cellRecAtrLocal[iCell][0];
                // auto eCell = Elem::ElementManager(cellAttribute.type, cellAttribute.intScheme);
                nmats = 1;
                matSizes.resize(nmats * 2);
                matSizes[0 * 2 + 0] = cellRecAttribute.NDOF - 1;
                matSizes[0 * 2 + 1] = cellRecAttribute.NDOF - 1;
            };

            matrixBatch = std::make_shared<decltype(matrixBatch)::element_type>(
                decltype(matrixBatch)::element_type::tContext(
                    [&](index i) -> rowsize
                    {
                        int nmats = -123;
                        std::vector<int> matSizes;
                        fGetMatrixSize(nmats, matSizes, i);
                        DNDS_assert(matSizes.size() == nmats * 2);
                        return DNDS::SmallMatricesBatch::predictSize(nmats, matSizes);
                    },
                    [&](uint8_t *data, index siz, index i)
                    {
                        int nmats = -123;
                        std::vector<int> matSizes;
                        fGetMatrixSize(nmats, matSizes, i);
                        DNDS_assert(matSizes.size() == nmats * 2);
                        DNDS::SmallMatricesBatch::initializeData(data, nmats, matSizes);
                    },
                    mesh->cell2faceLocal.dist->size()),
                mpi);

            forEachInArray(
                *mesh->cell2faceLocal.dist,
                [&](tAdjArray::tComponent &c2f, index iCell)
                {
                    auto &cellAttribute = mesh->cellAtrLocal[iCell][0];
                    auto &cellRecAttribute = cellRecAtrLocal[iCell][0];
                    auto eCell = Elem::ElementManager(cellAttribute.type, cellAttribute.intScheme);
                    DNDS_assert(c2f.size() == eCell.getNFace());

                    auto matrixBatchElem = (*matrixBatch)[iCell];

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
                        auto faceDiBjGaussBatchElem = (*faceDiBjGaussBatch)[iFace];
                        auto faceDiBjCenterBatchElem = (*faceDiBjCenterBatch)[iFace];

                        Elem::ElementManager eFace(faceAttribute.type, faceRecAttribute.intScheme);

                        // eFace.Integration(
                        //     A,
                        //     [&](Eigen::MatrixXd &incA, int ig, Elem::tPoint &ip, Elem::tDiFj &iDiNj)
                        //     {
                        //         Eigen::MatrixXd diffsI = faceDiBjGaussBatchElem.m(ig * 2 + iCellAtFace);
                        //         Eigen::MatrixXd incAFull;
                        //         FFaceFunctional(iFace, ig, diffsI.topRows(1), diffsI.topRows(1), (*faceWeights)[iFace].topRows(1), incAFull);
                        //         DNDS_assert(incAFull(Eigen::all, 0).norm() + incAFull(0, Eigen::all).norm() == 0);
                        //         incA = incAFull.bottomRightCorner(incAFull.rows() - 1, incAFull.cols() - 1);
                        //         incA *= faceNorms[iFace][ig].norm(); // note: don't forget the Jacobi!!!
                        //     });
                        Eigen::MatrixXd diffsI = faceDiBjCenterBatchElem.m(iCellAtFace);
                        Eigen::MatrixXd incAFull;
                        FFaceFunctional(iFace, -1, diffsI, diffsI, (*faceWeights)[iFace], incAFull);
                        if (incAFull(Eigen::all, 0).norm() + incAFull(0, Eigen::all).norm() != 0)
                        {
                            std::cout << incAFull << std::endl;
                            std::cout << "W " << (*faceWeights)[iFace] << std::endl;
                            std::cout << faceNormCenter[iFace] << std::endl;
                        }
                        DNDS_assert(incAFull(Eigen::all, 0).norm() + incAFull(0, Eigen::all).norm() == 0);
                        
                        Eigen::MatrixXd incA = incAFull.bottomRightCorner(incAFull.rows() - 1, incAFull.cols() - 1);
                        A += incA * faceNormCenter[iFace].norm();
                    }
                    Eigen::MatrixXd Ainv, AinvFilterd;
                    HardEigen::EigenLeastSquareInverse(A, Ainv);
                    // HardEigen::EigenLeastSquareInverse_Filtered(A, Ainv);

                    matrixBatchElem.m(0) = Ainv;
                    // exit(0);
                });
        }

        template <uint32_t vsize>
        // static const int vsize = 1;
        void Reconstruction(ArrayLocal<VecStaticBatch<vsize>> &u,
                            ArrayRecV &uRec, ArrayRecV &uRecNew)
        {
            // InsertCheck(mpi, "JR Reconstruction 0");
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
                auto &c2f = mesh->cell2faceLocal[iCell];
                int NDOFCell = cellRA.NDOF;
                Eigen::MatrixXd bi(NDOFCell - 1, vsize);
                bi.setZero();
                auto matrixBatchElem = (*matrixBatch)[iCell];

                Elem::tPoint gradAll{0, 0, 0};
                bool hasUpper = false;

                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    // this is a repeated code block START
                    index iFace = c2f[ic2f];
                    auto &f2c = (*mesh->face2cellPair)[iFace];
                    index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                    index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                    auto &faceAttribute = mesh->faceAtrLocal[iFace][0];
                    auto &faceRecAttribute = faceRecAtrLocal[iFace][0];
                    auto faceDiBjCenterBatchElem = (*faceDiBjCenterBatch)[iFace];

                    Elem::ElementManager eFace(faceAttribute.type, faceRecAttribute.intScheme);
                    Eigen::MatrixXd fcoords;
                    mesh->LoadCoords(mesh->face2nodeLocal[iFace], fcoords);
                    // Elem::tPoint faceL = fcoords(Eigen::all, 1) - fcoords(Eigen::all, 0);
                    // Elem::tPoint faceN{faceL(1), -faceL(0), 0.0}; // 2-d specific //pointing from left to right
                    Elem::tPoint faceN = faceNormCenter[iFace].normalized();

                    if (iCellOther != FACE_2_VOL_EMPTY)
                    {
                        auto &cellRAOther = cellRecAtrLocal[iCellOther][0];

                        //! don't forget the mean value between them
                        int NDOF = faceDiBjCenterBatchElem.m(iCellAtFace).cols();
                        Eigen::MatrixXd uRecVal = faceDiBjCenterBatchElem.m(iCellAtFace).rightCols(NDOF - 1) * uRec[iCell];
                        int NDOFOther = faceDiBjCenterBatchElem.m(1 - iCellAtFace).cols();
                        Eigen::MatrixXd uRecValOther = faceDiBjCenterBatchElem.m(1 - iCellAtFace).rightCols(NDOFOther - 1) * uRec[iCellOther];
                        uRecValOther.row(0) += (u[iCellOther].p() - u[iCell].p()).transpose();

                        Eigen::MatrixXd uRecBoundary = (uRecVal + uRecValOther) * 0.5;
                        Eigen::MatrixXd biIncFull;
                        Eigen::MatrixXd DiffI = faceDiBjCenterBatchElem.m(iCellAtFace);
                        // std::cout << "DIFFI " << std::endl;
                        // std::cout << DiffI << std::endl;
                        // std::cout << "uRecB" << std::endl
                        //           << uRecBoundary << std::endl;
                        FFaceFunctional(iFace, -1, DiffI, uRecBoundary, (*faceWeights)[iFace], biIncFull);
                        // InsertCheck(mpi, "JR Reconstruction FF1");
                        DNDS_assert(biIncFull(0, Eigen::all).norm() == 0);
                        Eigen::MatrixXd biInc = biIncFull.bottomRows(biIncFull.rows() - 1);
                        bi += biInc * faceNormCenter[iFace].norm();
                    }
                    else
                    {
                        if (faceAttribute.iPhy == BoundaryType::Wall)
                        {
                            int NDOF = faceDiBjCenterBatchElem.m(iCellAtFace).cols();
                            Eigen::MatrixXd uRecVal = faceDiBjCenterBatchElem.m(iCellAtFace).rightCols(NDOF - 1) * uRec[iCell];
                            Eigen::MatrixXd uRecValOther = uRecVal * 0.0;
                            uRecValOther.row(0) += (u[iCell].p() * 0.0 - u[iCell].p()).transpose();

                            Eigen::MatrixXd uRecBoundary = (uRecVal + uRecValOther) * 0.5;
                            Eigen::MatrixXd biIncFull;
                            Eigen::MatrixXd DiffI = faceDiBjCenterBatchElem.m(iCellAtFace);
                            FFaceFunctional(iFace, -1, DiffI, uRecBoundary, (*faceWeights)[iFace], biIncFull);
                            DNDS_assert(biIncFull(0, Eigen::all).norm() == 0);
                            Eigen::MatrixXd biInc = biIncFull.bottomRows(biIncFull.rows() - 1);
                            bi += biInc * faceNormCenter[iFace].norm();
                        }
                        else if (faceAttribute.iPhy == BoundaryType::Farfield)
                        {
                            int NDOF = faceDiBjCenterBatchElem.m(iCellAtFace).cols();
                            Eigen::MatrixXd uRecVal = faceDiBjCenterBatchElem.m(iCellAtFace).rightCols(NDOF - 1) * uRec[iCell];
                            Eigen::MatrixXd uRecValOther = uRecVal;
                            // uRecValOther.row(0) += (u[iCell].p() * 0.0 - u[iCell].p()).transpose();

                            Eigen::MatrixXd uRecBoundary = (uRecVal + uRecValOther) * 0.5;
                            Eigen::MatrixXd biIncFull;
                            Eigen::MatrixXd DiffI = faceDiBjCenterBatchElem.m(iCellAtFace);
                            FFaceFunctional(iFace, -1, DiffI, uRecBoundary, (*faceWeights)[iFace], biIncFull);
                            DNDS_assert(biIncFull(0, Eigen::all).norm() == 0);
                            Eigen::MatrixXd biInc = biIncFull.bottomRows(biIncFull.rows() - 1);
                            bi += biInc * faceNormCenter[iFace].norm();
                        }
                        else
                        {
                            DNDS_assert(false);
                        }
                    }
                }

                uRecNew[iCell] = matrixBatchElem.m(0) * bi;
                // std::cout << "DIFF\n"
                //           << uRecNewBuf[iCell] << std::endl;
            }
            // InsertCheck(mpi, "JR Reconstruction 1");
            real vall = 0;
            real nall = 0;
            real vallR, nallR;
            for (index iCell = 0; iCell < uRec.dist->size(); iCell++)
            {
                auto &uRecE = uRec[iCell];
                nall += (uRecE - uRecNew[iCell]).squaredNorm();
                vall += 1;
            }
            MPI_Allreduce(&vall, &vallR, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm); //? remove at release
            MPI_Allreduce(&nall, &nallR, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            real res = nallR / vallR;
            if (mpi.rank == 0)
            {
                auto fmt = log().flags();
                log() << " Rec RES " << std::scientific << std::setprecision(10) << res << std::endl;
                log().setf(fmt);
            }

            for (index iCell = 0; iCell < uRec.dist->size(); iCell++)
            {
                auto &uRecE = uRec[iCell];
                uRecE = uRecNew[iCell];
            }
        }
    };
}