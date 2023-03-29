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

// derive intscheme, ndof ,ndiff in rec attributes
void DNDS::VRFiniteVolume2D::initIntScheme() //  2-d specific
{
    cellRecAtrLocal.dist = std::make_shared<decltype(cellRecAtrLocal.dist)::element_type>(
        decltype(cellRecAtrLocal.dist)::element_type::tComponent::Context(mesh->cellAtrLocal.dist->size()), mpi);
    cellRecAtrLocal.CreateGhostCopyComm(mesh->cellAtrLocal);

    forEachInArray(
        *mesh->cellAtrLocal.dist,
        [&](tElemAtrArray::tComponent &atr, index iCell)
        {
            Elem::ElementManager eCell(atr[0].type, atr[0].intScheme);
            auto &recAtr = cellRecAtrLocal[iCell][0];
            switch (eCell.getPspace())
            {
            case Elem::ParamSpace::TriSpace:
                recAtr.intScheme = Elem::INT_SCHEME_TRI_4;
                recAtr.NDOF = PolynomialNDOF(P_ORDER);
                recAtr.NDIFF = PolynomialNDOF(P_ORDER);
                break;
            case Elem::ParamSpace::QuadSpace:
                recAtr.intScheme = Elem::INT_SCHEME_QUAD_4;
                recAtr.NDOF = PolynomialNDOF(P_ORDER);
                recAtr.NDIFF = PolynomialNDOF(P_ORDER);
                break;
            default:
                assert(false);
                break;
            }
            recAtr.relax = setting.JacobiRelax;
        });
    cellRecAtrLocal.PullOnce();
    faceRecAtrLocal.dist = std::make_shared<decltype(faceRecAtrLocal.dist)::element_type>(
        decltype(faceRecAtrLocal.dist)::element_type::tComponent::Context(mesh->faceAtrLocal.dist->size()), mpi);

    faceRecAtrLocal.CreateGhostCopyComm(mesh->faceAtrLocal);

    forEachInArray(
        *mesh->faceAtrLocal.dist,
        [&](tElemAtrArray::tComponent &atr, index iFace)
        {
            Elem::ElementManager eFace(atr[0].type, atr[0].intScheme);
            auto &recAtr = faceRecAtrLocal[iFace][0];
            auto &atrr = mesh->faceAtrLocal[iFace][0];
            switch (eFace.getPspace())
            {
            case Elem::ParamSpace::LineSpace:
                recAtr.intScheme = Elem::INT_SCHEME_LINE_3;
                recAtr.NDOF = PolynomialNDOF(P_ORDER);
                recAtr.NDIFF = PolynomialNDOF(P_ORDER);
                if (atrr.iPhy == BoundaryType::Wall)
                    recAtr.intScheme = Elem::INT_SCHEME_LINE_3;
                break;
            default:
                assert(false);
                break;
            }
        });
    faceRecAtrLocal.PullOnce();
}

void DNDS::VRFiniteVolume2D::initMoment()
{
    // InsertCheck(mpi, "InitMomentStart");
    index nlocalCells = mesh->cell2nodeLocal.size();
    baseMoments.resize(nlocalCells);
    cellCenters.resize(nlocalCells);
    cellBaries.resize(nlocalCells);
    cellIntertia = std::make_shared<decltype(cellIntertia)::element_type>(nlocalCells);
    forEachInArrayPair(
        *mesh->cell2nodeLocal.pair,
        [&](tAdjArray::tComponent &c2n, index iCell)
        {
            auto cellRecAtr = cellRecAtrLocal[iCell][0];
            auto cellAtr = mesh->cellAtrLocal[iCell][0];
            Elem::ElementManager eCell(cellAtr.type, cellRecAtr.intScheme);
            Eigen::MatrixXd coords;
            mesh->LoadCoords(c2n, coords);

            // get center
            Eigen::MatrixXd DiNj(4, eCell.getNNode());
            eCell.GetDiNj(eCell.getCenterPParam(), DiNj); // N_cache not using
            // InsertCheck(mpi, "Do1");
            cellCenters[iCell] = coords * DiNj(0, Eigen::all).transpose(); // center point derived
            // InsertCheck(mpi, "Do1End");
            Elem::tPoint sScale = CoordMinMaxScale(coords);

            cellBaries[iCell].setZero();
            eCell.Integration(
                cellBaries[iCell],
                [&](Elem::tPoint &incC, int ig, Elem::tPoint &ip, Elem::tDiFj &iDiNj)
                {
                    incC = coords * iDiNj(0, Eigen::all).transpose(); // = pPhysical
                    Elem::tJacobi Jacobi = Elem::DiNj2Jacobi(iDiNj, coords);
                    incC *= Jacobi({0, 1}, {0, 1}).determinant();
                });
            cellBaries[iCell] /= FV->volumeLocal[iCell];
            // std::cout << cellBaries[iCell] << std::endl;
            // exit(0);
            (*cellIntertia)[iCell].setZero();
            eCell.Integration(
                (*cellIntertia)[iCell],
                [&](Elem::tJacobi &incC, int ig, Elem::tPoint &ip, Elem::tDiFj &iDiNj)
                {
                    Elem::tPoint pPhysical = coords * iDiNj(0, Eigen::all).transpose() - cellBaries[iCell]; // = pPhysical
                    incC = pPhysical * pPhysical.transpose();
                    Elem::tJacobi Jacobi = Elem::DiNj2Jacobi(iDiNj, coords);
                    incC *= Jacobi({0, 1}, {0, 1}).determinant();
                });
            (*cellIntertia)[iCell] /= FV->volumeLocal[iCell];
            // std::cout << "I\n"
            //           << (*cellIntertia)[iCell] << std::endl;
            // auto iOrig = (*cellIntertia)[iCell];
            (*cellIntertia)[iCell] = HardEigen::Eigen3x3RealSymEigenDecomposition((*cellIntertia)[iCell]);
            // std::cout << "IS\n"
            //           << (*cellIntertia)[iCell] << std::endl;

            // std::cout << iOrig << std::endl;
            // std::cout << (*cellIntertia)[iCell] * (*cellIntertia)[iCell].transpose() << std::endl;
            // std::cout << (*cellIntertia)[iCell] << std::endl;
            // std::cout << (*cellIntertia)[iCell].col(0).norm() * 3 << " " << (*cellIntertia)[iCell].col(1).norm() * 3 << std::endl;
            // exit(0);

            Eigen::MatrixXd BjBuffer(1, cellRecAtr.NDOF);
            BjBuffer.setZero();
            eCell.Integration(
                BjBuffer,
                [&](Eigen::MatrixXd &incBj, int ig, Elem::tPoint &ip, Elem::tDiFj &iDiNj)
                {
                    // InsertCheck(mpi, "Do2");
                    incBj.resize(1, cellRecAtr.NDOF);
                    // InsertCheck(mpi, "Do2End");

                    FDiffBaseValue(iCell, eCell, coords, iDiNj,
                                   ip, getCellCenter(iCell), sScale,
                                   Eigen::VectorXd::Zero(cellRecAtr.NDOF),
                                   incBj);
                    Elem::tJacobi Jacobi = Elem::DiNj2Jacobi(iDiNj, coords);
                    incBj *= Jacobi({0, 1}, {0, 1}).determinant();
                    // std::cout << "JACOBI " << Jacobi({0, 1}, {0, 1}).determinant() * 4 << std::endl;
                    // std::cout << "INC " << incBj(1) * 4 << std::endl;
                    // std::cout << "IncBj 0 " << incBj(0, 0) << std::endl;
                });
            // std::cout << "BjBuffer0 " << BjBuffer(0, 0) << std::endl;
            baseMoments[iCell] = (BjBuffer / FV->volumeLocal[iCell]).transpose();
            // std::cout << "MOMENT\n"
            //           << baseMoments[iCell] << std::endl;
            // std::cout << "V " << FV->volumeLocal[iCell] << std::endl;
            // std::cout << "BM0 " << 1 - baseMoments[iCell](0) << std::endl; //should better be machine eps
            baseMoments[iCell](0) = 1.; // just for binary accuracy
        });
    // InsertCheck(mpi, "InitMomentEnd");
}

void DNDS::VRFiniteVolume2D::initBaseDiffCache()
{
    // InsertCheck(mpi, "initBaseDiffCache");
    index nlocalCells = mesh->cell2nodeLocal.size();
    cellGaussJacobiDets.resize(nlocalCells);

    matrixInnerProd = std::make_shared<decltype(matrixInnerProd)::element_type>(mesh->cell2faceLocal.pair->size());

    // * Allocate space for cellDiBjGaussBatch
    auto fGetCellDiBjGaussSize = [&](int &nmat, std::vector<int> &matSizes, index iCell)
    {
        auto cellRecAtr = cellRecAtrLocal[iCell][0];
        auto cellAtr = mesh->cellAtrLocal[iCell][0];
        Elem::ElementManager eCell(cellAtr.type, cellRecAtr.intScheme);
        nmat = eCell.getNInt();
        matSizes.resize(nmat * 2);
        for (int ig = 0; ig < eCell.getNInt(); ig++)
        {
            matSizes[ig * 2 + 0] = cellRecAtr.NDIFF;
            matSizes[ig * 2 + 1] = cellRecAtr.NDOF;
        }
    };
    cellDiBjGaussBatch = std::make_shared<decltype(cellDiBjGaussBatch)::element_type>(
        decltype(cellDiBjGaussBatch)::element_type::tContext(
            [&](index i) -> rowsize
            {
                int nmats = -123;
                std::vector<int> matSizes;
                fGetCellDiBjGaussSize(nmats, matSizes, i);
                assert(matSizes.size() == nmats * 2);
                return DNDS::SmallMatricesBatch::predictSize(nmats, matSizes);
            },
            [&](uint8_t *data, index siz, index i)
            {
                int nmats = -123;
                std::vector<int> matSizes;
                fGetCellDiBjGaussSize(nmats, matSizes, i);
                assert(matSizes.size() == nmats * 2);
                DNDS::SmallMatricesBatch::initializeData(data, nmats, matSizes);
            },
            mesh->cell2faceLocal.size()),
        mpi);

    // *Allocate space for cellDiBjCenterBatch
    auto fGetCellDiBjCenterSize = [&](int &nmat, std::vector<int> &matSizes, index iCell)
    {
        auto cellRecAtr = cellRecAtrLocal[iCell][0];
        auto cellAtr = mesh->cellAtrLocal[iCell][0];
        Elem::ElementManager eCell(cellAtr.type, cellRecAtr.intScheme);
        nmat = 1;
        matSizes.resize(nmat * 2);
        matSizes[0 * 2 + 0] = cellRecAtr.NDIFF;
        matSizes[0 * 2 + 1] = cellRecAtr.NDOF;
    };
    cellDiBjCenterBatch = std::make_shared<decltype(cellDiBjCenterBatch)::element_type>(
        decltype(cellDiBjCenterBatch)::element_type::tContext(
            [&](index i) -> rowsize
            {
                int nmats = -123;
                std::vector<int> matSizes;
                fGetCellDiBjCenterSize(nmats, matSizes, i);
                assert(matSizes.size() == nmats * 2);
                return DNDS::SmallMatricesBatch::predictSize(nmats, matSizes);
            },
            [&](uint8_t *data, index siz, index i)
            {
                int nmats = -123;
                std::vector<int> matSizes;
                fGetCellDiBjCenterSize(nmats, matSizes, i);
                assert(matSizes.size() == nmats * 2);
                DNDS::SmallMatricesBatch::initializeData(data, nmats, matSizes);
            },
            mesh->cell2faceLocal.size()),
        mpi);

    forEachInArrayPair(
        *mesh->cell2nodeLocal.pair,
        [&](tAdjArray::tComponent &c2n, index iCell)
        {
            auto cellRecAtr = cellRecAtrLocal[iCell][0];
            auto cellAtr = mesh->cellAtrLocal[iCell][0];
            Elem::ElementManager eCell(cellAtr.type, cellRecAtr.intScheme);
            auto cellDiBjCenterBatchElem = (*cellDiBjCenterBatch)[iCell];
            auto cellDiBjGaussBatchElem = (*cellDiBjGaussBatch)[iCell];
            // center part
            Elem::tPoint p = eCell.getCenterPParam();
            Eigen::MatrixXd coords;
            mesh->LoadCoords(c2n, coords);
            Eigen::MatrixXd DiNj(4, eCell.getNNode());
            eCell.GetDiNj(p, DiNj); // N_cache not using
            Elem::tPoint sScale = CoordMinMaxScale(coords);
            FDiffBaseValue(iCell, eCell, coords, DiNj,
                           p, getCellCenter(iCell), sScale,
                           baseMoments[iCell],
                           cellDiBjCenterBatchElem.m(0));

            // iGaussPart
            cellGaussJacobiDets[iCell].resize(eCell.getNInt());
            for (int ig = 0; ig < eCell.getNInt(); ig++)
            {
                eCell.GetIntPoint(ig, p);
                eCell.GetDiNj(p, DiNj); // N_cache not using
                FDiffBaseValue(iCell, eCell, coords, DiNj,
                               p, getCellCenter(iCell), sScale,
                               baseMoments[iCell],
                               cellDiBjGaussBatchElem.m(ig));
                cellGaussJacobiDets[iCell][ig] = Elem::DiNj2Jacobi(DiNj, coords)({0, 1}, {0, 1}).determinant();
            }

            (*matrixInnerProd)[iCell].resize(cellRecAtr.NDOF - 1, cellRecAtr.NDOF - 1);
            (*matrixInnerProd)[iCell].setZero();

            Elem::tIntScheme HighScheme = -1;
            switch (eCell.getPspace())
            {
            case Elem::ParamSpace::QuadSpace:
                HighScheme = Elem::INT_SCHEME_QUAD_25;
                break;
            case Elem::ParamSpace::TriSpace:
                HighScheme = Elem::INT_SCHEME_TRI_13;
                break;
            default:
                assert(false);
                break;
            }

            Elem::ElementManager eCellHigh(cellAtr.type, HighScheme);
            eCellHigh.Integration(
                (*matrixInnerProd)[iCell],
                [&](Eigen::MatrixXd &incA, int ig, Elem::tPoint &ip, Elem::tDiFj &iDiNj)
                {
                    Eigen::MatrixXd DiBj(1, cellRecAtr.NDOF);
                    FDiffBaseValue(iCell, eCell, coords, iDiNj,
                                   ip, getCellCenter(iCell), sScale,
                                   baseMoments[iCell],
                                   DiBj);
                    Eigen::MatrixXd ZeroDs = DiBj({0}, Eigen::seq(Eigen::fix<1>, Eigen::last));
                    incA = ZeroDs.transpose() * ZeroDs * (1.0 / FV->volumeLocal[iCell]);
                    incA *= Elem::DiNj2Jacobi(iDiNj, coords)({0, 1}, {0, 1}).determinant();
                });
            if (setting.orthogonalizeBase)
            {
                //* do the converting from natural base to orth base
                auto ldltResult = (*matrixInnerProd)[iCell].ldlt();
                if (ldltResult.vectorD()(Eigen::last) < smallReal)
                {
                    std::cout << "Orth llt failure";
                    assert(false);
                }
                auto lltResult = (*matrixInnerProd)[iCell].llt();
                Eigen::Index nllt = (*matrixInnerProd)[iCell].rows();
                Eigen::MatrixXd LOrth = lltResult.matrixL().solve(Eigen::MatrixXd::Identity(nllt, nllt));
                // std::cout << "Orth converter Orig" << std::endl;
                // std::cout << LOrth << std::endl;
                // std::cout << LOrth.array().abs().rowwise().sum().matrix().asDiagonal() << std::endl;//! using asDiagonal is buggy here!
                // LOrth = LOrth.array().abs().rowwise().sum().matrix().asDiagonal().inverse() * LOrth; //! buggy inplace calculation!
                // std::cout << LOrth << std::endl;
                // std::cout << LOrth.array().abs().rowwise().sum().matrix().asDiagonal().inverse() * LOrth << std::endl;

                LOrth = (LOrth.array().colwise() / LOrth.array().abs().rowwise().sum()).matrix();
                // std::cout << "Inner Product" << std::endl;
                // std::cout << (*matrixInnerProd)[iCell] << std::endl;
                // std::cout << "Orth converter" << std::endl;
                // std::cout << LOrth << std::endl;
                // assert(false);

                cellDiBjCenterBatchElem.m(0)(
                    Eigen::seq(Eigen::fix<1>, Eigen::last),
                    Eigen::seq(Eigen::fix<1>, Eigen::last)) *= LOrth.transpose();
                for (int ig = 0; ig < eCell.getNInt(); ig++)
                {
                    cellDiBjGaussBatchElem.m(ig)(
                        Eigen::seq(Eigen::fix<1>, Eigen::last),
                        Eigen::seq(Eigen::fix<1>, Eigen::last)) *= LOrth.transpose();
                }
            }
        });
    // InsertCheck(mpi, "initBaseDiffCache Cell Ended");

    // *face part: sides
    index nlocalFaces = mesh->face2nodeLocal.size();
    // faceDiBjCenterCache.resize(nlocalFaces);
    // faceDiBjGaussCache.resize(nlocalFaces);
    faceNorms.resize(nlocalFaces);
    faceWeights = std::make_shared<decltype(faceWeights)::element_type>(nlocalFaces);
    faceCenters.resize(nlocalFaces);
    faceNormCenter.resize(nlocalFaces);

    // *Allocate space for faceDiBjCenterBatch
    auto fGetFaceDiBjCenterSize = [&](int &nmat, std::vector<int> &matSizes, index iFace)
    {
        auto faceRecAtr = faceRecAtrLocal[iFace][0];
        auto faceAtr = mesh->faceAtrLocal[iFace][0];
        Elem::ElementManager eFace(faceAtr.type, faceRecAtr.intScheme);
        auto f2n = mesh->face2nodeLocal[iFace];
        auto f2c = mesh->face2cellLocal[iFace];
        nmat = 2;
        matSizes.resize(nmat * 2, 0);

        DNDS::RecAtr *cellRecAtrL{nullptr};
        DNDS::RecAtr *cellRecAtrR{nullptr};
        DNDS::ElemAttributes *cellAtrL{nullptr};
        DNDS::ElemAttributes *cellAtrR{nullptr};
        cellRecAtrL = &cellRecAtrLocal[f2c[0]][0];
        cellAtrL = &mesh->cellAtrLocal[f2c[0]][0];
        if (f2c[1] != FACE_2_VOL_EMPTY)
        {
            cellRecAtrR = &cellRecAtrLocal[f2c[1]][0];
            cellAtrR = &mesh->cellAtrLocal[f2c[1]][0];
        }
        // doing center
        matSizes[0 * 2 + 0] = faceRecAtr.NDIFF;
        matSizes[0 * 2 + 1] = cellRecAtrL->NDOF;
        if (f2c[1] != FACE_2_VOL_EMPTY)
        {
            matSizes[1 * 2 + 0] = faceRecAtr.NDIFF;
            matSizes[1 * 2 + 1] = cellRecAtrR->NDOF;
        }
    };
    faceDiBjCenterBatch = std::make_shared<decltype(faceDiBjCenterBatch)::element_type>(
        decltype(faceDiBjCenterBatch)::element_type::tContext(
            [&](index i) -> rowsize
            {
                int nmats = -123;
                std::vector<int> matSizes;
                fGetFaceDiBjCenterSize(nmats, matSizes, i);
                assert(matSizes.size() == nmats * 2);
                return DNDS::SmallMatricesBatch::predictSize(nmats, matSizes);
            },
            [&](uint8_t *data, index siz, index i)
            {
                int nmats = -123;
                std::vector<int> matSizes;
                fGetFaceDiBjCenterSize(nmats, matSizes, i);
                assert(matSizes.size() == nmats * 2);
                DNDS::SmallMatricesBatch::initializeData(data, nmats, matSizes);
            },
            mesh->face2cellLocal.size()),
        mpi);

    // *Allocate space for faceDiBjGaussBatch
    auto fGetFaceDiBjGaussSize = [&](int &nmat, std::vector<int> &matSizes, index iFace)
    {
        auto faceRecAtr = faceRecAtrLocal[iFace][0];
        auto faceAtr = mesh->faceAtrLocal[iFace][0];
        Elem::ElementManager eFace(faceAtr.type, faceRecAtr.intScheme);
        auto f2n = mesh->face2nodeLocal[iFace];
        auto f2c = mesh->face2cellLocal[iFace];
        nmat = eFace.getNInt() * 2;
        matSizes.resize(nmat * 2, 0);

        DNDS::RecAtr *cellRecAtrL{nullptr};
        DNDS::RecAtr *cellRecAtrR{nullptr};
        DNDS::ElemAttributes *cellAtrL{nullptr};
        DNDS::ElemAttributes *cellAtrR{nullptr};
        cellRecAtrL = &cellRecAtrLocal[f2c[0]][0];
        cellAtrL = &mesh->cellAtrLocal[f2c[0]][0];
        if (f2c[1] != FACE_2_VOL_EMPTY)
        {
            cellRecAtrR = &cellRecAtrLocal[f2c[1]][0];
            cellAtrR = &mesh->cellAtrLocal[f2c[1]][0];
        }
        // start doing gauss
        for (int ig = 0; ig < eFace.getNInt(); ig++)
        {
            matSizes[(ig * 2 + 0) * 2 + 0] = faceRecAtr.NDIFF;
            matSizes[(ig * 2 + 0) * 2 + 1] = cellRecAtrL->NDOF;
            if (f2c[1] != FACE_2_VOL_EMPTY)
            {
                // Right side
                matSizes[(ig * 2 + 1) * 2 + 0] = faceRecAtr.NDIFF;
                matSizes[(ig * 2 + 1) * 2 + 1] = cellRecAtrR->NDOF;
            }
        }
    };
    faceDiBjGaussBatch = std::make_shared<decltype(faceDiBjGaussBatch)::element_type>(
        decltype(faceDiBjGaussBatch)::element_type::tContext(
            [&](index i) -> rowsize
            {
                int nmats = -123;
                std::vector<int> matSizes;
                fGetFaceDiBjGaussSize(nmats, matSizes, i);
                assert(matSizes.size() == nmats * 2);
                return DNDS::SmallMatricesBatch::predictSize(nmats, matSizes);
            },
            [&](uint8_t *data, index siz, index i)
            {
                int nmats = -123;
                std::vector<int> matSizes;
                fGetFaceDiBjGaussSize(nmats, matSizes, i);
                assert(matSizes.size() == nmats * 2);
                DNDS::SmallMatricesBatch::initializeData(data, nmats, matSizes);
            },
            mesh->face2cellLocal.size()),
        mpi);

    // *Allocate space for matrixSecondaryBatch
    auto fGetMatrixSecondarySize = [&](int &nmat, std::vector<int> &matSizes, index iFace)
    {
        auto faceRecAtr = faceRecAtrLocal[iFace][0];
        auto faceAtr = mesh->faceAtrLocal[iFace][0];
        Elem::ElementManager eFace(faceAtr.type, faceRecAtr.intScheme);
        auto f2n = mesh->face2nodeLocal[iFace];
        auto f2c = mesh->face2cellLocal[iFace];
        nmat = 2;
        matSizes.resize(nmat * 2, 0);

        DNDS::RecAtr *cellRecAtrL{nullptr};
        DNDS::RecAtr *cellRecAtrR{nullptr};
        DNDS::ElemAttributes *cellAtrL{nullptr};
        DNDS::ElemAttributes *cellAtrR{nullptr};
        cellRecAtrL = &cellRecAtrLocal[f2c[0]][0];
        cellAtrL = &mesh->cellAtrLocal[f2c[0]][0];
        if (f2c[1] != FACE_2_VOL_EMPTY)
        {
            cellRecAtrR = &cellRecAtrLocal[f2c[1]][0];
            cellAtrR = &mesh->cellAtrLocal[f2c[1]][0];
        }

        if (f2c[1] != FACE_2_VOL_EMPTY)
        {
            matSizes[0 * 2 + 0] = cellRecAtrL->NDOF - 1;
            matSizes[0 * 2 + 1] = cellRecAtrR->NDOF - 1;
            matSizes[1 * 2 + 0] = cellRecAtrR->NDOF - 1;
            matSizes[1 * 2 + 1] = cellRecAtrL->NDOF - 1;
        }
    };
    matrixSecondaryBatch = std::make_shared<decltype(matrixSecondaryBatch)::element_type>(
        decltype(matrixSecondaryBatch)::element_type::tContext(
            [&](index i) -> rowsize
            {
                int nmats = -123;
                std::vector<int> matSizes;
                fGetMatrixSecondarySize(nmats, matSizes, i);
                assert(matSizes.size() == nmats * 2);
                return DNDS::SmallMatricesBatch::predictSize(nmats, matSizes);
            },
            [&](uint8_t *data, index siz, index i)
            {
                int nmats = -123;
                std::vector<int> matSizes;
                fGetMatrixSecondarySize(nmats, matSizes, i);
                assert(matSizes.size() == nmats * 2);
                DNDS::SmallMatricesBatch::initializeData(data, nmats, matSizes);
            },
            mesh->face2cellLocal.size()),
        mpi);

    forEachInArrayPair(
        *mesh->face2cellLocal.pair,
        [&](tAdjStatic2Array::tComponent &f2c, index iFace)
        {
            auto faceRecAtr = faceRecAtrLocal[iFace][0];
            auto faceAtr = mesh->faceAtrLocal[iFace][0];
            Elem::ElementManager eFace(faceAtr.type, faceRecAtr.intScheme);
            auto f2n = mesh->face2nodeLocal[iFace];
            auto faceDiBjCenterBatchElem = (*faceDiBjCenterBatch)[iFace];
            auto faceDiBjGaussBatchElem = (*faceDiBjGaussBatch)[iFace];
            auto matrixSecondaryBatchElem = (*matrixSecondaryBatch)[iFace];

            // faceDiBjGaussCache[iFace].resize(eFace.getNInt() * 2); // note the 2x size
            faceNorms[iFace].resize(eFace.getNInt());

            Eigen::MatrixXd faceCoords;
            mesh->LoadCoords(f2n, faceCoords);
            Eigen::MatrixXd cellCoordsL;
            Eigen::MatrixXd cellCoordsR;
            Elem::tPoint sScaleL;
            Elem::tPoint sScaleR;
            DNDS::RecAtr *cellRecAtrL{nullptr};
            DNDS::RecAtr *cellRecAtrR{nullptr};
            DNDS::ElemAttributes *cellAtrL{nullptr};
            DNDS::ElemAttributes *cellAtrR{nullptr};
            mesh->LoadCoords(mesh->cell2nodeLocal[f2c[0]], cellCoordsL);
            sScaleL = CoordMinMaxScale(cellCoordsL);
            cellRecAtrL = &cellRecAtrLocal[f2c[0]][0];
            cellAtrL = &mesh->cellAtrLocal[f2c[0]][0];
            if (f2c[1] != FACE_2_VOL_EMPTY)
            {
                mesh->LoadCoords(mesh->cell2nodeLocal[f2c[1]], cellCoordsR);
                sScaleR = CoordMinMaxScale(cellCoordsR);
                cellRecAtrR = &cellRecAtrLocal[f2c[1]][0];
                cellAtrR = &mesh->cellAtrLocal[f2c[1]][0];
            }

            /*******************************************/
            // start doing center
            Elem::tPoint pFace = eFace.getCenterPParam();

            Elem::tDiFj faceDiNj(4, eFace.getNNode());
            eFace.GetDiNj(pFace, faceDiNj);
            faceCenters[iFace] = faceCoords * faceDiNj(0, Eigen::all).transpose();
            faceNormCenter[iFace] = Elem::Jacobi2LineNorm2D(Elem::DiNj2Jacobi(faceDiNj, faceCoords));
            assert(faceNormCenter[iFace].dot(cellCenters[f2c[0]] - faceCenters[iFace]) < 0.0);

            // Left side
            index iCell = f2c[0];
            Elem::tPoint pCell;
            mesh->FacePParam2Cell(iCell, 0, iFace, f2n, eFace, pFace, pCell);
            Elem::ElementManager eCell(cellAtrL->type, 0); // int scheme is not relevant here
            // faceDiBjCenterCache[iFace].first.resize(faceRecAtr.NDIFF, cellRecAtrL->NDOF);
            Elem::tDiFj cellDiNj(4, eCell.getNNode());
            eCell.GetDiNj(pCell, cellDiNj);
            FDiffBaseValue(iCell, eCell, cellCoordsL, cellDiNj,
                           pCell, getCellCenter(iCell), sScaleL,
                           baseMoments[iCell], faceDiBjCenterBatchElem.m(0));

            // Elem::tPoint pCellLPhy = cellCoordsL * cellDiNj({0}, Eigen::all).transpose();
            if (f2c[1] != FACE_2_VOL_EMPTY)
            {
                // right side
                index iCell = f2c[1];
                Elem::tPoint pCell; // IND identical apart from "second", R
                // InsertCheck(mpi, "DO1");
                mesh->FacePParam2Cell(iCell, 1, iFace, f2n, eFace, pFace, pCell);
                // InsertCheck(mpi, "DO1End");
                Elem::ElementManager eCell(cellAtrR->type, 0); // int scheme is not relevant here
                // faceDiBjCenterCache[iFace].second.resize(faceRecAtr.NDIFF, cellRecAtrR->NDOF);
                Elem::tDiFj cellDiNj(4, eCell.getNNode());
                eCell.GetDiNj(pCell, cellDiNj);
                FDiffBaseValue(iCell, eCell, cellCoordsR, cellDiNj,
                               pCell, getCellCenter(iCell), sScaleR,
                               baseMoments[iCell], faceDiBjCenterBatchElem.m(1));
                // Elem::tPoint pCellRPhy = cellCoordsR * cellDiNj({0}, Eigen::all).transpose();
                // std::cout << " " << pCellLPhy.transpose() << " " << pCellRPhy.transpose() << std::endl;
            }

            /*******************************************/
            // start doing gauss
            for (int ig = 0; ig < eFace.getNInt(); ig++)
            {
                Elem::tPoint pFace;
                eFace.GetIntPoint(ig, pFace);
                Elem::tDiFj faceDiNj(4, eFace.getNNode());
                eFace.GetDiNj(pFace, faceDiNj);
                faceNorms[iFace][ig] = Elem::Jacobi2LineNorm2D(Elem::DiNj2Jacobi(faceDiNj, faceCoords)); // face norms
                // InsertCheck(mpi, "DOA");
                // Left side
                index iCell = f2c[0];
                Elem::tPoint pCell; // IND identical apart from faceDiBjGaussCache[iFace][ig * 2 + 0]
                mesh->FacePParam2Cell(iCell, 0, iFace, f2n, eFace, pFace, pCell);
                Elem::ElementManager eCell(cellAtrL->type, 0); // int scheme is not relevant here
                // faceDiBjGaussCache[iFace][ig * 2 + 0].resize(faceRecAtr.NDIFF, cellRecAtrL->NDOF);
                Elem::tDiFj cellDiNj(4, eCell.getNNode());
                eCell.GetDiNj(pCell, cellDiNj);
                FDiffBaseValue(iCell, eCell, cellCoordsL, cellDiNj,
                               pCell, getCellCenter(iCell), sScaleL,
                               baseMoments[iCell], faceDiBjGaussBatchElem.m(ig * 2 + 0));
                // std::cout << "GP" << cellCoordsL * cellDiNj(0, Eigen::all).transpose() << std::endl;
                // InsertCheck(mpi, "DOAEND");
                if (f2c[1] != FACE_2_VOL_EMPTY)
                {
                    // Right side
                    index iCell = f2c[1];
                    Elem::tPoint pCell; // IND identical apart from faceDiBjGaussCache[iFace][ig * 2 + 1], R
                    mesh->FacePParam2Cell(iCell, 1, iFace, f2n, eFace, pFace, pCell);
                    Elem::ElementManager eCell(cellAtrR->type, 0); // int scheme is not relevant here
                    // faceDiBjGaussCache[iFace][ig * 2 + 1].resize(faceRecAtr.NDIFF, cellRecAtrR->NDOF);
                    Elem::tDiFj cellDiNj(4, eCell.getNNode());
                    eCell.GetDiNj(pCell, cellDiNj);
                    FDiffBaseValue(iCell, eCell, cellCoordsR, cellDiNj,
                                   pCell, getCellCenter(iCell), sScaleR,
                                   baseMoments[iCell], faceDiBjGaussBatchElem.m(ig * 2 + 1));
                }
            }
            if (setting.orthogonalizeBase)
            {
                //* do the converting from natural base to orth base
                auto ldltResult = (*matrixInnerProd)[iCell].ldlt();
                if (ldltResult.vectorD()(Eigen::last) < smallReal)
                {
                    std::cout << "Orth llt failure";
                    assert(false);
                }
                auto lltResult = (*matrixInnerProd)[iCell].llt();
                Eigen::Index nllt = (*matrixInnerProd)[iCell].rows();
                Eigen::MatrixXd LOrth = lltResult.matrixL().solve(Eigen::MatrixXd::Identity(nllt, nllt));
                LOrth = (LOrth.array().colwise() / LOrth.array().abs().rowwise().sum()).matrix();
                // std::cout << "Inner Product" << std::endl;
                // std::cout << (*matrixInnerProd)[iCell] << std::endl;
                // std::cout << "Face Orth converter" << std::endl;
                // std::cout << LOrth << std::endl;
                // assert(false);

                faceDiBjCenterBatchElem.m(0)(Eigen::seq(1, Eigen::last), Eigen::seq(1, Eigen::last)) *= LOrth.transpose();
                for (int ig = 0; ig < eFace.getNInt(); ig++)
                {
                    faceDiBjGaussBatchElem.m(ig * 2 + 0)(Eigen::seq(1, Eigen::last), Eigen::seq(1, Eigen::last)) *= LOrth.transpose();
                }
                if (f2c[1] != FACE_2_VOL_EMPTY)
                {
                    faceDiBjCenterBatchElem.m(1)(Eigen::seq(1, Eigen::last), Eigen::seq(1, Eigen::last)) *= LOrth.transpose();
                    for (int ig = 0; ig < eFace.getNInt(); ig++)
                    {
                        faceDiBjGaussBatchElem.m(ig * 2 + 1)(Eigen::seq(1, Eigen::last), Eigen::seq(1, Eigen::last)) *= LOrth.transpose();
                    }
                }
            }

            // *deal with matrixSecondaryBatch
            if (f2c[1] != FACE_2_VOL_EMPTY)
            {

                Eigen::MatrixXd msR2L, msL2R;
                // std::cout << faceDiBjCenterBatchElem.m(0) << std::endl;
                int nColL = faceDiBjCenterBatchElem.m(0).cols();
                int nRowL = faceDiBjCenterBatchElem.m(0).rows();
                int nColR = faceDiBjCenterBatchElem.m(1).cols();
                int nRowR = faceDiBjCenterBatchElem.m(1).rows();
                HardEigen::EigenLeastSquareSolve(faceDiBjCenterBatchElem.m(0).bottomRightCorner(nRowL - 1, nColL - 1),
                                                 faceDiBjCenterBatchElem.m(1).bottomRightCorner(nRowR - 1, nColR - 1), msR2L);
                HardEigen::EigenLeastSquareSolve(faceDiBjCenterBatchElem.m(1).bottomRightCorner(nRowR - 1, nColR - 1),
                                                 faceDiBjCenterBatchElem.m(0).bottomRightCorner(nRowL - 1, nColL - 1), msL2R);
                matrixSecondaryBatchElem.m(0) = msR2L;
                matrixSecondaryBatchElem.m(1) = msL2R;
                // std::cout << msR2L << std::endl;
                // std::abort();
            }
            // exit(0);

            // *Do weights!!
            // if (f2c[1] == FACE_2_VOL_EMPTY)
            //     std::cout << faceAtr.iPhy << std::endl;
            (*faceWeights)[iFace].resize(faceRecAtr.NDIFF);
            Elem::tPoint delta;
            if (f2c[1] != FACE_2_VOL_EMPTY)
            {
                delta = cellBaries[f2c[1]] - cellBaries[f2c[0]];
                (*faceWeights)[iFace].setConstant(1.0);
            }
            else if (faceAtr.iPhy == BoundaryType::Wall)
            {
                (*faceWeights)[iFace].setConstant(0.0);
                (*faceWeights)[iFace][0] = setting.wallWeight;
                // delta = (faceCoords(Eigen::all, 0) + faceCoords(Eigen::all, 1)) * 0.5 - cellCenters[f2c[0]];
                delta = pFace - cellBaries[f2c[0]];
                delta *= 1.0;
            }
            else if (faceAtr.iPhy == BoundaryType::Farfield ||
                     faceAtr.iPhy == BoundaryType::Special_DMRFar ||
                     faceAtr.iPhy == BoundaryType::Special_RTFar)
            {
                (*faceWeights)[iFace].setConstant(0.0);
                (*faceWeights)[iFace][0] = setting.farWeight;
                delta = pFace - cellBaries[f2c[0]];
            }
            else if (faceAtr.iPhy == BoundaryType::Wall_Euler)
            {
                (*faceWeights)[iFace].setConstant(0.0);
                (*faceWeights)[iFace][0] = setting.wallWeight;
                delta = pFace - cellBaries[f2c[0]];
                delta *= 1.0;
            }
            else if (faceAtr.iPhy == BoundaryType::Wall_NoSlip)
            {
                (*faceWeights)[iFace].setConstant(0.0);
                (*faceWeights)[iFace][0] = setting.wallWeight;
                delta = pFace - cellBaries[f2c[0]];
                delta *= 1.0;
            }
            else
            {
                log() << faceAtr.iPhy << std::endl;
                assert(false);
            }
            real D = delta.norm();
            real S = FV->faceArea[iFace];
            real GW = 1;
            switch (setting.weightSchemeGeom)
            {
            case Setting::WeightSchemeGeom::None:
                break;
            case Setting::WeightSchemeGeom::D:
                GW = 1. / D;
                break;
            case Setting::WeightSchemeGeom::S:
                GW = 1. / S;
                break;
            default:
                assert(false);
            }
            for (int idiff = 0; idiff < faceRecAtr.NDIFF; idiff++)
            {
                int ndx = Elem::diffOperatorOrderList2D[idiff][0];
                int ndy = Elem::diffOperatorOrderList2D[idiff][1];
                (*faceWeights)[iFace][idiff] *= GW *
                                                std::pow(delta[0], ndx) *
                                                std::pow(delta[1], ndy) *
                                                real(Elem::diffNCombs2D[idiff]) / real(Elem::factorials[ndx + ndy]);
            }
        });

    // InsertCheck(mpi, "initBaseDiffCache Ended", __FILE__, __LINE__);
}

void DNDS::VRFiniteVolume2D::initReconstructionMatVec()
{

    auto fGetMatSizes = [&](int &nmat, std::vector<int> &matSizes, index iCell)
    {
        auto &cellAttribute = mesh->cellAtrLocal[iCell][0];
        auto &cellRecAttribute = cellRecAtrLocal[iCell][0];
        auto eCell = Elem::ElementManager(cellAttribute.type, cellAttribute.intScheme);
        auto c2f = mesh->cell2faceLocal[iCell];
        nmat = c2f.size() + 1;
        matSizes.resize(nmat * 2);
        matSizes[0 * 2 + 0] = matSizes[0 * 2 + 1] = cellRecAttribute.NDOF - 1;
        for (int ic2f = 0; ic2f < int(c2f.size()); ic2f++)
        {
            index iFace = c2f[ic2f];
            auto f2c = mesh->face2cellLocal[iFace];
            index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
            index iCellAtFace = f2c[0] == iCell ? 0 : 1;
            auto &faceAttribute = mesh->faceAtrLocal[iFace][0];
            if (iCellOther != FACE_2_VOL_EMPTY)
            {
                auto &cellRecAttributeOther = cellRecAtrLocal[iCellOther][0];
                matSizes[(ic2f + 1) * 2 + 0] = cellRecAttribute.NDOF - 1;
                matSizes[(ic2f + 1) * 2 + 1] = cellRecAttributeOther.NDOF - 1;
            }
            else if (faceAttribute.iPhy != BoundaryType::Inner)
            {
                matSizes[(ic2f + 1) * 2 + 0] = cellRecAttribute.NDOF - 1;
                matSizes[(ic2f + 1) * 2 + 1] = cellRecAttribute.NDOF - 1;
            }
            else
            {
                assert(false);
            }
        }
    };

    auto fGetVecSize = [&](std::vector<int> &matSizes, index iCell)
    {
        auto &cellAttribute = mesh->cellAtrLocal[iCell][0];
        auto &cellRecAttribute = cellRecAtrLocal[iCell][0];
        auto eCell = Elem::ElementManager(cellAttribute.type, cellAttribute.intScheme);
        auto c2f = mesh->cell2faceLocal[iCell];
        matSizes.resize(2);
        matSizes[0 * 2 + 0] = c2f.size();
        matSizes[0 * 2 + 1] = cellRecAttribute.NDOF - 1;
    };

    vectorBatch = std::make_shared<decltype(vectorBatch)::element_type>(
        decltype(vectorBatch)::element_type::tContext(
            [&](index i) -> rowsize
            {
                int nmats = 1;
                std::vector<int> matSizes;
                fGetVecSize(matSizes, i);
                assert(matSizes.size() == nmats * 2);
                return DNDS::SmallMatricesBatch::predictSize(nmats, matSizes);
            },
            [&](uint8_t *data, index siz, index i)
            {
                int nmats = 1;
                std::vector<int> matSizes;
                fGetVecSize(matSizes, i);
                assert(matSizes.size() == nmats * 2);
                DNDS::SmallMatricesBatch::initializeData(data, nmats, matSizes);
            },
            mesh->cell2faceLocal.dist->size()),
        mpi);
    matrixBatch = std::make_shared<decltype(matrixBatch)::element_type>(
        decltype(matrixBatch)::element_type::tContext(
            [&](index i) -> rowsize
            {
                int nmats = -123;
                std::vector<int> matSizes;
                fGetMatSizes(nmats, matSizes, i);
                assert(matSizes.size() == nmats * 2);
                return DNDS::SmallMatricesBatch::predictSize(nmats, matSizes);
            },
            [&](uint8_t *data, index siz, index i)
            {
                int nmats = -123;
                std::vector<int> matSizes;
                fGetMatSizes(nmats, matSizes, i);
                assert(matSizes.size() == nmats * 2);
                DNDS::SmallMatricesBatch::initializeData(data, nmats, matSizes);
            },
            mesh->cell2faceLocal.dist->size()),
        mpi);

    matrixAii = std::make_shared<decltype(matrixAii)::element_type>(mesh->cell2faceLocal.dist->size());

    // for each inner cell (ghost cell no need)
    forEachInArray(
        *mesh->cell2faceLocal.dist,
        [&](tAdjArray::tComponent &c2f, index iCell)
        {
            auto &cellAttribute = mesh->cellAtrLocal[iCell][0];
            auto &cellRecAttribute = cellRecAtrLocal[iCell][0];
            auto eCell = Elem::ElementManager(cellAttribute.type, cellAttribute.intScheme);
            assert(c2f.size() == eCell.getNFace());
            auto matrixBatchElem = (*matrixBatch)[iCell];
            auto vectorBatchElem = (*vectorBatch)[iCell];

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
                Elem::ElementManager eFace(faceAttribute.type, faceRecAttribute.intScheme);
                auto faceDiBjGaussBatchElem = (*faceDiBjGaussBatch)[iFace];
                eFace.Integration(
                    A,
                    [&](Eigen::MatrixXd &incA, int ig, Elem::tPoint &ip, Elem::tDiFj &iDiNj)
                    {
                        auto diffsI = faceDiBjGaussBatchElem.m(ig * 2 + iCellAtFace);
                        Eigen::MatrixXd incAFull;
                        FFaceFunctional(iFace, ig, diffsI, diffsI, (*faceWeights)[iFace], incAFull);
                        // std::cout << diffsI << std::endl;
                        assert(incAFull(Eigen::all, 0).norm() + incAFull(0, Eigen::all).norm() == 0);
                        // std::cout << "\nincAFULL" << incAFull << std::endl;
                        // std::cout << "\ndiffsI" << diffsI << std::endl;

                        incA = incAFull.bottomRightCorner(incAFull.rows() - 1, incAFull.cols() - 1);
                        incA *= faceNorms[iFace][ig].norm(); // note: don't forget the Jacobi!!!
                    });
            }

            Eigen::MatrixXd Ainv;
            HardEigen::EigenLeastSquareInverse(A, Ainv);
            // HardEigen::EigenLeastSquareInverse_Filtered(A, Ainv);
            matrixAii->operator[](iCell) = A;
            matrixBatchElem.m(0) = Ainv;
            // if (iCell == 71)
            // {
            //     std::cout << "A " << A << std::endl;
            //     std::cout << mesh->faceAtrLocal[c2f[0]][0].iPhy << "  " << mesh->faceAtrLocal[c2f[1]][0].iPhy << " " << mesh->faceAtrLocal[c2f[2]][0].iPhy << " " << std::endl;
            //     std::cout << cellCenters[iCell] << std::endl;
            //     abort();
            // }

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++) // for each face of cell
            {
                index iFace = c2f[ic2f];
                auto f2c = mesh->face2cellLocal[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];

                index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                auto &faceAttribute = mesh->faceAtrLocal[iFace][0];
                auto &faceRecAttribute = faceRecAtrLocal[iFace][0];
                Elem::ElementManager eFace(faceAttribute.type, faceRecAttribute.intScheme);
                auto faceDiBjGaussBatchElem = (*faceDiBjGaussBatch)[iFace];

                if (iCellOther != FACE_2_VOL_EMPTY)
                {
                    auto &cellAttributeOther = mesh->cellAtrLocal[iCellOther][0];
                    auto &cellRecAttributeOther = cellRecAtrLocal[iCellOther][0];
                    Eigen::MatrixXd B;
                    B.resizeLike(matrixBatchElem.m(ic2f + 1));
                    B.setZero();
                    eFace.Integration(
                        B,
                        [&](Eigen::MatrixXd &incB, int ig, Elem::tPoint &ip, Elem::tDiFj &iDiNj)
                        {
                            auto diffsI = faceDiBjGaussBatchElem.m(ig * 2 + iCellAtFace);
                            auto diffsJ = faceDiBjGaussBatchElem.m(ig * 2 + 1 - iCellAtFace);
                            Eigen::MatrixXd incBFull;
                            FFaceFunctional(iFace, ig, diffsI, diffsJ, (*faceWeights)[iFace], incBFull);
                            assert(incBFull(Eigen::all, 0).norm() + incBFull(0, Eigen::all).norm() == 0);
                            incB = incBFull.bottomRightCorner(incBFull.rows() - 1, incBFull.cols() - 1);
                            incB *= faceNorms[iFace][ig].norm(); // note: don't forget the
                            // std::cout << "DI " << std::endl;
                            // std::cout << faceDiBjGaussCache[iFace][ig * 2 + iCellFace] << std::endl;
                        });
                    matrixBatchElem.m(ic2f + 1) = Ainv * B;
                }
                else if (faceAttribute.iPhy == BoundaryType::Wall ||
                         faceAttribute.iPhy == BoundaryType::Wall_Euler ||
                         faceAttribute.iPhy == BoundaryType::Wall_NoSlip)
                {
                    matrixBatchElem.m(ic2f + 1).setZero(); // the other 'cell' has no rec
                }
                else if (faceAttribute.iPhy == BoundaryType::Farfield ||
                         faceAttribute.iPhy == BoundaryType::Special_DMRFar ||
                         faceAttribute.iPhy == BoundaryType::Special_RTFar)
                {
                    Eigen::MatrixXd B;
                    B.resizeLike(matrixBatchElem.m(ic2f + 1));
                    B.setZero();
                    eFace.Integration(
                        B,
                        [&](Eigen::MatrixXd &incA, int ig, Elem::tPoint &ip, Elem::tDiFj &iDiNj)
                        {
                            auto diffsI = faceDiBjGaussBatchElem.m(ig * 2 + iCellAtFace);
                            Eigen::MatrixXd incAFull;
                            FFaceFunctional(iFace, ig, diffsI, diffsI, (*faceWeights)[iFace], incAFull);
                            // std::cout << "W " << faceWeights[iFace] << std::endl;
                            assert(incAFull(Eigen::all, 0).norm() + incAFull(0, Eigen::all).norm() == 0);
                            incA = incAFull.bottomRightCorner(incAFull.rows() - 1, incAFull.cols() - 1);
                            incA *= faceNorms[iFace][ig].norm(); // note: don't forget the Jacobi!!!
                        });
                    matrixBatchElem.m(ic2f + 1) = Ainv * B;
                }
                else
                {
                    assert(false);
                }
                // everyone is welcome to have this
                {
                    Eigen::RowVectorXd row(vectorBatchElem.m(0).cols());
                    row.setZero();
                    eFace.Integration(
                        row,
                        [&](Eigen::RowVectorXd &incb, int ig, Elem::tPoint &ip, Elem::tDiFj &iDiNj)
                        {
                            auto diffsI = faceDiBjGaussBatchElem.m(ig * 2 + iCellAtFace);
                            Eigen::MatrixXd incbFull;
                            Eigen::MatrixXd rowDiffI = diffsI({0}, Eigen::all);
                            Eigen::MatrixXd fw0 = (*faceWeights)[iFace]({0});
                            FFaceFunctional(iFace, ig, Eigen::MatrixXd::Ones(1, 1), rowDiffI, fw0, incbFull);
                            // std::cout << incbFull(0, 0) << " " << incbFull.size() << incb.size() << std::endl;
                            assert(incbFull(0, 0) == 0);
                            incb = incbFull.rightCols(incbFull.size() - 1);
                            incb *= faceNorms[iFace][ig].norm(); // note: don't forget the
                            // std::cout << "DI " << std::endl;
                            // std::cout << faceDiBjGaussCache[iFace][ig * 2 + iCellFace] << std::endl;
                        });
                    vectorBatchElem.m(0).row(ic2f) = row;
                }
            }
            vectorBatchElem.m(0) = vectorBatchElem.m(0) * Ainv.transpose(); // must be outside the loop as it operates all rows at once
        });
}