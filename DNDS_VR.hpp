#pragma once

#include "DNDS_Mesh.hpp"

namespace DNDS
{
    struct RecAtr
    {
        uint8_t NDOF = -1;
        uint8_t NDIFF = -1;
        Elem::tIntScheme intScheme = -100;
    };

    class ImplicitFiniteVolume2D
    {
    public:
        MPIInfo mpi;
        CompactFacedMeshSerialRW *mesh;
        std::vector<real> volumeLocal;

        ImplicitFiniteVolume2D(CompactFacedMeshSerialRW *nMesh) : mesh(nMesh), mpi(nMesh->mpi)
        {
            if (!Elem::ElementManager::NBufferInit)
                Elem::ElementManager::InitNBuffer(); //! do not asuume it't been initialized
            Elem::tIntScheme schemeTri = Elem::INT_SCHEME_TRI_7;
            Elem::tIntScheme schemeQuad = Elem::INT_SCHEME_QUAD_16;
            volumeLocal.resize(mesh->cell2nodeLocal.size());

            // std::cout << mpi.rank << " " << mesh->cell2nodeLocal.size() << std::endl;
            forEachInArrayPair( // get volumes
                *mesh->cell2nodeLocal.pair,
                [&](tAdjArrayCascade::tComponent &c2n, index iv)
                {
                    auto atr = mesh->cellAtrLocal[iv][0];
                    Elem::ElementManager elemMan(atr.type, 0);
                    switch (elemMan.getPspace())
                    {
                    case Elem::ParamSpace::TriSpace:
                        elemMan.setType(atr.type, schemeTri);
                        break;
                    case Elem::ParamSpace::QuadSpace:
                        elemMan.setType(atr.type, schemeQuad);
                        break;
                    default:
                        assert(false);
                    }
                    real v = 0.0;
                    Eigen::MatrixXd coords(3, elemMan.getNNode());
                    // for (int in = 0; in < elemMan.getNNode(); in++)
                    //     coords(Eigen::all, in) = mesh->nodeCoordsLocal[c2n[in]].p(); // the coords data ordering is conforming with that of cell2node
                    mesh->LoadCoords(c2n, coords);
                    elemMan.Integration(
                        v,
                        [&](real &vinc, int m, DNDS::Elem::tPoint &p, DNDS::Elem::tDiFj &DiNj) -> void
                        {
                            vinc = DNDS::Elem::DiNj2Jacobi(DiNj, coords)({0, 1}, {0, 1}).determinant();
                            if (vinc < 0)
                                log() << "Error: 2d vol orientation wrong or distorted" << std::endl;
                            assert(vinc > 0);
                        });
                    volumeLocal[iv] = v;
                    // if (mpi.rank == 0)
                    //     std::cout << "V " << v << std::endl;
                });
        }
    };

    class VRFiniteVolume2D
    {
    public:
        MPIInfo mpi;
        const static int P_ORDER = 3;
        typedef ArrayCascade<SmallMatricesBatch> tMatArray;
        CompactFacedMeshSerialRW *mesh; // a mere reference to mesh, user responsible for keeping it valid
        ImplicitFiniteVolume2D *FV;
        ArrayCascadeLocal<Batch<RecAtr, 1>> cellRecAtrLocal;
        ArrayCascadeLocal<Batch<RecAtr, 1>> faceRecAtrLocal;

        // ArrayCascadeLocal<SmallMatricesBatch> cellDiBjGaussCache;  // DiBjCache[i].m(iGauss) = Di(Bj) at this gaussPoint, Di is diff according to diffOperatorOrderList2D[i][:]
        // ArrayCascadeLocal<SmallMatricesBatch> cellDiBjCenterCache; // center, only order 0, 1 diffs
        // ArrayCascadeLocal<SmallMatricesBatch> faceDiBjGaussCache;  // DiBjCache[i].m(iGauss * 2 + 0/1) = Di(Bj) at this gaussPoint, Di is diff according to diffOperatorOrderList2D[i][:]
        // ArrayCascadeLocal<SmallMatricesBatch> faceDiBjCenterCache; // center, only order 0, 1 diffs
        // ArrayCascadeLocal<VarVector> baseMoment;                   // full dofs, like baseMoment[i].v()(0) == 1

        std::vector<std::vector<Eigen::MatrixXd>> cellDiBjGaussCache;                 // DiBjCache[i].m(iGauss) = Di(Bj) at this gaussPoint, Di is diff according to diffOperatorOrderList2D[i][:]
        std::vector<Eigen::MatrixXd> cellDiBjCenterCache;                             // center, only order 0, 1 diffs
        std::vector<std::vector<Eigen::MatrixXd>> faceDiBjGaussCache;                 // DiBjCache[i].m(iGauss * 2 + 0/1) = Di(Bj) at this gaussPoint, Di is diff according to diffOperatorOrderList2D[i][:]
        std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> faceDiBjCenterCache; // center, only order 0, 1 diffs
        std::vector<Eigen::VectorXd> baseMoments;                                     // full dofs, like baseMoment[i].v()(0) == 1
        std::vector<Elem::tPoint> cellCenters;
        std::vector<std::vector<Elem::tPoint>> faceNorms;

        ArrayCascadeLocal<SmallMatricesBatch> vectorInvAb; // invAb[i].m(0).row(icf) = the A^-1b of cell i's icf neighbour
        ArrayCascadeLocal<SmallMatricesBatch> matrixInvAB; // matrixInvAB[i].m(icf + 1) = the A^-1B of cell i's icf neighbour, invAb[i].m(0) is cell i's A^-1
                                                           // note that the dof dimensions of these rec data excludes the mean-value/const-rec dof

        VRFiniteVolume2D(CompactFacedMeshSerialRW *nMesh, ImplicitFiniteVolume2D *nFV) : mesh(nMesh), FV(nFV), mpi(nMesh->mpi)
        {
            assert(FV->mpi == mpi);
        }

        static int PolynomialNDOF(int order) //  2-d specific
        {
            return (order + 2) * (order + 1) / 2;
        }

        inline static real FPolynomial3D(int px, int py, int pz, int dx, int dy, int dz, real x, real y, real z)
        {
            int c = Elem::dFactorials[px][dx] * Elem::dFactorials[py][dy] * Elem::dFactorials[pz][dz];
            return c ? c * Elem::iPow(px - dx, x) * Elem::iPow(py - dy, y) * Elem::iPow(pz - dz, z) : 0.;
        }

        inline static Elem::tPoint CoordMinMaxScale(const Eigen::MatrixXd &coords)
        {
            return Elem::tPoint{
                coords(0, Eigen::all).maxCoeff() - coords(0, Eigen::all).minCoeff(),
                coords(1, Eigen::all).maxCoeff() - coords(1, Eigen::all).minCoeff(),
                coords(2, Eigen::all).maxCoeff() - coords(2, Eigen::all).minCoeff()};
        }

        static void FDiffBaseValue(Elem::ElementManager &cElem,
                                   const Eigen::MatrixXd &coords,
                                   const Elem::tDiFj &DiNj, //@ pParam
                                   const Elem::tPoint &pParam,
                                   const Elem::tPoint &cPhysics,
                                   const Elem::tPoint &simpleScale,
                                   const Eigen::VectorXd &baseMoment,
                                   Eigen::MatrixXd &DiBj) // for 2d polynomials here
        {
            assert(coords.cols() == cElem.getNNode() && DiNj.cols() == cElem.getNNode());
            Elem::tPoint pParamC = pParam - cElem.getCenterPParam();
            Elem::tPoint pPhysics = coords * DiNj(0, Eigen::all).transpose();
            Elem::tPoint pPhysicsC = pPhysics - cPhysics;
            Elem::tPoint pPhysicsCScaled = (pPhysicsC.array() / simpleScale.array()).matrix();
            pPhysicsCScaled(2) = 0.; // for 2d volumes
            for (int idiff = 0; idiff < DiBj.rows(); idiff++)
                for (int ibase = 0; ibase < DiBj.cols(); ibase++)
                    DiBj(idiff, ibase) = FPolynomial3D(
                        Elem::diffOperatorOrderList2D[ibase][0],
                        Elem::diffOperatorOrderList2D[ibase][1],
                        Elem::diffOperatorOrderList2D[ibase][2],
                        Elem::diffOperatorOrderList2D[idiff][0],
                        Elem::diffOperatorOrderList2D[idiff][1],
                        Elem::diffOperatorOrderList2D[idiff][2],
                        pPhysicsCScaled(0), pPhysicsCScaled(1), pPhysicsCScaled(2));
            // std::cout << DiBj << std::endl;
            // exit(-1);
            DiBj(0, Eigen::all) -= baseMoment.transpose();
        }

        // derive intscheme, ndof ,ndiff in rec attributes
        void initIntScheme() //  2-d specific
        {
            cellRecAtrLocal.dist = std::make_shared<decltype(cellRecAtrLocal.dist)::element_type>(
                decltype(cellRecAtrLocal.dist)::element_type::tComponent::Context(mesh->cellAtrLocal.dist->size()), mpi);
            cellRecAtrLocal.CreateGhostCopyComm(mesh->cellAtrLocal);

            forEachInArray(
                *mesh->cellAtrLocal.dist,
                [&](tElemAtrArrayCascade::tComponent &atr, index iCell)
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
                        recAtr.intScheme = Elem::INT_SCHEME_QUAD_9;
                        recAtr.NDOF = PolynomialNDOF(P_ORDER);
                        recAtr.NDIFF = PolynomialNDOF(P_ORDER);
                        break;
                    default:
                        assert(false);
                        break;
                    }
                });
            cellRecAtrLocal.PullOnce();
            faceRecAtrLocal.dist = std::make_shared<decltype(faceRecAtrLocal.dist)::element_type>(
                decltype(faceRecAtrLocal.dist)::element_type::tComponent::Context(mesh->faceAtrLocal.dist->size()), mpi);

            faceRecAtrLocal.CreateGhostCopyComm(mesh->faceAtrLocal);

            forEachInArray(
                *mesh->faceAtrLocal.dist,
                [&](tElemAtrArrayCascade::tComponent &atr, index iFace)
                {
                    Elem::ElementManager eFace(atr[0].type, atr[0].intScheme);
                    auto &recAtr = faceRecAtrLocal[iFace][0];
                    switch (eFace.getPspace())
                    {
                    case Elem::ParamSpace::LineSpace:
                        recAtr.intScheme = Elem::INT_SCHEME_LINE_3;
                        recAtr.NDOF = PolynomialNDOF(P_ORDER);
                        recAtr.NDIFF = PolynomialNDOF(P_ORDER);
                        break;
                    default:
                        assert(false);
                        break;
                    }
                });
            faceRecAtrLocal.PullOnce();
        }

        void initMoment()
        {
            InsertCheck(mpi, "InitMomentStart");
            index nlocalCells = mesh->cell2nodeLocal.size();
            baseMoments.resize(nlocalCells);
            cellCenters.resize(nlocalCells);
            forEachInArrayPair(
                *mesh->cell2nodeLocal.pair,
                [&](tAdjArrayCascade::tComponent &c2n, index iCell)
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
                    Eigen::MatrixXd BjBuffer(1, cellRecAtr.NDOF);
                    BjBuffer.setZero();
                    eCell.Integration(
                        BjBuffer,
                        [&](Eigen::MatrixXd &incBj, int ig, Elem::tPoint &ip, Elem::tDiFj &iDiNj)
                        {
                            // InsertCheck(mpi, "Do2");
                            incBj.resize(1, cellRecAtr.NDOF);
                            // InsertCheck(mpi, "Do2End");
                            FDiffBaseValue(eCell, coords, iDiNj,
                                           ip, cellCenters[iCell], sScale,
                                           Eigen::VectorXd::Zero(cellRecAtr.NDOF),
                                           incBj);
                            Elem::tJacobi Jacobi = Elem::DiNj2Jacobi(iDiNj, coords);
                            incBj *= Jacobi({0, 1}, {0, 1}).determinant();
                            // std::cout << "IncBj 0 " << incBj(0, 0) << std::endl;
                        });
                    // std::cout << "BjBuffer0 " << BjBuffer(0, 0) << std::endl;
                    baseMoments[iCell] = (BjBuffer / FV->volumeLocal[iCell]).transpose();
                    // std::cout << "BM0 " << 1 - baseMoments[iCell](0) << std::endl; //should better be machine eps
                    baseMoments[iCell](0) = 1.;
                });
            InsertCheck(mpi, "InitMomentEnd");
        }

        void initBaseDiffCache()
        {
            InsertCheck(mpi, "initBaseDiffCache");
            index nlocalCells = mesh->cell2nodeLocal.size();
            cellDiBjCenterCache.resize(nlocalCells);
            cellDiBjGaussCache.resize(nlocalCells);
            forEachInArrayPair(
                *mesh->cell2nodeLocal.pair,
                [&](tAdjArrayCascade::tComponent &c2n, index iCell)
                {
                    auto cellRecAtr = cellRecAtrLocal[iCell][0];
                    auto cellAtr = mesh->cellAtrLocal[iCell][0];
                    Elem::ElementManager eCell(cellAtr.type, cellRecAtr.intScheme);
                    // center part
                    Elem::tPoint p = eCell.getCenterPParam();
                    cellDiBjCenterCache[iCell].resize(cellRecAtr.NDIFF, cellRecAtr.NDOF);
                    Eigen::MatrixXd coords;
                    mesh->LoadCoords(c2n, coords);
                    Eigen::MatrixXd DiNj(4, eCell.getNNode());
                    eCell.GetDiNj(p, DiNj); // N_cache not using
                    Elem::tPoint sScale = CoordMinMaxScale(coords);
                    FDiffBaseValue(eCell, coords, DiNj,
                                   p, cellCenters[iCell], sScale,
                                   baseMoments[iCell],
                                   cellDiBjCenterCache[iCell]);

                    // iGaussPart
                    cellDiBjGaussCache[iCell].resize(eCell.getNInt());
                    for (int ig = 0; ig < eCell.getNInt(); ig++)
                    {
                        eCell.GetIntPoint(ig, p);
                        cellDiBjGaussCache[iCell][ig].resize(cellRecAtr.NDIFF, cellRecAtr.NDOF);
                        eCell.GetDiNj(p, DiNj); // N_cache not using
                        FDiffBaseValue(eCell, coords, DiNj,
                                       p, cellCenters[iCell], sScale,
                                       baseMoments[iCell],
                                       cellDiBjGaussCache[iCell][ig]);
                    }
                });
            InsertCheck(mpi, "initBaseDiffCache Cell Ended");
            // face part: sides
            index nlocalFaces = mesh->face2nodeLocal.size();
            faceDiBjCenterCache.resize(nlocalFaces);
            faceDiBjGaussCache.resize(nlocalFaces);
            faceNorms.resize(nlocalFaces);
            forEachInArrayPair(
                *mesh->face2cellLocal.pair,
                [&](tAdjStatic2ArrayCascade::tComponent &f2c, index iFace)
                {
                    auto faceRecAtr = faceRecAtrLocal[iFace][0];
                    auto faceAtr = mesh->faceAtrLocal[iFace][0];
                    Elem::ElementManager eFace(faceAtr.type, faceRecAtr.intScheme);
                    auto f2n = mesh->face2nodeLocal[iFace];
                    faceDiBjGaussCache[iFace].resize(eFace.getNInt() * 2); // note the 2x size
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

                    // Left side
                    index iCell = f2c[0];
                    Elem::tPoint pCell;
                    mesh->FacePParam2Cell(iCell, 0, iFace, f2n, eFace, pFace, pCell);
                    Elem::ElementManager eCell(cellAtrL->type, 0); // int scheme is not relevant here
                    faceDiBjCenterCache[iFace].first.resize(faceRecAtr.NDIFF, cellRecAtrL->NDOF);
                    Elem::tDiFj cellDiNj(4, eCell.getNNode());
                    eCell.GetDiNj(pCell, cellDiNj);
                    FDiffBaseValue(eCell, cellCoordsL, cellDiNj,
                                   pCell, cellCenters[iCell], sScaleL,
                                   baseMoments[iCell], faceDiBjCenterCache[iFace].first);

                    if (f2c[1] != FACE_2_VOL_EMPTY)
                    {
                        // right side
                        index iCell = f2c[1];
                        Elem::tPoint pCell; // IND identical apart from "second", R
                        // InsertCheck(mpi, "DO1");
                        mesh->FacePParam2Cell(iCell, 1, iFace, f2n, eFace, pFace, pCell);
                        // InsertCheck(mpi, "DO1End");
                        Elem::ElementManager eCell(cellAtrR->type, 0); // int scheme is not relevant here
                        faceDiBjCenterCache[iFace].second.resize(faceRecAtr.NDIFF, cellRecAtrR->NDOF);
                        Elem::tDiFj cellDiNj(4, eCell.getNNode());
                        eCell.GetDiNj(pCell, cellDiNj);
                        FDiffBaseValue(eCell, cellCoordsR, cellDiNj,
                                       pCell, cellCenters[iCell], sScaleR,
                                       baseMoments[iCell], faceDiBjCenterCache[iFace].second);
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
                        faceDiBjGaussCache[iFace][ig * 2 + 0].resize(faceRecAtr.NDIFF, cellRecAtrL->NDOF);
                        Elem::tDiFj cellDiNj(4, eCell.getNNode());
                        eCell.GetDiNj(pCell, cellDiNj);
                        FDiffBaseValue(eCell, cellCoordsL, cellDiNj,
                                       pCell, cellCenters[iCell], sScaleL,
                                       baseMoments[iCell], faceDiBjGaussCache[iFace][ig * 2 + 0]);
                        // InsertCheck(mpi, "DOAEND");
                        if (f2c[1] != FACE_2_VOL_EMPTY)
                        {
                            // Right side
                            index iCell = f2c[1];
                            Elem::tPoint pCell; // IND identical apart from faceDiBjGaussCache[iFace][ig * 2 + 1], R
                            mesh->FacePParam2Cell(iCell, 1, iFace, f2n, eFace, pFace, pCell);
                            Elem::ElementManager eCell(cellAtrR->type, 0); // int scheme is not relevant here
                            faceDiBjGaussCache[iFace][ig * 2 + 1].resize(faceRecAtr.NDIFF, cellRecAtrR->NDOF);
                            Elem::tDiFj cellDiNj(4, eCell.getNNode());
                            eCell.GetDiNj(pCell, cellDiNj);
                            FDiffBaseValue(eCell, cellCoordsR, cellDiNj,
                                           pCell, cellCenters[iCell], sScaleR,
                                           baseMoments[iCell], faceDiBjGaussCache[iFace][ig * 2 + 1]);
                        }
                    }
                });

            InsertCheck(mpi, "initBaseDiffCache Ended");
        }

        void initReconstructionMatVec()
        {

            // for each inner cell (ghost cell no need)
            forEachInArray(
                *mesh->cell2faceLocal.dist,
                [&](tAdjArrayCascade::tComponent &c2f, index iCell)
                {
                    auto &cellAttribute = mesh->cellAtr->operator[](iCell)[0];
                    auto eCell = Elem::ElementManager(cellAttribute.type, cellAttribute.intScheme);
                    assert(c2f.size() == eCell.getNFace());
                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++) // for each face of cell
                    {
                        index iFace = c2f[ic2f];
                        auto f2c = (*mesh->face2cellPair)[iFace];
                        index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                    }
                });
        }
    };
}