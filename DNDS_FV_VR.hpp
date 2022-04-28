#pragma once

#include "DNDS_Mesh.hpp"
#include "DNDS_HardEigen.h"

namespace DNDS
{

    struct RecAtr
    {
        real relax = UnInitReal;
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
        std::vector<real> faceArea;

        ImplicitFiniteVolume2D(CompactFacedMeshSerialRW *nMesh) : mpi(nMesh->mpi), mesh(nMesh)
        {
            if (!Elem::ElementManager::NBufferInit)
                Elem::ElementManager::InitNBuffer(); //! do not asuume it't been initialized
            Elem::tIntScheme schemeTri = Elem::INT_SCHEME_TRI_4;
            Elem::tIntScheme schemeQuad = Elem::INT_SCHEME_QUAD_9;
            Elem::tIntScheme schemeLine = Elem::INT_SCHEME_LINE_5;
            volumeLocal.resize(mesh->cell2nodeLocal.size());
            faceArea.resize(mesh->face2nodeLocal.size());

            // std::cout << mpi.rank << " " << mesh->cell2nodeLocal.size() << std::endl;
            forEachInArrayPair( // get volumes
                *mesh->cell2nodeLocal.pair,
                [&](tAdjArray::tComponent &c2n, index iv)
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
            forEachInArrayPair(
                *mesh->face2nodeLocal.pair,
                [&](tAdjArray::tComponent &f2n, index iff)
                {
                    auto atr = mesh->faceAtrLocal[iff][0];
                    Elem::ElementManager elemMan(atr.type, 0);
                    switch (elemMan.getPspace())
                    {
                    case Elem::ParamSpace::LineSpace:
                        elemMan.setType(atr.type, schemeLine);
                        break;
                    default:
                        assert(false);
                    }
                    real v = 0.0;
                    Eigen::MatrixXd coords;
                    mesh->LoadCoords(f2n, coords);
                    elemMan.Integration(
                        v,
                        [&](real &vinc, int m, DNDS::Elem::tPoint &p, DNDS::Elem::tDiFj &DiNj) -> void
                        {
                            vinc = DNDS::Elem::Jacobi2LineNorm2D(DNDS::Elem::DiNj2Jacobi(DiNj, coords)).norm();
                        });
                    faceArea[iff] = v;
                    // if (mpi.rank == 0)
                    //     std::cout << "V " << v << std::endl;
                });
        }

        template <uint32_t vsize>
        // static const int vsize = 1; // intellisense helper: give example...
        void BuildMean(ArrayLocal<VecStaticBatch<vsize>> &u)
        {
            index nCellDist = mesh->cell2nodeLocal.dist->size();
            u.dist = std::make_shared<typename decltype(u.dist)::element_type>(
                typename decltype(u.dist)::element_type::tComponent::Context(nCellDist), mpi);
            u.CreateGhostCopyComm(mesh->cellAtrLocal);
            u.InitPersistentPullClean();
        }
    };

    class VRFiniteVolume2D
    {
    public:
        MPIInfo mpi;
        const static int P_ORDER = 3;
        typedef Array<SmallMatricesBatch> tMatArray;
        CompactFacedMeshSerialRW *mesh; // a mere reference to mesh, user responsible for keeping it valid
        ImplicitFiniteVolume2D *FV;
        ArrayLocal<Batch<RecAtr, 1>> cellRecAtrLocal;
        ArrayLocal<Batch<RecAtr, 1>> faceRecAtrLocal;

        // ArrayLocal<SmallMatricesBatch> cellDiBjGaussCache;  // DiBjCache[i].m(iGauss) = Di(Bj) at this gaussPoint, Di is diff according to diffOperatorOrderList2D[i][:]
        // ArrayLocal<SmallMatricesBatch> cellDiBjCenterCache; // center, only order 0, 1 diffs
        // ArrayLocal<SmallMatricesBatch> faceDiBjGaussCache;  // DiBjCache[i].m(iGauss * 2 + 0/1) = Di(Bj) at this gaussPoint, Di is diff according to diffOperatorOrderList2D[i][:]
        // ArrayLocal<SmallMatricesBatch> faceDiBjCenterCache; // center, only order 0, 1 diffs
        // ArrayLocal<VarVector> baseMoment;                   // full dofs, like baseMoment[i].v()(0) == 1

        std::vector<std::vector<Eigen::MatrixXd>> cellDiBjGaussCache;                 // DiBjCache[i].m(iGauss) = Di(Bj) at this gaussPoint, Di is diff according to diffOperatorOrderList2D[i][:]
        std::vector<Eigen::MatrixXd> cellDiBjCenterCache;                             // center, only order 0, 1 diffs
        std::vector<std::vector<Eigen::MatrixXd>> faceDiBjGaussCache;                 // DiBjCache[i].m(iGauss * 2 + 0/1) = Di(Bj) at this gaussPoint, Di is diff according to diffOperatorOrderList2D[i][:]
        std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> faceDiBjCenterCache; // center, only order 0, 1 diffs
        std::vector<Eigen::VectorXd> baseMoments;                                     // full dofs, like baseMoment[i].v()(0) == 1
        std::vector<Elem::tPoint> cellCenters;
        std::vector<Elem::tPoint> cellBaries;
        std::vector<std::vector<real>> cellGaussJacobiDets;
        std::vector<Elem::tPoint> faceCenters;
        std::vector<std::vector<Elem::tPoint>> faceNorms;
        std::vector<Elem::tPoint> faceNormCenter;
        std::shared_ptr<std::vector<Eigen::VectorXd>> faceWeights;

        std::shared_ptr<std::vector<Eigen::MatrixXd>> vectorInvAb;              // invAb[i][icf] = the A^-1b of cell i's icf neighbour
        std::shared_ptr<std::vector<std::vector<Eigen::MatrixXd>>> matrixInvAB; // matrixInvAB[i][icf + 1] = the A^-1B of cell i's icf neighbour, invAb[i].m(0) is cell i's A^-1
                                                                                // note that the dof dimensions of these rec data excludes the mean-value/const-rec dof

        VRFiniteVolume2D(CompactFacedMeshSerialRW *nMesh, ImplicitFiniteVolume2D *nFV) : mpi(nMesh->mpi), mesh(nMesh), FV(nFV)
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
            // return c ? c * Elem::iPow(px - dx, x) * Elem::iPow(py - dy, y) * Elem::iPow(pz - dz, z) : 0.;
            return c ? c * std::pow(x, px - dx) * std::pow(y, py - dy) * std::pow(z, pz - dz) : 0.;
        }

        inline static Elem::tPoint CoordMinMaxScale(const Eigen::MatrixXd &coords)
        {
            return Elem::tPoint{
                       coords(0, Eigen::all).maxCoeff() - coords(0, Eigen::all).minCoeff(),
                       coords(1, Eigen::all).maxCoeff() - coords(1, Eigen::all).minCoeff(),
                       coords(2, Eigen::all).maxCoeff() - coords(2, Eigen::all).minCoeff()} *
                   0.5;
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
            // static int i = 0;
            // if (i == 9)
            //     exit(0);
            assert(coords.cols() == cElem.getNNode() && DiNj.cols() == cElem.getNNode());
            Elem::tPoint pParamC = pParam - cElem.getCenterPParam();
            Elem::tPoint pPhysics = coords * DiNj(0, Eigen::all).transpose();
            Elem::tPoint pPhysicsC = pPhysics - cPhysics;
            Elem::tPoint pPhysicsCScaled = (pPhysicsC.array() / simpleScale.array()).matrix();
            // std::cout << "CHECK FDBV" << std::endl;
            // std::cout << pPhysicsC << std::endl;
            // std::cout << pPhysicsCScaled << std::endl;
            // std::cout << simpleScale << std::endl;
            pPhysicsCScaled(2) = 0.; // for 2d volumes
            for (int idiff = 0; idiff < DiBj.rows(); idiff++)
                for (int ibase = 0; ibase < DiBj.cols(); ibase++)
                {
                    int px = Elem::diffOperatorOrderList2D[ibase][0];
                    int py = Elem::diffOperatorOrderList2D[ibase][1];
                    int pz = Elem::diffOperatorOrderList2D[ibase][2];
                    int ndx = Elem::diffOperatorOrderList2D[idiff][0];
                    int ndy = Elem::diffOperatorOrderList2D[idiff][1];
                    int ndz = Elem::diffOperatorOrderList2D[idiff][2];
                    DiBj(idiff, ibase) = FPolynomial3D(px, py, pz, ndx, ndy, ndz,
                                                       pPhysicsCScaled(0), pPhysicsCScaled(1), pPhysicsCScaled(2)) /
                                         (std::pow(simpleScale(0), ndx) * std::pow(simpleScale(1), ndy) * std::pow(1, ndz));
                }
            // std::cout << "XXXXXXX\n"
            //           << cPhysics << std::endl
            //           << pPhysicsC << std::endl;
            // std::cout << DiBj << std::endl;
            DiBj(0, Eigen::all) -= baseMoment.transpose();
            // i++;
        }

        static void FFaceFunctional(const Eigen::MatrixXd &DiffI, const Eigen::MatrixXd &DiffJ, const Eigen::VectorXd &Weights, Eigen::MatrixXd &Conj)
        {
            assert(Weights.size() == DiffI.rows() && DiffI.rows() == DiffJ.rows()); // has same n diffs
            Conj = DiffI.transpose() * Weights.asDiagonal() * Weights.asDiagonal() * DiffJ;
            // Conj.resize(DiffI.cols(), DiffJ.cols());
            // Conj.setZero();
            // for (int i = 0; i < DiffI.cols(); i++)
            //     for (int j = 0; j < DiffJ.cols(); j++)
            //         for (int k = 0; k < DiffI.rows(); k++)
            //             Conj(i, j) += DiffI(k, i) * DiffJ(k, j) * Weights(k) * Weights(k);
        }

        // derive intscheme, ndof ,ndiff in rec attributes
        void initIntScheme() //  2-d specific
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
                        recAtr.intScheme = Elem::INT_SCHEME_QUAD_9;
                        recAtr.NDOF = PolynomialNDOF(P_ORDER);
                        recAtr.NDIFF = PolynomialNDOF(P_ORDER);
                        break;
                    default:
                        assert(false);
                        break;
                    }
                    recAtr.relax = 1.;
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
            // InsertCheck(mpi, "InitMomentStart");
            index nlocalCells = mesh->cell2nodeLocal.size();
            baseMoments.resize(nlocalCells);
            cellCenters.resize(nlocalCells);
            cellBaries.resize(nlocalCells);
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
                                           ip, cellBaries[iCell], sScale,
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

        void initBaseDiffCache()
        {
            // InsertCheck(mpi, "initBaseDiffCache");
            index nlocalCells = mesh->cell2nodeLocal.size();
            cellDiBjCenterCache.resize(nlocalCells);
            cellDiBjGaussCache.resize(nlocalCells);
            cellGaussJacobiDets.resize(nlocalCells);
            forEachInArrayPair(
                *mesh->cell2nodeLocal.pair,
                [&](tAdjArray::tComponent &c2n, index iCell)
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
                                   p, cellBaries[iCell], sScale,
                                   baseMoments[iCell],
                                   cellDiBjCenterCache[iCell]);

                    // iGaussPart
                    cellDiBjGaussCache[iCell].resize(eCell.getNInt());
                    cellGaussJacobiDets[iCell].resize(eCell.getNInt());
                    for (int ig = 0; ig < eCell.getNInt(); ig++)
                    {
                        eCell.GetIntPoint(ig, p);
                        cellDiBjGaussCache[iCell][ig].resize(cellRecAtr.NDIFF, cellRecAtr.NDOF);
                        eCell.GetDiNj(p, DiNj); // N_cache not using
                        FDiffBaseValue(eCell, coords, DiNj,
                                       p, cellBaries[iCell], sScale,
                                       baseMoments[iCell],
                                       cellDiBjGaussCache[iCell][ig]);
                        cellGaussJacobiDets[iCell][ig] = Elem::DiNj2Jacobi(DiNj, coords)({0, 1}, {0, 1}).determinant();
                    }
                });
            // InsertCheck(mpi, "initBaseDiffCache Cell Ended");

            // face part: sides
            index nlocalFaces = mesh->face2nodeLocal.size();
            faceDiBjCenterCache.resize(nlocalFaces);
            faceDiBjGaussCache.resize(nlocalFaces);
            faceNorms.resize(nlocalFaces);
            faceWeights = std::make_shared<decltype(faceWeights)::element_type>(nlocalFaces);
            faceCenters.resize(nlocalFaces);
            faceNormCenter.resize(nlocalFaces);
            forEachInArrayPair(
                *mesh->face2cellLocal.pair,
                [&](tAdjStatic2Array::tComponent &f2c, index iFace)
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
                    faceDiBjCenterCache[iFace].first.resize(faceRecAtr.NDIFF, cellRecAtrL->NDOF);
                    Elem::tDiFj cellDiNj(4, eCell.getNNode());
                    eCell.GetDiNj(pCell, cellDiNj);
                    FDiffBaseValue(eCell, cellCoordsL, cellDiNj,
                                   pCell, cellBaries[iCell], sScaleL,
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
                                       pCell, cellBaries[iCell], sScaleR,
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
                                       pCell, cellBaries[iCell], sScaleL,
                                       baseMoments[iCell], faceDiBjGaussCache[iFace][ig * 2 + 0]);
                        // std::cout << "GP" << cellCoordsL * cellDiNj(0, Eigen::all).transpose() << std::endl;
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
                                           pCell, cellBaries[iCell], sScaleR,
                                           baseMoments[iCell], faceDiBjGaussCache[iFace][ig * 2 + 1]);
                        }
                    }
                    // exit(0);

                    // Do weights!!
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
                        (*faceWeights)[iFace][0] = 1.0;
                        // delta = (faceCoords(Eigen::all, 0) + faceCoords(Eigen::all, 1)) * 0.5 - cellCenters[f2c[0]];
                        delta = pFace - cellBaries[f2c[0]];
                        delta *= 1.0;
                    }
                    else if (faceAtr.iPhy == BoundaryType::Farfield)
                    {
                        (*faceWeights)[iFace].setConstant(0.0);
                        (*faceWeights)[iFace][0] = 1.0;
                        delta = pFace - cellBaries[f2c[0]];
                    }
                    else
                    {
                        log() << faceAtr.iPhy << std::endl;
                        assert(false);
                    }
                    for (int idiff = 0; idiff < faceRecAtr.NDIFF; idiff++)
                    {
                        int ndx = Elem::diffOperatorOrderList2D[idiff][0];
                        int ndy = Elem::diffOperatorOrderList2D[idiff][1];
                        (*faceWeights)[iFace][idiff] *= std::pow(delta[0], ndx) *
                                                        std::pow(delta[1], ndy) *
                                                        real(Elem::diffNCombs2D[idiff]) / real(Elem::factorials[ndx + ndy]);
                    }
                });

            // InsertCheck(mpi, "initBaseDiffCache Ended", __FILE__, __LINE__);
        }

        void initReconstructionMatVec()
        {

            vectorInvAb = std::make_shared<decltype(vectorInvAb)::element_type>(mesh->cell2faceLocal.dist->size());
            matrixInvAB = std::make_shared<decltype(matrixInvAB)::element_type>(mesh->cell2faceLocal.dist->size());

            // for each inner cell (ghost cell no need)
            forEachInArray(
                *mesh->cell2faceLocal.dist,
                [&](tAdjArray::tComponent &c2f, index iCell)
                {
                    auto &cellAttribute = mesh->cellAtrLocal[iCell][0];
                    auto &cellRecAttribute = cellRecAtrLocal[iCell][0];
                    auto eCell = Elem::ElementManager(cellAttribute.type, cellAttribute.intScheme);
                    assert(c2f.size() == eCell.getNFace());
                    (*matrixInvAB)[iCell].resize(c2f.size() + 1);
                    (*vectorInvAb)[iCell].resize(c2f.size(), cellRecAttribute.NDOF - 1);
                    (*vectorInvAb)[iCell].setConstant(UnInitReal);
                    (*matrixInvAB)[iCell][0].resize(cellRecAttribute.NDOF - 1, cellRecAttribute.NDOF - 1);
                    (*matrixInvAB)[iCell][0].setZero();

                    // get Aii
                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++) // for each face of cell
                    {
                        index iFace = c2f[ic2f];
                        auto f2c = mesh->face2cellLocal[iFace];
                        index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                        index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                        auto &faceAttribute = mesh->faceAtrLocal[iFace][0];
                        auto &faceRecAttribute = faceRecAtrLocal[iFace][0];
                        Elem::ElementManager eFace(faceAttribute.type, faceRecAttribute.intScheme);

                        eFace.Integration(
                            (*matrixInvAB)[iCell][0],
                            [&](Eigen::MatrixXd &incA, int ig, Elem::tPoint &ip, Elem::tDiFj &iDiNj)
                            {
                                Eigen::MatrixXd &diffsI = faceDiBjGaussCache[iFace][ig * 2 + iCellAtFace];
                                Eigen::MatrixXd incAFull;
                                FFaceFunctional(diffsI, diffsI, (*faceWeights)[iFace], incAFull);
                                assert(incAFull(Eigen::all, 0).norm() + incAFull(0, Eigen::all).norm() == 0);
                                incA = incAFull.bottomRightCorner(incAFull.rows() - 1, incAFull.cols() - 1);
                                incA *= faceNorms[iFace][ig].norm(); // note: don't forget the Jacobi!!!
                                // std::cout << "DI " << std::endl;
                                // std::cout << faceDiBjGaussCache[iFace][ig * 2 + iCellAtFace] << std::endl;
                            });
                        // if (mpi.rank == 0)
                        // std::cout << "FW " << iCell << "\n"
                        //           << (*faceWeights)[iFace] << std::endl;
                    }
                    // if (mpi.rank == 0)
                    // std::cout << "MOMENT" << baseMoments[iCell] << std::endl;
                    // std::cout << "A " << iCell << "\n"
                    //           << (*matrixInvAB)[iCell][0] << std::endl;

                    Eigen::MatrixXd Ainv;
                    HardEigen::EigenLeastSquareInverse((*matrixInvAB)[iCell][0], Ainv);

                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++) // for each face of cell
                    {
                        index iFace = c2f[ic2f];
                        auto f2c = (*mesh->face2cellPair)[iFace];
                        index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];

                        index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                        auto &faceAttribute = mesh->faceAtrLocal[iFace][0];
                        auto &faceRecAttribute = faceRecAtrLocal[iFace][0];
                        Elem::ElementManager eFace(faceAttribute.type, faceRecAttribute.intScheme);

                        if (iCellOther != FACE_2_VOL_EMPTY)
                        {
                            auto &cellAttributeOther = mesh->cellAtrLocal[iCellOther][0];
                            auto &cellRecAttributeOther = cellRecAtrLocal[iCellOther][0];
                            (*matrixInvAB)[iCell][ic2f + 1].resize(cellRecAttribute.NDOF - 1, cellRecAttributeOther.NDOF - 1);
                            (*matrixInvAB)[iCell][ic2f + 1].setZero();
                            eFace.Integration(
                                (*matrixInvAB)[iCell][ic2f + 1],
                                [&](Eigen::MatrixXd &incB, int ig, Elem::tPoint &ip, Elem::tDiFj &iDiNj)
                                {
                                    Eigen::MatrixXd &diffsI = faceDiBjGaussCache[iFace][ig * 2 + iCellAtFace];
                                    Eigen::MatrixXd &diffsJ = faceDiBjGaussCache[iFace][ig * 2 + 1 - iCellAtFace];
                                    Eigen::MatrixXd incBFull;
                                    FFaceFunctional(diffsI, diffsJ, (*faceWeights)[iFace], incBFull);
                                    assert(incBFull(Eigen::all, 0).norm() + incBFull(0, Eigen::all).norm() == 0);
                                    incB = incBFull.bottomRightCorner(incBFull.rows() - 1, incBFull.cols() - 1);
                                    incB *= faceNorms[iFace][ig].norm(); // note: don't forget the
                                    // std::cout << "DI " << std::endl;
                                    // std::cout << faceDiBjGaussCache[iFace][ig * 2 + iCellFace] << std::endl;
                                });
                            // std::cout << "Bij " << (*matrixInvAB)[iCell][ic2f + 1] << std::endl;
                            (*matrixInvAB)[iCell][ic2f + 1] = Ainv * (*matrixInvAB)[iCell][ic2f + 1];
                        }
                        else if (faceAttribute.iPhy == BoundaryType::Wall)
                        {
                            (*matrixInvAB)[iCell][ic2f + 1].resize(cellRecAttribute.NDOF - 1, cellRecAttribute.NDOF - 1);
                            (*matrixInvAB)[iCell][ic2f + 1].setZero(); // the other 'cell' has no rec
                        }
                        else if (faceAttribute.iPhy == BoundaryType::Farfield)
                        {
                            (*matrixInvAB)[iCell][ic2f + 1].resize(cellRecAttribute.NDOF - 1, cellRecAttribute.NDOF - 1);
                            (*matrixInvAB)[iCell][ic2f + 1].setZero();
                            eFace.Integration(
                                (*matrixInvAB)[iCell][ic2f + 1],
                                [&](Eigen::MatrixXd &incA, int ig, Elem::tPoint &ip, Elem::tDiFj &iDiNj)
                                {
                                    Eigen::MatrixXd &diffsI = faceDiBjGaussCache[iFace][ig * 2 + iCellAtFace];
                                    Eigen::MatrixXd incAFull;
                                    FFaceFunctional(diffsI, diffsI, (*faceWeights)[iFace], incAFull);
                                    // std::cout << "W " << faceWeights[iFace] << std::endl;
                                    assert(incAFull(Eigen::all, 0).norm() + incAFull(0, Eigen::all).norm() == 0);
                                    incA = incAFull.bottomRightCorner(incAFull.rows() - 1, incAFull.cols() - 1);
                                    incA *= faceNorms[iFace][ig].norm(); // note: don't forget the Jacobi!!!
                                });
                            (*matrixInvAB)[iCell][ic2f + 1] = Ainv * (*matrixInvAB)[iCell][ic2f + 1];
                            // std::cout << "FFAiB" << (*matrixInvAB)[iCell][ic2f + 1] << std::endl;
                            // std::cout << "W " << faceWeights[iFace] << std::endl;
                        }
                        else
                        {
                            assert(false);
                        }
                        // everyone is welcome to have this
                        {
                            Eigen::RowVectorXd row((*vectorInvAb)[iCell].cols());
                            row.setZero();
                            eFace.Integration(
                                row,
                                [&](Eigen::RowVectorXd &incb, int ig, Elem::tPoint &ip, Elem::tDiFj &iDiNj)
                                {
                                    Eigen::MatrixXd &diffsI = faceDiBjGaussCache[iFace][ig * 2 + iCellAtFace];
                                    Eigen::MatrixXd incbFull;
                                    FFaceFunctional(Eigen::MatrixXd::Ones(1, 1), diffsI({0}, Eigen::all), (*faceWeights)[iFace]({0}), incbFull);
                                    // std::cout << incbFull(0, 0) << " " << incbFull.size() << incb.size() << std::endl;
                                    assert(incbFull(0, 0) == 0);
                                    incb = incbFull.rightCols(incbFull.size() - 1);
                                    incb *= faceNorms[iFace][ig].norm(); // note: don't forget the
                                    // std::cout << "DI " << std::endl;
                                    // std::cout << faceDiBjGaussCache[iFace][ig * 2 + iCellFace] << std::endl;
                                });
                            (*vectorInvAb)[iCell].row(ic2f) = row;
                        }
                    }
                    // std::cout << "Ainv " << iCell << "\n"
                    //           << Ainv << std::endl;
                    // exit(0);
                    (*vectorInvAb)[iCell] = (*vectorInvAb)[iCell] * Ainv.transpose(); // must be outside the loop as it operates all rows at once
                    // std::cout << "bMat " << std::endl;
                    // std::cout << vectorInvAb[iCell] << std::endl;
                    (*matrixInvAB)[iCell][0] = Ainv; // converts to Ainv instead of A
                });
        }

        template <uint32_t vsize>
        // static const int vsize = 1; // intellisense helper: give example...
        void BuildRec(ArrayLocal<SemiVarMatrix<vsize>> &uR)
        {
            index nCellDist = mesh->cell2nodeLocal.dist->size();

            uR.dist = std::make_shared<typename decltype(uR.dist)::element_type>(
                typename decltype(uR.dist)::element_type::tContext(
                    [&](index i) -> rowsize
                    {
                        return vsize * (cellRecAtrLocal[i][0].NDOF - 1); // note - 1!!
                    },
                    nCellDist),
                mpi);
            uR.CreateGhostCopyComm(mesh->cellAtrLocal);
            uR.InitPersistentPullClean();
        }

        /**
         * @brief
         * \pre u need to StartPersistentPullClean()
         * \post u,uR need to WaitPersistentPullClean();
         */
        template <uint32_t vsize>
        // static const int vsize = 1; // intellisense helper: give example...
        void ReconstructionJacobiStep(ArrayLocal<VecStaticBatch<vsize>> &u,
                                      ArrayLocal<SemiVarMatrix<vsize>> &uRec,
                                      ArrayLocal<SemiVarMatrix<vsize>> &uRecNewBuf)
        {

            static int icount = 0;

            // InsertCheck(mpi, "ReconstructionJacobiStep Start");
            // forEachInArray(
            //     *uRec.dist,
            //     [&](typename decltype(uRec.dist)::element_type::tComponent &uRecE, index iCell)
            for (index iCell = 0; iCell < uRec.dist->size(); iCell++)
            {
                auto &uRecE = uRec[iCell];
                real relax = cellRecAtrLocal[iCell][0].relax;
                auto &c2f = mesh->cell2faceLocal[iCell];
                uRecNewBuf[iCell].m() = (1 - relax) * uRec[iCell].m();

                Eigen::MatrixXd coords;
                mesh->LoadCoords(mesh->cell2nodeLocal[iCell], coords);
                // std::cout << "COORDS" << coords << std::endl;
                // std::cout << (*matrixInvAB)[iCell][0] << std::endl
                //           << std::endl;
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    // this is a repeated code block START
                    index iFace = c2f[ic2f];
                    auto &f2c = (*mesh->face2cellPair)[iFace];
                    index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                    index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                    auto &faceAttribute = mesh->faceAtrLocal[iFace][0];
                    auto &faceRecAttribute = faceRecAtrLocal[iFace][0];
                    Elem::ElementManager eFace(faceAttribute.type, faceRecAttribute.intScheme);
                    Eigen::MatrixXd BCCorrection;
                    BCCorrection.resizeLike(uRec[iCell].m());
                    // this is a repeated code block END

                    if (iCellOther != FACE_2_VOL_EMPTY)
                    {
                        uRecNewBuf[iCell].m() +=
                            relax * (((*matrixInvAB)[iCell][ic2f + 1] * uRec[iCellOther].m()) +
                                     ((u[iCellOther].p() - u[iCell].p()) * (*vectorInvAb)[iCell].row(ic2f)).transpose());
                        // std::cout << "MIABInner" << (*matrixInvAB)[iCell][ic2f + 1] << std::endl;
                        // InsertCheck(mpi, "ReconstructionJacobiStep AD1");
                    }
                    else
                    {
                        if (faceAttribute.iPhy == BoundaryType::Wall)
                        {
                            Eigen::Vector<real, vsize> uBV;
                            uBV.setZero();
                            // Eigen::MatrixXd fc;
                            // mesh->LoadCoords(mesh->face2nodeLocal[iFace], fc);
                            // real ytest = (fc(1, 0) + fc(1, 1)) * 0.5;
                            // uBV.setConstant(ytest * 4);
                            // Eigen::MatrixXd uRecBV;
                            uRecNewBuf[iCell].m() +=
                                relax * (((uBV - u[iCell].p()) * (*vectorInvAb)[iCell].row(ic2f)).transpose());

                            // eFace.Integration(
                            //     BCCorrection,
                            //     [&](Eigen::MatrixXd &corInc, int ig, Elem::tPoint &p, Elem::tDiFj &iDiNj)
                            //     {
                            //         auto &diffI = faceDiBjGaussCache[iFace][ig * 2 + 0]; // must be left

                            //         Eigen::MatrixXd rowUD = (uBV - uRec[iCell].p()).transpose();
                            //         Eigen::MatrixXd rowDiffI = diffI.row(0);
                            //         FFaceFunctional(rowDiffI, rowUD, faceWeights[iFace]({0}), corInc);
                            //         corInc *= faceNorms[iFace][ig].norm();
                            //     });
                        }
                        else if (faceAttribute.iPhy == BoundaryType::Farfield)
                        {
                            uRecNewBuf[iCell].m() +=
                                relax * (((*matrixInvAB)[iCell][ic2f + 1] * uRec[iCell].m()));
                        }
                        else
                        {
                            assert(false);
                        }
                    }
                }
                // exit(0);
                if (icount == 1)
                {
                    // std::cout << "DIFF " << iCell << "\n"
                    //           << uRecNewBuf[iCell].m() << std::endl;
                    // exit(0);
                }
            }
            // );

            real vall = 0;
            real nall = 0;
            // forEachInArray(
            //     *uRec.dist,
            //     [&](typename decltype(uRec.dist)::element_type::tComponent &uRecE, index iCell)
            for (index iCell = 0; iCell < uRec.dist->size(); iCell++)
            {
                auto &uRecE = uRec[iCell];
                nall += (uRecE.m() - uRecNewBuf[iCell].m()).squaredNorm();
                vall += 1;
                // std::cout << "INC:\n";
                // std::cout << uRecE.m() << std::endl;
                uRecE.m() = uRecNewBuf[iCell].m();
                // std::cout << uRecE.m() << std::endl;
            }
            // );
            // // std::cout << "NEW\n"
            // //           << uRecNewBuf[0] << std::endl;
            real res = nall / vall;
            // std::cout << "RES " << res << std::endl;
            icount++;
        }

        void ReconstructionJacobiStep(ArrayLocal<VecStaticBatch<1>> &u,
                                      ArrayLocal<SemiVarMatrix<1>> &uRec,
                                      ArrayLocal<SemiVarMatrix<1>> &uRecNewBuf);

        void Initialization();
    };
}