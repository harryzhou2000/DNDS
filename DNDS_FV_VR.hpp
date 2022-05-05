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

        // ArrayLocal<SmallMatricesBatch> faceDiBjGaussCache;  // DiBjCache[i].m(iGauss * 2 + 0/1) = Di(Bj) at this gaussPoint, Di is diff according to diffOperatorOrderList2D[i][:]
        // ArrayLocal<SmallMatricesBatch> faceDiBjCenterCache; // center, only order 0, 1 diffs
        // ArrayLocal<VarVector> baseMoment;                   // full dofs, like baseMoment[i].v()(0) == 1

        // std::vector<std::vector<Eigen::MatrixXd>> faceDiBjGaussCache;                 // DiBjCache[i].m(iGauss * 2 + 0/1) = Di(Bj) at this gaussPoint, Di is diff according to diffOperatorOrderList2D[i][:]
        // std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> faceDiBjCenterCache; // center, only order 0, 1 diffs

        std::shared_ptr<Array<SmallMatricesBatch>> cellDiBjGaussBatch;  // DiBjCache[i].m(iGauss) = Di(Bj) at this gaussPoint, Di is diff according to diffOperatorOrderList2D[i][:]
        std::shared_ptr<Array<SmallMatricesBatch>> cellDiBjCenterBatch; // center, only order 0, 1 diffs
        std::shared_ptr<Array<SmallMatricesBatch>> faceDiBjGaussBatch;  // DiBjCache[i].m(iGauss * 2 + 0/1) = Di(Bj) at this gaussPoint, Di is diff according to diffOperatorOrderList2D[i][:]
        std::shared_ptr<Array<SmallMatricesBatch>> faceDiBjCenterBatch; // center, only order 0, 1 diffs

        std::vector<Eigen::VectorXd> baseMoments; // full dofs, like baseMoment[i].v()(0) == 1
        std::vector<Elem::tPoint> cellCenters;
        std::vector<Elem::tPoint> cellBaries;
        std::shared_ptr<std::vector<Elem::tJacobi>> cellIntertia;
        std::vector<std::vector<real>> cellGaussJacobiDets;
        std::vector<Elem::tPoint> faceCenters;
        std::vector<std::vector<Elem::tPoint>> faceNorms;
        std::vector<Elem::tPoint> faceNormCenter;
        std::shared_ptr<std::vector<Eigen::VectorXd>> faceWeights;

        std::shared_ptr<Array<SmallMatricesBatch>> vectorBatch; // invAb[i][icf] = the A^-1b of cell i's icf neighbour
        std::shared_ptr<Array<SmallMatricesBatch>> matrixBatch; // matrixInvAB[i][icf + 1] = the A^-1B of cell i's icf neighbour, invAb[i].m(0) is cell i's A^-1
                                                                // note that the dof dimensions of these rec data excludes the mean-value/const-rec dof

        struct Setting
        {
            real tangWeight = 1;
        } setting;
        // **********************************************************************************************************************
        /*







        */
        // **********************************************************************************************************************

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

        template <class TWrite>
        void FDiffBaseValue(index iCell,
                            Elem::ElementManager &cElem,
                            const Eigen::MatrixXd &coords,
                            const Elem::tDiFj &DiNj, //@ pParam
                            const Elem::tPoint &pParam,
                            const Elem::tPoint &cPhysics,
                            const Elem::tPoint &simpleScale,
                            const Eigen::VectorXd &baseMoment,
                            TWrite &&DiBj) // for 2d polynomials here
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
#ifndef USE_LOCAL_COORD
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

            // i++;
            DiBj(0, Eigen::all) -= baseMoment.transpose();
            return;
#endif
            real scaleM = simpleScale.maxCoeff();
            Eigen::Matrix2d pJacobi = (*cellIntertia)[iCell]({0, 1}, {0, 1}) * 3;
            pJacobi.col(0) = pJacobi.col(0).normalized() * scaleM;
            pJacobi.col(1) = pJacobi.col(1).normalized() * scaleM;
            Eigen::Matrix2d invPJacobi = pJacobi.inverse();
            Eigen::Vector2d pParamL = invPJacobi * pPhysicsC.topRows(2);
            // std::cout << pPhysicsCScaled << "\n"
            //           << pJacobi << "\n"
            //           << pParamL << "\n";
            // exit(0);
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
                                                       pParamL(0), pParamL(1), 0);
                }
            Elem::Convert2dDiffsLinMap(DiBj, invPJacobi.transpose());
            DiBj(0, Eigen::all) -= baseMoment.transpose();
            return;
        }

        template <class TDIFFI, class TDIFFJ>
        void FFaceFunctional(index iFace, index iGauss, TDIFFI &&DiffI, TDIFFJ &&DiffJ, const Eigen::VectorXd &Weights, Eigen::MatrixXd &Conj)
        {
            assert(Weights.size() == DiffI.rows() && DiffI.rows() == DiffJ.rows()); // has same n diffs
#ifndef USE_NORM_FUNCTIONAL
            Conj = DiffI.transpose() * Weights.asDiagonal() * Weights.asDiagonal() * DiffJ;
            return;
#endif

            if (Weights.size() == 10)
            {
                const real epsR = 1e-50;
                real w2r, w3r, length;
                if (std::fabs(Weights[1]) >= epsR || std::fabs(Weights[2]) >= epsR)
                {
                    w2r = std::fabs(Weights[1]) >= std::fabs(Weights[2]) ? Weights[3] / (Weights[1] * Weights[1]) : Weights[5] / (Weights[2] * Weights[2]);
                    w3r = std::fabs(Weights[1]) >= std::fabs(Weights[2]) ? Weights[6] / (Weights[1] * Weights[1] * Weights[1]) : Weights[9] / (Weights[2] * Weights[2] * Weights[2]);
                    length = std::sqrt(Weights[1] * Weights[1] + Weights[2] * Weights[2]);
                }
                else
                {
                    w2r = w3r = 0.0;
                    length = 0;
                }
                Eigen::Vector2d fNorm = faceNorms[iFace][iGauss]({0, 1}).stableNormalized() * length;
                Eigen::Vector2d fTang{-fNorm(1), fNorm(0)};
                fTang *= setting.tangWeight;
                real n0 = fNorm(0), n1 = fNorm(1);
                real t0 = fTang(0), t1 = fTang(1);
                // w2r *= 0.1;
                // w3r *= 0.1;

                Conj.resize(DiffI.cols(), DiffJ.cols());
                Conj.setZero();
                for (int i = 0; i < DiffI.cols(); i++)
                    for (int j = 0; j < DiffJ.cols(); j++)
                    {
                        Conj(i, j) += DiffI(0, i) * DiffJ(0, j) * Weights(0) * Weights(0);

                        real csumA, csumB;
                        csumA = DiffI(1, i) * n0 + DiffI(2, i) * n1;
                        csumB = DiffJ(1, j) * n0 + DiffJ(2, j) * n1;
                        Conj(i, j) += csumA * csumB;
                        csumA = DiffI(1, i) * t0 + DiffI(2, i) * t1;
                        csumB = DiffJ(1, j) * t0 + DiffJ(2, j) * t1;
                        Conj(i, j) += csumA * csumB;

                        csumA = (DiffI(3, i) * n0 * n0 +
                                 DiffI(4, i) * n0 * n1 * 2 +
                                 DiffI(5, i) * n1 * n1) *
                                w2r;
                        csumB = (DiffJ(3, j) * n0 * n0 +
                                 DiffJ(4, j) * n0 * n1 * 2 +
                                 DiffJ(5, j) * n1 * n1) *
                                w2r;
                        Conj(i, j) += csumA * csumB;

                        csumA = (DiffI(3, i) * n0 * t0 +
                                 DiffI(4, i) * n0 * t1 * 2 +
                                 DiffI(5, i) * n1 * t1) *
                                w2r;
                        csumB = (DiffJ(3, j) * n0 * t0 +
                                 DiffJ(4, j) * n0 * t1 * 2 +
                                 DiffJ(5, j) * n1 * t1) *
                                w2r;
                        Conj(i, j) += csumA * csumB * 2;

                        csumA = (DiffI(3, i) * t0 * t0 +
                                 DiffI(4, i) * t0 * t1 * 2 +
                                 DiffI(5, i) * t1 * t1) *
                                w2r;
                        csumB = (DiffJ(3, j) * t0 * t0 +
                                 DiffJ(4, j) * t0 * t1 * 2 +
                                 DiffJ(5, j) * t1 * t1) *
                                w2r;
                        Conj(i, j) += csumA * csumB;

                        csumA = (DiffI(6, i) * n0 * n0 * n0 +
                                 DiffI(7, i) * n0 * n0 * n1 * 3 +
                                 DiffI(8, i) * n0 * n1 * n1 * 3 +
                                 DiffI(9, i) * n1 * n1 * n1) *
                                w3r;
                        csumB = (DiffJ(6, j) * n0 * n0 * n0 +
                                 DiffJ(7, j) * n0 * n0 * n1 * 3 +
                                 DiffJ(8, j) * n0 * n1 * n1 * 3 +
                                 DiffJ(9, j) * n1 * n1 * n1) *
                                w3r;
                        Conj(i, j) += csumA * csumB;

                        csumA = (DiffI(6, i) * n0 * n0 * t0 +
                                 DiffI(7, i) * n0 * n0 * t1 * 3 +
                                 DiffI(8, i) * n0 * n1 * t1 * 3 +
                                 DiffI(9, i) * n1 * n1 * t1) *
                                w3r;
                        csumB = (DiffJ(6, j) * n0 * n0 * t0 +
                                 DiffJ(7, j) * n0 * n0 * t1 * 3 +
                                 DiffJ(8, j) * n0 * n1 * t1 * 3 +
                                 DiffJ(9, j) * n1 * n1 * t1) *
                                w3r;
                        Conj(i, j) += csumA * csumB * 3;

                        csumA = (DiffI(6, i) * n0 * t0 * t0 +
                                 DiffI(7, i) * n0 * t0 * t1 * 3 +
                                 DiffI(8, i) * n0 * t1 * t1 * 3 +
                                 DiffI(9, i) * n1 * t1 * t1) *
                                w3r;
                        csumB = (DiffJ(6, j) * n0 * t0 * t0 +
                                 DiffJ(7, j) * n0 * t0 * t1 * 3 +
                                 DiffJ(8, j) * n0 * t1 * t1 * 3 +
                                 DiffJ(9, j) * n1 * t1 * t1) *
                                w3r;
                        Conj(i, j) += csumA * csumB * 3;

                        csumA = (DiffI(6, i) * t0 * t0 * t0 +
                                 DiffI(7, i) * t0 * t0 * t1 * 3 +
                                 DiffI(8, i) * t0 * t1 * t1 * 3 +
                                 DiffI(9, i) * t1 * t1 * t1) *
                                w3r;
                        csumB = (DiffJ(6, j) * t0 * t0 * t0 +
                                 DiffJ(7, j) * t0 * t0 * t1 * 3 +
                                 DiffJ(8, j) * t0 * t1 * t1 * 3 +
                                 DiffJ(9, j) * t1 * t1 * t1) *
                                w3r;
                        Conj(i, j) += csumA * csumB;
                    }
            }
            else
            {
                Conj = DiffI.transpose() * Weights.asDiagonal() * Weights.asDiagonal() * DiffJ;
            }
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
            cellGaussJacobiDets.resize(nlocalCells);

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
                                   p, cellBaries[iCell], sScale,
                                   baseMoments[iCell],
                                   cellDiBjCenterBatchElem.m(0));

                    // iGaussPart
                    cellGaussJacobiDets[iCell].resize(eCell.getNInt());
                    for (int ig = 0; ig < eCell.getNInt(); ig++)
                    {
                        eCell.GetIntPoint(ig, p);
                        eCell.GetDiNj(p, DiNj); // N_cache not using
                        FDiffBaseValue(iCell, eCell, coords, DiNj,
                                       p, cellBaries[iCell], sScale,
                                       baseMoments[iCell],
                                       cellDiBjGaussBatchElem.m(ig));
                        cellGaussJacobiDets[iCell][ig] = Elem::DiNj2Jacobi(DiNj, coords)({0, 1}, {0, 1}).determinant();
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
                                   pCell, cellBaries[iCell], sScaleL,
                                   baseMoments[iCell], faceDiBjCenterBatchElem.m(0));

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
                                       pCell, cellBaries[iCell], sScaleR,
                                       baseMoments[iCell], faceDiBjCenterBatchElem.m(1));
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
                                       pCell, cellBaries[iCell], sScaleL,
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
                                           pCell, cellBaries[iCell], sScaleR,
                                           baseMoments[iCell], faceDiBjGaussBatchElem.m(ig * 2 + 1));
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
                        (*faceWeights)[iFace][0] = 0.0;
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
                    else if (faceAttribute.iPhy == BoundaryType::Wall || faceAttribute.iPhy == BoundaryType::Farfield)
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
                                assert(incAFull(Eigen::all, 0).norm() + incAFull(0, Eigen::all).norm() == 0);
                                incA = incAFull.bottomRightCorner(incAFull.rows() - 1, incAFull.cols() - 1);
                                incA *= faceNorms[iFace][ig].norm(); // note: don't forget the Jacobi!!!
                            });
                    }

                    Eigen::MatrixXd Ainv;
                    HardEigen::EigenLeastSquareInverse(A, Ainv);
                    matrixBatchElem.m(0) = Ainv;

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
                        else if (faceAttribute.iPhy == BoundaryType::Wall)
                        {
                            matrixBatchElem.m(ic2f + 1).setZero(); // the other 'cell' has no rec
                        }
                        else if (faceAttribute.iPhy == BoundaryType::Farfield)
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
                auto matrixBatchElem = (*matrixBatch)[iCell];
                auto vectorBatchElem = (*vectorBatch)[iCell];

                Eigen::MatrixXd coords;
                mesh->LoadCoords(mesh->cell2nodeLocal[iCell], coords);
                // std::cout << "COORDS" << coords << std::endl;
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    auto &f2c = (*mesh->face2cellPair)[iFace];
                    index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                    index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                    auto &faceAttribute = mesh->faceAtrLocal[iFace][0];
                    auto &faceRecAttribute = faceRecAtrLocal[iFace][0];
                    Elem::ElementManager eFace(faceAttribute.type, faceRecAttribute.intScheme);
                    Eigen::MatrixXd BCCorrection;
                    BCCorrection.resizeLike(uRec[iCell].m());

                    if (iCellOther != FACE_2_VOL_EMPTY)
                    {
                        uRecNewBuf[iCell].m() +=
                            relax * ((matrixBatchElem.m(ic2f + 1) * uRec[iCellOther].m()) +
                                     ((u[iCellOther].p() - u[iCell].p()) * vectorBatchElem.m(0).row(ic2f)).transpose());
                    }
                    else
                    {
                        if (faceAttribute.iPhy == BoundaryType::Wall)
                        {
                            Eigen::Vector<real, vsize> uBV;
                            uBV.setZero();
                            uRecNewBuf[iCell].m() +=
                                relax * (((uBV - u[iCell].p()) * vectorBatchElem.m(0).row(ic2f)).transpose());

                            // eFace.Integration(
                            //     BCCorrection,
                            //     [&](Eigen::MatrixXd &corInc, int ig, Elem::tPoint &p, Elem::tDiFj &iDiNj)
                            //     {
                            //         auto &diffI = faceDiBjGaussCache[iFace][ig * 2 + 0]; // must be left

                            //         Eigen::MatrixXd rowUD = (uBV - uRec[iCell].p()).transpose();
                            //         Eigen::MatrixXd rowDiffI = diffI.row(0);
                            //         FFaceFunctional(iFace,rowDiffI, rowUD, faceWeights[iFace]({0}), corInc);
                            //         corInc *= faceNorms[iFace][ig].norm();
                            //     });
                        }
                        else if (faceAttribute.iPhy == BoundaryType::Farfield)
                        {
                            uRecNewBuf[iCell].m() +=
                                relax * ((matrixBatchElem.m(ic2f + 1) * uRec[iCell].m()));
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