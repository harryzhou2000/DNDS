#pragma once

#include "DNDS_Mesh.hpp"
#include "DNDS_HardEigen.h"
#include "unsupported/Eigen/CXX11/Tensor"

#include <set>

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
                Elem::ElementManager::InitNBuffer(); //! do not assume it't been initialized
            Elem::tIntScheme schemeTri = Elem::INT_SCHEME_TRI_7;
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

        void BuildMean(ArrayDOFV &u, index nvars)
        {
            index nCellDist = mesh->cell2nodeLocal.dist->size();
            u.resize(nCellDist, mpi, nvars);
            u.CreateGhostCopyComm(mesh->cellAtrLocal);
            u.InitPersistentPullClean();
        }
    };

    class VRFiniteVolume2D
    {
    public:
        MPIInfo mpi;
        const int P_ORDER = 3;
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

        std::shared_ptr<Array<SmallMatricesBatch>> cellDiBjGaussBatch;   // DiBjCache[i].m(iGauss) = Di(Bj) at this gaussPoint, Di is diff according to diffOperatorOrderList2D[i][:]
        std::shared_ptr<Array<SmallMatricesBatch>> cellDiBjCenterBatch;  // center, only order 0, 1 diffs
        std::shared_ptr<Array<SmallMatricesBatch>> faceDiBjGaussBatch;   // DiBjCache[i].m(iGauss * 2 + 0/1) = Di(Bj) at this gaussPoint, Di is diff according to diffOperatorOrderList2D[i][:]
        std::shared_ptr<Array<SmallMatricesBatch>> faceDiBjCenterBatch;  // center, all diffs
        std::shared_ptr<Array<SmallMatricesBatch>> matrixSecondaryBatch; // for secondary rec

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
        std::shared_ptr<std::vector<Eigen::MatrixXd>> matrixAii;
        std::shared_ptr<std::vector<Eigen::MatrixXd>> matrixInnerProd;
        std::shared_ptr<std::vector<index>> SOR_iScan2iCell;
        std::shared_ptr<std::vector<index>> SOR_iCell2iScan;

        //* curvilinear doings:
        ArrayLocal<SemiVarMatrix<2>> uCurve;

        struct Setting
        {
            real wallWeight = 1.0;
            real farWeight = 0.0;

            real tangWeight = 5e-3;
            // center type
            std::string baseCenterTypeName = "Bary";
            enum BaseCenterType
            {
                Barycenter,
                Paramcenter
            } baseCenterType = Barycenter;
            // mscale calculating
            real scaleMLargerPortion = 1.;

            bool SOR_Instead = false;
            bool SOR_InverseScanning = false;
            bool SOR_RedBlack = false;
            real JacobiRelax = 1.0;

            int curvilinearOrder = 1;
            bool anistropicLengths = false;

            enum WeightSchemeGeom
            {
                None = 0,
                S = 2,
                D = 1
            } weightSchemeGeom = WeightSchemeGeom::None;
            std::string weightSchemeGeomName;

            real WBAP_SmoothIndicatorScale = 1e-10;
            real WBAP_nStd = 10.0;
            bool normWBAP = false;

            bool orthogonalizeBase = false;

        } setting;
        // **********************************************************************************************************************
        /*







        */
        // **********************************************************************************************************************

        VRFiniteVolume2D(CompactFacedMeshSerialRW *nMesh, ImplicitFiniteVolume2D *nFV, int nPOrder = 3)
            : mpi(nMesh->mpi), P_ORDER(nPOrder), mesh(nMesh), FV(nFV)
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

        Elem::tPoint getCellCenter(index iCell)
        {
            Elem::tPoint cent;
            switch (setting.baseCenterType)
            {
            case Setting::BaseCenterType::Barycenter:
                cent = cellBaries[iCell];
                break;
            case Setting::BaseCenterType::Paramcenter:
                cent = cellCenters[iCell];
                break;
            default:
                assert(false);
                break;
            }
            // std::cout << cellBaries[iCell] << " " << cellCenters[iCell] << std::endl;
            return cent;
        }

        struct FDiffBaseValueOptions
        {
            bool disableLocalCoord2GlobalDiffs = false;
            bool disableCurvilinear = false;

            FDiffBaseValueOptions() {}

            FDiffBaseValueOptions(bool NdisableLocalCoord2GlobalDiffs, bool NdisableCurvilinear)
                : disableLocalCoord2GlobalDiffs(NdisableLocalCoord2GlobalDiffs), disableCurvilinear(NdisableCurvilinear) {}
        };

        template <class TWrite>
        void FDiffBaseValue(index iCell,
                            Elem::ElementManager &cElem,
                            const Eigen::MatrixXd &coords,
                            const Elem::tDiFj &DiNj, //@ pParam
                            const Elem::tPoint &pParam,
                            const Elem::tPoint &cPhysics,
                            Elem::tPoint &simpleScale,
                            const Eigen::VectorXd &baseMoment,
                            TWrite &&DiBj,
                            FDiffBaseValueOptions options = FDiffBaseValueOptions()) // for 2d polynomials here
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
            auto coordsC = coords.colwise() - cPhysics;
            // ** minmax scaling
            auto P2CDist = coordsC.colwise().norm();
            Eigen::Index maxDistIndex, minDistIndex;
            real maxDist = P2CDist.maxCoeff(&maxDistIndex);
            real minDist = P2CDist.minCoeff(&minDistIndex);
            // ** minmax scaling
            real scaleMLarge = simpleScale({0, 1}).maxCoeff();
            real scaleMSmall = simpleScale({0, 1}).minCoeff();
            scaleMLarge = maxDist;
            scaleMSmall = minDist;
            real scaleM = std::pow(scaleMLarge, setting.scaleMLargerPortion) *
                          std::pow(scaleMSmall, 1 - setting.scaleMLargerPortion);

            Eigen::Matrix2d pJacobi = (*cellIntertia)[iCell]({0, 1}, {0, 1}) * 1;
            real scaleL0 = pJacobi.col(0).norm();
            real scaleL1 = pJacobi.col(1).norm();
            Eigen::Matrix2d invPJacobi;

            pJacobi.col(0) = pJacobi.col(0).normalized();
            pJacobi.col(1) = pJacobi.col(1).normalized();

            invPJacobi = pJacobi.inverse();

            auto ncoords = (invPJacobi * coordsC.topRows(2));
            auto nSizes = ncoords.rowwise().maxCoeff() - ncoords.rowwise().minCoeff();

            scaleL0 = nSizes(0);
            scaleL1 = nSizes(1);
            // std::cout << scaleM << "\t" << scaleL0 << "\t" << scaleL1 << "\t" << std::endl;
            // abort();
            real scaleUni = std::pow(scaleL0, 1) * std::pow(scaleL1, 0);
            if (!setting.anistropicLengths)
            {
                pJacobi.col(0) = pJacobi.col(0) * scaleUni;
                pJacobi.col(1) = pJacobi.col(1) * scaleUni;
                simpleScale(0) = simpleScale(1) = scaleUni;
            }
            else
            {
                pJacobi.col(0) = pJacobi.col(0) * scaleL0;
                pJacobi.col(1) = pJacobi.col(1) * scaleL1;
                simpleScale(0) = simpleScale(1) = scaleUni;
            }

            invPJacobi = pJacobi.inverse();

            Eigen::Vector2d pParamL = invPJacobi * pPhysicsC.topRows(2);
// std::cout << pPhysicsCScaled << "\n"
//           << pJacobi << "\n"
//           << pParamL << "\n";
// exit(0);
#ifndef USE_LOCAL_COORD_CURVILINEAR
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
                    // todo: upgrade FPoly to considering using zeta-based curvilinear [matlab exposure? or recursion?] using zeta_i(xi_i), D_i_zeta_i(xi_i)
                    // todo: which can be calculated [contraction of tensors] level 0 of the function, given zeta's coeff a_zeta
                    // todo: a_zeta is a approximation of solution in cell, with some approximation functional
                    // todo: D_i means d/dxi_1, d/dxi_2, d2/dxi_1dxi_1 ...
                }
#else
            if (options.disableCurvilinear)
            {
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
            }
            else
            {

                Eigen::TensorFixedSize<real, Eigen::Sizes<3>> curveA1;
                Eigen::TensorFixedSize<real, Eigen::Sizes<3, 3>> curveA2;
                Eigen::TensorFixedSize<real, Eigen::Sizes<3, 3, 3>> curveA3;
                curveA1.setZero(), curveA2.setZero(), curveA3.setZero();
                auto curveCoeff = uCurve[iCell].m();
                int curveNBase = curveCoeff.rows();
                switch (curveNBase)
                {
                case 9:
                    curveA3(0, 0, 0) = curveCoeff(6 - 1, 0);
                    curveA3(0, 0, 1) = curveA3(1, 0, 0) = curveA3(0, 1, 0) = curveCoeff(7 - 1, 0) / 3.0;
                    curveA3(0, 1, 1) = curveA3(1, 1, 0) = curveA3(1, 0, 1) = curveCoeff(8 - 1, 0) / 3.0;
                    curveA3(1, 1, 1) = curveCoeff(9 - 1, 0);
                case 5:
                    curveA2(0, 0) = curveCoeff(3 - 1, 0);
                    curveA2(0, 1) = curveA2(1, 0) = curveCoeff(4 - 1, 0) / 2.0;
                    curveA2(1, 1) = curveCoeff(5 - 1, 0);
                case 2:
                    curveA1(0) = curveCoeff(1 - 1, 0);
                    curveA1(1) = curveCoeff(2 - 1, 0);
                    break;
                default:
                    assert(false);
                    break;
                }
                Eigen::TensorFixedSize<real, Eigen::Sizes<3>> xi;
                xi.setValues({pParamL(0), pParamL(1), 0});
                // Eigen::TensorFixedSize<real, Eigen::Sizes<3>> xiN1;
                // xiN1.setValues({0, 0, pParamL.norm()});
                Eigen::TensorFixedSize<real, Eigen::Sizes<3>> xiM1;
                xiM1.setValues({-pParamL(1), pParamL(0), 0});

                // xiM1 = -1;
                //  xiN1 *= -1;

                Eigen::TensorFixedSize<real, Eigen::Sizes<3, 3>> rotXi2XiN;
                rotXi2XiN.setZero();
                rotXi2XiN(2, 2) = 1;
                rotXi2XiN(0, 1) = -1;
                rotXi2XiN(1, 0) = 1;

                auto EvalTensorPoly =
                    [&curveA1, &curveA2, &curveA3, &xi](
                        Eigen::TensorFixedSize<real, Eigen::Sizes<>> &D0,
                        Eigen::TensorFixedSize<real, Eigen::Sizes<3>> &D1,
                        Eigen::TensorFixedSize<real, Eigen::Sizes<3, 3>> &D2,
                        Eigen::TensorFixedSize<real, Eigen::Sizes<3, 3, 3>> &D3) -> void
                {
                    D0 =
                        curveA1
                            .contract(xi, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 0)}) +
                        curveA2
                            .contract(xi, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 0)})
                            .contract(xi, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 0)}) +
                        curveA3
                            .contract(xi, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 0)})
                            .contract(xi, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 0)})
                            .contract(xi, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 0)});
                    D1 =
                        curveA1 +
                        (curveA2 + curveA2.shuffle(std::vector<int>{1, 0}))
                            .contract(xi, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 0)}) +
                        (curveA3 + curveA3.shuffle(std::vector<int>{1, 2, 0}) + curveA3.shuffle(std::vector<int>{2, 0, 1}))
                            .contract(xi, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 0)})
                            .contract(xi, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 0)});
                    D2 =
                        (curveA2 + curveA2.shuffle(std::vector<int>{1, 0})) +
                        (curveA3 + curveA3.shuffle(std::vector<int>{0, 2, 1}) + curveA3.shuffle(std::vector<int>{1, 2, 0}) + curveA3.shuffle(std::vector<int>{1, 0, 2}) + curveA3.shuffle(std::vector<int>{2, 0, 1}) + curveA3.shuffle(std::vector<int>{2, 1, 0}))
                            .contract(xi, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 0)});
                    D3 =
                        (curveA3 + curveA3.shuffle(std::vector<int>{0, 2, 1}) + curveA3.shuffle(std::vector<int>{1, 2, 0}) + curveA3.shuffle(std::vector<int>{1, 0, 2}) + curveA3.shuffle(std::vector<int>{2, 0, 1}) + curveA3.shuffle(std::vector<int>{2, 1, 0}));
                };
                Eigen::TensorFixedSize<real, Eigen::Sizes<>> D0zeta0;
                Eigen::TensorFixedSize<real, Eigen::Sizes<3>> D1zeta0;
                Eigen::TensorFixedSize<real, Eigen::Sizes<3, 3>> D2zeta0;
                Eigen::TensorFixedSize<real, Eigen::Sizes<3, 3, 3>> D3zeta0;
                Eigen::TensorFixedSize<real, Eigen::Sizes<>> D0zeta1;
                Eigen::TensorFixedSize<real, Eigen::Sizes<3>> D1zeta1;
                Eigen::TensorFixedSize<real, Eigen::Sizes<3, 3>> D2zeta1;
                Eigen::TensorFixedSize<real, Eigen::Sizes<3, 3, 3>> D3zeta1;
                Eigen::TensorFixedSize<real, Eigen::Sizes<>> D0zeta2;
                Eigen::TensorFixedSize<real, Eigen::Sizes<3>> D1zeta2;
                Eigen::TensorFixedSize<real, Eigen::Sizes<3, 3>> D2zeta2;
                Eigen::TensorFixedSize<real, Eigen::Sizes<3, 3, 3>> D3zeta2;

                EvalTensorPoly(D0zeta0, D1zeta0, D2zeta0, D3zeta0);
                // std::cout << "CurveA1 " << curveA1 << std::endl;
                // std::cout << "res " << curveA1.contract(rotXi2XiN, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 1)}) << std::endl;
                // Eigen::TensorFixedSize<real, Eigen::Sizes<3>> curveA1E = curveA1.contract(rotXi2XiN, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 1)});
                // Eigen::TensorFixedSize<real, Eigen::Sizes<3, 3>> curveA2E = curveA2.contract(rotXi2XiN, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 1)});
                // Eigen::TensorFixedSize<real, Eigen::Sizes<3, 3, 3>> curveA3E = curveA3.contract(rotXi2XiN, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(2, 1)}); // converts to second zeta
                // curveA1 = curveA1E;
                // curveA2 = curveA2E;
                // curveA3 = curveA3E;
                switch (curveNBase)
                {
                case 9:
                    curveA3(0, 0, 0) = curveCoeff(6 - 1, 1);
                    curveA3(0, 0, 1) = curveA3(1, 0, 0) = curveA3(0, 1, 0) = curveCoeff(7 - 1, 1) / 3.0;
                    curveA3(0, 1, 1) = curveA3(1, 1, 0) = curveA3(1, 0, 1) = curveCoeff(8 - 1, 1) / 3.0;
                    curveA3(1, 1, 1) = curveCoeff(9 - 1, 1);
                case 5:
                    curveA2(0, 0) = curveCoeff(3 - 1, 1);
                    curveA2(0, 1) = curveA2(1, 0) = curveCoeff(4 - 1, 1) / 2.0;
                    curveA2(1, 1) = curveCoeff(5 - 1, 1);
                case 2:
                    curveA1(0) = curveCoeff(1 - 1, 1);
                    curveA1(1) = curveCoeff(2 - 1, 1);
                    break;
                default:
                    assert(false);
                    break;
                }

                // std::cout << "CurveA1 " << curveA1 << std::endl;

                EvalTensorPoly(D0zeta1, D1zeta1, D2zeta1, D3zeta1);
                // std::cout << "D0zeta1 " << D0zeta1 << std::endl;
                D0zeta2.setZero(), D1zeta2.setZero(), D2zeta2.setZero(), D3zeta2.setZero();
                D1zeta2(2) = 1;

                Eigen::MatrixXd DZeta(20, 3);
                DZeta.setZero();
                auto ExpandTensor2Col =
                    [](int icol, Eigen::MatrixXd &Diffs,
                       Eigen::TensorFixedSize<real, Eigen::Sizes<>> &D0,
                       Eigen::TensorFixedSize<real, Eigen::Sizes<3>> &D1,
                       Eigen::TensorFixedSize<real, Eigen::Sizes<3, 3>> &D2,
                       Eigen::TensorFixedSize<real, Eigen::Sizes<3, 3, 3>> &D3) -> void
                {
                    Diffs(0, icol) = D0(0);
                    Diffs(1, icol) = D1(0);
                    Diffs(2, icol) = D1(1);
                    Diffs(3, icol) = D1(2);
                    Diffs(4, icol) = D2(0, 0);
                    Diffs(5, icol) = D2(0, 1);
                    Diffs(6, icol) = D2(0, 2);
                    Diffs(7, icol) = D2(1, 1);
                    Diffs(8, icol) = D2(1, 2);
                    Diffs(9, icol) = D2(2, 2);
                    Diffs(10, icol) = D3(0, 0, 0);
                    Diffs(11, icol) = D3(0, 0, 1);
                    Diffs(12, icol) = D3(0, 0, 2);
                    Diffs(13, icol) = D3(0, 1, 1);
                    Diffs(14, icol) = D3(0, 1, 2);
                    Diffs(15, icol) = D3(0, 2, 2);
                    Diffs(16, icol) = D3(1, 1, 1);
                    Diffs(17, icol) = D3(1, 1, 2);
                    Diffs(18, icol) = D3(1, 2, 2);
                    Diffs(19, icol) = D3(2, 2, 2);
                };

                ExpandTensor2Col(0, DZeta, D0zeta0, D1zeta0, D2zeta0, D3zeta0);
                ExpandTensor2Col(1, DZeta, D0zeta1, D1zeta1, D2zeta1, D3zeta1);
                ExpandTensor2Col(2, DZeta, D0zeta2, D1zeta2, D2zeta2, D3zeta2);
                // assert(DZeta({4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}, Eigen::all).norm() == 0);

                Eigen::MatrixXd DiBj3(20, 20);
                DiBj3.setZero();
                DiBj3(0, 0) = 1.0;
                DiBj3(0, 1) = DZeta(0, 0);
                DiBj3(0, 2) = DZeta(0, 1);
                DiBj3(0, 3) = DZeta(0, 2);
                DiBj3(0, 4) = DZeta(0, 0) * DZeta(0, 0);
                DiBj3(0, 5) = DZeta(0, 0) * DZeta(0, 1);
                DiBj3(0, 6) = DZeta(0, 0) * DZeta(0, 2);
                DiBj3(0, 7) = DZeta(0, 1) * DZeta(0, 1);
                DiBj3(0, 8) = DZeta(0, 1) * DZeta(0, 2);
                DiBj3(0, 9) = DZeta(0, 2) * DZeta(0, 2);
                DiBj3(0, 10) = DZeta(0, 0) * DZeta(0, 0) * DZeta(0, 0);
                DiBj3(0, 11) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(0, 1);
                DiBj3(0, 12) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(0, 2);
                DiBj3(0, 13) = DZeta(0, 0) * (DZeta(0, 1) * DZeta(0, 1));
                DiBj3(0, 14) = DZeta(0, 0) * DZeta(0, 1) * DZeta(0, 2);
                DiBj3(0, 15) = DZeta(0, 0) * (DZeta(0, 2) * DZeta(0, 2));
                DiBj3(0, 16) = DZeta(0, 1) * DZeta(0, 1) * DZeta(0, 1);
                DiBj3(0, 17) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(0, 2);
                DiBj3(0, 18) = DZeta(0, 1) * (DZeta(0, 2) * DZeta(0, 2));
                DiBj3(0, 19) = DZeta(0, 2) * DZeta(0, 2) * DZeta(0, 2);
                DiBj3(1, 1) = DZeta(1, 0);
                DiBj3(1, 2) = DZeta(1, 1);
                DiBj3(1, 3) = DZeta(1, 2);
                DiBj3(1, 4) = DZeta(0, 0) * DZeta(1, 0) * 2.0;
                DiBj3(1, 5) = DZeta(0, 0) * DZeta(1, 1) + DZeta(0, 1) * DZeta(1, 0);
                DiBj3(1, 6) = DZeta(0, 0) * DZeta(1, 2) + DZeta(0, 2) * DZeta(1, 0);
                DiBj3(1, 7) = DZeta(0, 1) * DZeta(1, 1) * 2.0;
                DiBj3(1, 8) = DZeta(0, 1) * DZeta(1, 2) + DZeta(0, 2) * DZeta(1, 1);
                DiBj3(1, 9) = DZeta(0, 2) * DZeta(1, 2) * 2.0;
                DiBj3(1, 10) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(1, 0) * 3.0;
                DiBj3(1, 11) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(1, 1) + DZeta(0, 0) * DZeta(0, 1) * DZeta(1, 0) * 2.0;
                DiBj3(1, 12) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(1, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(1, 0) * 2.0;
                DiBj3(1, 13) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(1, 0) + DZeta(0, 0) * DZeta(0, 1) * DZeta(1, 1) * 2.0;
                DiBj3(1, 14) = DZeta(0, 0) * DZeta(0, 1) * DZeta(1, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(1, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(1, 0);
                DiBj3(1, 15) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(1, 0) + DZeta(0, 0) * DZeta(0, 2) * DZeta(1, 2) * 2.0;
                DiBj3(1, 16) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(1, 1) * 3.0;
                DiBj3(1, 17) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(1, 2) + DZeta(0, 1) * DZeta(0, 2) * DZeta(1, 1) * 2.0;
                DiBj3(1, 18) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(1, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(1, 2) * 2.0;
                DiBj3(1, 19) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(1, 2) * 3.0;
                DiBj3(2, 1) = DZeta(2, 0);
                DiBj3(2, 2) = DZeta(2, 1);
                DiBj3(2, 3) = DZeta(2, 2);
                DiBj3(2, 4) = DZeta(0, 0) * DZeta(2, 0) * 2.0;
                DiBj3(2, 5) = DZeta(0, 0) * DZeta(2, 1) + DZeta(0, 1) * DZeta(2, 0);
                DiBj3(2, 6) = DZeta(0, 0) * DZeta(2, 2) + DZeta(0, 2) * DZeta(2, 0);
                DiBj3(2, 7) = DZeta(0, 1) * DZeta(2, 1) * 2.0;
                DiBj3(2, 8) = DZeta(0, 1) * DZeta(2, 2) + DZeta(0, 2) * DZeta(2, 1);
                DiBj3(2, 9) = DZeta(0, 2) * DZeta(2, 2) * 2.0;
                DiBj3(2, 10) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(2, 0) * 3.0;
                DiBj3(2, 11) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(2, 1) + DZeta(0, 0) * DZeta(0, 1) * DZeta(2, 0) * 2.0;
                DiBj3(2, 12) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(2, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(2, 0) * 2.0;
                DiBj3(2, 13) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(2, 0) + DZeta(0, 0) * DZeta(0, 1) * DZeta(2, 1) * 2.0;
                DiBj3(2, 14) = DZeta(0, 0) * DZeta(0, 1) * DZeta(2, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(2, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(2, 0);
                DiBj3(2, 15) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(2, 0) + DZeta(0, 0) * DZeta(0, 2) * DZeta(2, 2) * 2.0;
                DiBj3(2, 16) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(2, 1) * 3.0;
                DiBj3(2, 17) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(2, 2) + DZeta(0, 1) * DZeta(0, 2) * DZeta(2, 1) * 2.0;
                DiBj3(2, 18) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(2, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(2, 2) * 2.0;
                DiBj3(2, 19) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(2, 2) * 3.0;
                DiBj3(3, 1) = DZeta(3, 0);
                DiBj3(3, 2) = DZeta(3, 1);
                DiBj3(3, 3) = DZeta(3, 2);
                DiBj3(3, 4) = DZeta(0, 0) * DZeta(3, 0) * 2.0;
                DiBj3(3, 5) = DZeta(0, 0) * DZeta(3, 1) + DZeta(0, 1) * DZeta(3, 0);
                DiBj3(3, 6) = DZeta(0, 0) * DZeta(3, 2) + DZeta(0, 2) * DZeta(3, 0);
                DiBj3(3, 7) = DZeta(0, 1) * DZeta(3, 1) * 2.0;
                DiBj3(3, 8) = DZeta(0, 1) * DZeta(3, 2) + DZeta(0, 2) * DZeta(3, 1);
                DiBj3(3, 9) = DZeta(0, 2) * DZeta(3, 2) * 2.0;
                DiBj3(3, 10) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(3, 0) * 3.0;
                DiBj3(3, 11) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(3, 1) + DZeta(0, 0) * DZeta(0, 1) * DZeta(3, 0) * 2.0;
                DiBj3(3, 12) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(3, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(3, 0) * 2.0;
                DiBj3(3, 13) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(3, 0) + DZeta(0, 0) * DZeta(0, 1) * DZeta(3, 1) * 2.0;
                DiBj3(3, 14) = DZeta(0, 0) * DZeta(0, 1) * DZeta(3, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(3, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(3, 0);
                DiBj3(3, 15) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(3, 0) + DZeta(0, 0) * DZeta(0, 2) * DZeta(3, 2) * 2.0;
                DiBj3(3, 16) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(3, 1) * 3.0;
                DiBj3(3, 17) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(3, 2) + DZeta(0, 1) * DZeta(0, 2) * DZeta(3, 1) * 2.0;
                DiBj3(3, 18) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(3, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(3, 2) * 2.0;
                DiBj3(3, 19) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(3, 2) * 3.0;
                DiBj3(4, 1) = DZeta(4, 0);
                DiBj3(4, 2) = DZeta(4, 1);
                DiBj3(4, 3) = DZeta(4, 2);
                DiBj3(4, 4) = (DZeta(1, 0) * DZeta(1, 0)) * 2.0 + DZeta(0, 0) * DZeta(4, 0) * 2.0;
                DiBj3(4, 5) = DZeta(1, 0) * DZeta(1, 1) * 2.0 + DZeta(0, 0) * DZeta(4, 1) + DZeta(0, 1) * DZeta(4, 0);
                DiBj3(4, 6) = DZeta(1, 0) * DZeta(1, 2) * 2.0 + DZeta(0, 0) * DZeta(4, 2) + DZeta(0, 2) * DZeta(4, 0);
                DiBj3(4, 7) = (DZeta(1, 1) * DZeta(1, 1)) * 2.0 + DZeta(0, 1) * DZeta(4, 1) * 2.0;
                DiBj3(4, 8) = DZeta(1, 1) * DZeta(1, 2) * 2.0 + DZeta(0, 1) * DZeta(4, 2) + DZeta(0, 2) * DZeta(4, 1);
                DiBj3(4, 9) = (DZeta(1, 2) * DZeta(1, 2)) * 2.0 + DZeta(0, 2) * DZeta(4, 2) * 2.0;
                DiBj3(4, 10) = DZeta(0, 0) * (DZeta(1, 0) * DZeta(1, 0)) * 6.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(4, 0) * 3.0;
                DiBj3(4, 11) = DZeta(0, 1) * (DZeta(1, 0) * DZeta(1, 0)) * 2.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(4, 1) + DZeta(0, 0) * DZeta(1, 0) * DZeta(1, 1) * 4.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(4, 0) * 2.0;
                DiBj3(4, 12) = DZeta(0, 2) * (DZeta(1, 0) * DZeta(1, 0)) * 2.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(4, 2) + DZeta(0, 0) * DZeta(1, 0) * DZeta(1, 2) * 4.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(4, 0) * 2.0;
                DiBj3(4, 13) = DZeta(0, 0) * (DZeta(1, 1) * DZeta(1, 1)) * 2.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(4, 0) + DZeta(0, 1) * DZeta(1, 0) * DZeta(1, 1) * 4.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(4, 1) * 2.0;
                DiBj3(4, 14) = DZeta(0, 0) * DZeta(1, 1) * DZeta(1, 2) * 2.0 + DZeta(0, 1) * DZeta(1, 0) * DZeta(1, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 0) * DZeta(1, 1) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(4, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(4, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(4, 0);
                DiBj3(4, 15) = DZeta(0, 0) * (DZeta(1, 2) * DZeta(1, 2)) * 2.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(4, 0) + DZeta(0, 2) * DZeta(1, 0) * DZeta(1, 2) * 4.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(4, 2) * 2.0;
                DiBj3(4, 16) = DZeta(0, 1) * (DZeta(1, 1) * DZeta(1, 1)) * 6.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(4, 1) * 3.0;
                DiBj3(4, 17) = DZeta(0, 2) * (DZeta(1, 1) * DZeta(1, 1)) * 2.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(4, 2) + DZeta(0, 1) * DZeta(1, 1) * DZeta(1, 2) * 4.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(4, 1) * 2.0;
                DiBj3(4, 18) = DZeta(0, 1) * (DZeta(1, 2) * DZeta(1, 2)) * 2.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(4, 1) + DZeta(0, 2) * DZeta(1, 1) * DZeta(1, 2) * 4.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(4, 2) * 2.0;
                DiBj3(4, 19) = DZeta(0, 2) * (DZeta(1, 2) * DZeta(1, 2)) * 6.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(4, 2) * 3.0;
                DiBj3(5, 1) = DZeta(5, 0);
                DiBj3(5, 2) = DZeta(5, 1);
                DiBj3(5, 3) = DZeta(5, 2);
                DiBj3(5, 4) = DZeta(1, 0) * DZeta(2, 0) * 2.0 + DZeta(0, 0) * DZeta(5, 0) * 2.0;
                DiBj3(5, 5) = DZeta(1, 0) * DZeta(2, 1) + DZeta(1, 1) * DZeta(2, 0) + DZeta(0, 0) * DZeta(5, 1) + DZeta(0, 1) * DZeta(5, 0);
                DiBj3(5, 6) = DZeta(1, 0) * DZeta(2, 2) + DZeta(1, 2) * DZeta(2, 0) + DZeta(0, 0) * DZeta(5, 2) + DZeta(0, 2) * DZeta(5, 0);
                DiBj3(5, 7) = DZeta(1, 1) * DZeta(2, 1) * 2.0 + DZeta(0, 1) * DZeta(5, 1) * 2.0;
                DiBj3(5, 8) = DZeta(1, 1) * DZeta(2, 2) + DZeta(1, 2) * DZeta(2, 1) + DZeta(0, 1) * DZeta(5, 2) + DZeta(0, 2) * DZeta(5, 1);
                DiBj3(5, 9) = DZeta(1, 2) * DZeta(2, 2) * 2.0 + DZeta(0, 2) * DZeta(5, 2) * 2.0;
                DiBj3(5, 10) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(5, 0) * 3.0 + DZeta(0, 0) * DZeta(1, 0) * DZeta(2, 0) * 6.0;
                DiBj3(5, 11) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(5, 1) + DZeta(0, 0) * DZeta(1, 0) * DZeta(2, 1) * 2.0 + DZeta(0, 0) * DZeta(1, 1) * DZeta(2, 0) * 2.0 + DZeta(0, 1) * DZeta(1, 0) * DZeta(2, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(5, 0) * 2.0;
                DiBj3(5, 12) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(5, 2) + DZeta(0, 0) * DZeta(1, 0) * DZeta(2, 2) * 2.0 + DZeta(0, 0) * DZeta(1, 2) * DZeta(2, 0) * 2.0 + DZeta(0, 2) * DZeta(1, 0) * DZeta(2, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(5, 0) * 2.0;
                DiBj3(5, 13) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(5, 0) + DZeta(0, 0) * DZeta(1, 1) * DZeta(2, 1) * 2.0 + DZeta(0, 1) * DZeta(1, 0) * DZeta(2, 1) * 2.0 + DZeta(0, 1) * DZeta(1, 1) * DZeta(2, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(5, 1) * 2.0;
                DiBj3(5, 14) = DZeta(0, 0) * DZeta(1, 1) * DZeta(2, 2) + DZeta(0, 0) * DZeta(1, 2) * DZeta(2, 1) + DZeta(0, 1) * DZeta(1, 0) * DZeta(2, 2) + DZeta(0, 1) * DZeta(1, 2) * DZeta(2, 0) + DZeta(0, 2) * DZeta(1, 0) * DZeta(2, 1) + DZeta(0, 2) * DZeta(1, 1) * DZeta(2, 0) + DZeta(0, 0) * DZeta(0, 1) * DZeta(5, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(5, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(5, 0);
                DiBj3(5, 15) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(5, 0) + DZeta(0, 0) * DZeta(1, 2) * DZeta(2, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 0) * DZeta(2, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(2, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(5, 2) * 2.0;
                DiBj3(5, 16) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(5, 1) * 3.0 + DZeta(0, 1) * DZeta(1, 1) * DZeta(2, 1) * 6.0;
                DiBj3(5, 17) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(5, 2) + DZeta(0, 1) * DZeta(1, 1) * DZeta(2, 2) * 2.0 + DZeta(0, 1) * DZeta(1, 2) * DZeta(2, 1) * 2.0 + DZeta(0, 2) * DZeta(1, 1) * DZeta(2, 1) * 2.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(5, 1) * 2.0;
                DiBj3(5, 18) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(5, 1) + DZeta(0, 1) * DZeta(1, 2) * DZeta(2, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 1) * DZeta(2, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(2, 1) * 2.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(5, 2) * 2.0;
                DiBj3(5, 19) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(5, 2) * 3.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(2, 2) * 6.0;
                DiBj3(6, 1) = DZeta(6, 0);
                DiBj3(6, 2) = DZeta(6, 1);
                DiBj3(6, 3) = DZeta(6, 2);
                DiBj3(6, 4) = DZeta(1, 0) * DZeta(3, 0) * 2.0 + DZeta(0, 0) * DZeta(6, 0) * 2.0;
                DiBj3(6, 5) = DZeta(1, 0) * DZeta(3, 1) + DZeta(1, 1) * DZeta(3, 0) + DZeta(0, 0) * DZeta(6, 1) + DZeta(0, 1) * DZeta(6, 0);
                DiBj3(6, 6) = DZeta(1, 0) * DZeta(3, 2) + DZeta(1, 2) * DZeta(3, 0) + DZeta(0, 0) * DZeta(6, 2) + DZeta(0, 2) * DZeta(6, 0);
                DiBj3(6, 7) = DZeta(1, 1) * DZeta(3, 1) * 2.0 + DZeta(0, 1) * DZeta(6, 1) * 2.0;
                DiBj3(6, 8) = DZeta(1, 1) * DZeta(3, 2) + DZeta(1, 2) * DZeta(3, 1) + DZeta(0, 1) * DZeta(6, 2) + DZeta(0, 2) * DZeta(6, 1);
                DiBj3(6, 9) = DZeta(1, 2) * DZeta(3, 2) * 2.0 + DZeta(0, 2) * DZeta(6, 2) * 2.0;
                DiBj3(6, 10) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(6, 0) * 3.0 + DZeta(0, 0) * DZeta(1, 0) * DZeta(3, 0) * 6.0;
                DiBj3(6, 11) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(6, 1) + DZeta(0, 0) * DZeta(1, 0) * DZeta(3, 1) * 2.0 + DZeta(0, 0) * DZeta(1, 1) * DZeta(3, 0) * 2.0 + DZeta(0, 1) * DZeta(1, 0) * DZeta(3, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(6, 0) * 2.0;
                DiBj3(6, 12) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(6, 2) + DZeta(0, 0) * DZeta(1, 0) * DZeta(3, 2) * 2.0 + DZeta(0, 0) * DZeta(1, 2) * DZeta(3, 0) * 2.0 + DZeta(0, 2) * DZeta(1, 0) * DZeta(3, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(6, 0) * 2.0;
                DiBj3(6, 13) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(6, 0) + DZeta(0, 0) * DZeta(1, 1) * DZeta(3, 1) * 2.0 + DZeta(0, 1) * DZeta(1, 0) * DZeta(3, 1) * 2.0 + DZeta(0, 1) * DZeta(1, 1) * DZeta(3, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(6, 1) * 2.0;
                DiBj3(6, 14) = DZeta(0, 0) * DZeta(1, 1) * DZeta(3, 2) + DZeta(0, 0) * DZeta(1, 2) * DZeta(3, 1) + DZeta(0, 1) * DZeta(1, 0) * DZeta(3, 2) + DZeta(0, 1) * DZeta(1, 2) * DZeta(3, 0) + DZeta(0, 2) * DZeta(1, 0) * DZeta(3, 1) + DZeta(0, 2) * DZeta(1, 1) * DZeta(3, 0) + DZeta(0, 0) * DZeta(0, 1) * DZeta(6, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(6, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(6, 0);
                DiBj3(6, 15) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(6, 0) + DZeta(0, 0) * DZeta(1, 2) * DZeta(3, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 0) * DZeta(3, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(3, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(6, 2) * 2.0;
                DiBj3(6, 16) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(6, 1) * 3.0 + DZeta(0, 1) * DZeta(1, 1) * DZeta(3, 1) * 6.0;
                DiBj3(6, 17) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(6, 2) + DZeta(0, 1) * DZeta(1, 1) * DZeta(3, 2) * 2.0 + DZeta(0, 1) * DZeta(1, 2) * DZeta(3, 1) * 2.0 + DZeta(0, 2) * DZeta(1, 1) * DZeta(3, 1) * 2.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(6, 1) * 2.0;
                DiBj3(6, 18) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(6, 1) + DZeta(0, 1) * DZeta(1, 2) * DZeta(3, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 1) * DZeta(3, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(3, 1) * 2.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(6, 2) * 2.0;
                DiBj3(6, 19) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(6, 2) * 3.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(3, 2) * 6.0;
                DiBj3(7, 1) = DZeta(7, 0);
                DiBj3(7, 2) = DZeta(7, 1);
                DiBj3(7, 3) = DZeta(7, 2);
                DiBj3(7, 4) = (DZeta(2, 0) * DZeta(2, 0)) * 2.0 + DZeta(0, 0) * DZeta(7, 0) * 2.0;
                DiBj3(7, 5) = DZeta(2, 0) * DZeta(2, 1) * 2.0 + DZeta(0, 0) * DZeta(7, 1) + DZeta(0, 1) * DZeta(7, 0);
                DiBj3(7, 6) = DZeta(2, 0) * DZeta(2, 2) * 2.0 + DZeta(0, 0) * DZeta(7, 2) + DZeta(0, 2) * DZeta(7, 0);
                DiBj3(7, 7) = (DZeta(2, 1) * DZeta(2, 1)) * 2.0 + DZeta(0, 1) * DZeta(7, 1) * 2.0;
                DiBj3(7, 8) = DZeta(2, 1) * DZeta(2, 2) * 2.0 + DZeta(0, 1) * DZeta(7, 2) + DZeta(0, 2) * DZeta(7, 1);
                DiBj3(7, 9) = (DZeta(2, 2) * DZeta(2, 2)) * 2.0 + DZeta(0, 2) * DZeta(7, 2) * 2.0;
                DiBj3(7, 10) = DZeta(0, 0) * (DZeta(2, 0) * DZeta(2, 0)) * 6.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(7, 0) * 3.0;
                DiBj3(7, 11) = DZeta(0, 1) * (DZeta(2, 0) * DZeta(2, 0)) * 2.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(7, 1) + DZeta(0, 0) * DZeta(2, 0) * DZeta(2, 1) * 4.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(7, 0) * 2.0;
                DiBj3(7, 12) = DZeta(0, 2) * (DZeta(2, 0) * DZeta(2, 0)) * 2.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(7, 2) + DZeta(0, 0) * DZeta(2, 0) * DZeta(2, 2) * 4.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(7, 0) * 2.0;
                DiBj3(7, 13) = DZeta(0, 0) * (DZeta(2, 1) * DZeta(2, 1)) * 2.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(7, 0) + DZeta(0, 1) * DZeta(2, 0) * DZeta(2, 1) * 4.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(7, 1) * 2.0;
                DiBj3(7, 14) = DZeta(0, 0) * DZeta(2, 1) * DZeta(2, 2) * 2.0 + DZeta(0, 1) * DZeta(2, 0) * DZeta(2, 2) * 2.0 + DZeta(0, 2) * DZeta(2, 0) * DZeta(2, 1) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(7, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(7, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(7, 0);
                DiBj3(7, 15) = DZeta(0, 0) * (DZeta(2, 2) * DZeta(2, 2)) * 2.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(7, 0) + DZeta(0, 2) * DZeta(2, 0) * DZeta(2, 2) * 4.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(7, 2) * 2.0;
                DiBj3(7, 16) = DZeta(0, 1) * (DZeta(2, 1) * DZeta(2, 1)) * 6.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(7, 1) * 3.0;
                DiBj3(7, 17) = DZeta(0, 2) * (DZeta(2, 1) * DZeta(2, 1)) * 2.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(7, 2) + DZeta(0, 1) * DZeta(2, 1) * DZeta(2, 2) * 4.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(7, 1) * 2.0;
                DiBj3(7, 18) = DZeta(0, 1) * (DZeta(2, 2) * DZeta(2, 2)) * 2.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(7, 1) + DZeta(0, 2) * DZeta(2, 1) * DZeta(2, 2) * 4.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(7, 2) * 2.0;
                DiBj3(7, 19) = DZeta(0, 2) * (DZeta(2, 2) * DZeta(2, 2)) * 6.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(7, 2) * 3.0;
                DiBj3(8, 1) = DZeta(8, 0);
                DiBj3(8, 2) = DZeta(8, 1);
                DiBj3(8, 3) = DZeta(8, 2);
                DiBj3(8, 4) = DZeta(2, 0) * DZeta(3, 0) * 2.0 + DZeta(0, 0) * DZeta(8, 0) * 2.0;
                DiBj3(8, 5) = DZeta(2, 0) * DZeta(3, 1) + DZeta(2, 1) * DZeta(3, 0) + DZeta(0, 0) * DZeta(8, 1) + DZeta(0, 1) * DZeta(8, 0);
                DiBj3(8, 6) = DZeta(2, 0) * DZeta(3, 2) + DZeta(2, 2) * DZeta(3, 0) + DZeta(0, 0) * DZeta(8, 2) + DZeta(0, 2) * DZeta(8, 0);
                DiBj3(8, 7) = DZeta(2, 1) * DZeta(3, 1) * 2.0 + DZeta(0, 1) * DZeta(8, 1) * 2.0;
                DiBj3(8, 8) = DZeta(2, 1) * DZeta(3, 2) + DZeta(2, 2) * DZeta(3, 1) + DZeta(0, 1) * DZeta(8, 2) + DZeta(0, 2) * DZeta(8, 1);
                DiBj3(8, 9) = DZeta(2, 2) * DZeta(3, 2) * 2.0 + DZeta(0, 2) * DZeta(8, 2) * 2.0;
                DiBj3(8, 10) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(8, 0) * 3.0 + DZeta(0, 0) * DZeta(2, 0) * DZeta(3, 0) * 6.0;
                DiBj3(8, 11) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(8, 1) + DZeta(0, 0) * DZeta(2, 0) * DZeta(3, 1) * 2.0 + DZeta(0, 0) * DZeta(2, 1) * DZeta(3, 0) * 2.0 + DZeta(0, 1) * DZeta(2, 0) * DZeta(3, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(8, 0) * 2.0;
                DiBj3(8, 12) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(8, 2) + DZeta(0, 0) * DZeta(2, 0) * DZeta(3, 2) * 2.0 + DZeta(0, 0) * DZeta(2, 2) * DZeta(3, 0) * 2.0 + DZeta(0, 2) * DZeta(2, 0) * DZeta(3, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(8, 0) * 2.0;
                DiBj3(8, 13) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(8, 0) + DZeta(0, 0) * DZeta(2, 1) * DZeta(3, 1) * 2.0 + DZeta(0, 1) * DZeta(2, 0) * DZeta(3, 1) * 2.0 + DZeta(0, 1) * DZeta(2, 1) * DZeta(3, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(8, 1) * 2.0;
                DiBj3(8, 14) = DZeta(0, 0) * DZeta(2, 1) * DZeta(3, 2) + DZeta(0, 0) * DZeta(2, 2) * DZeta(3, 1) + DZeta(0, 1) * DZeta(2, 0) * DZeta(3, 2) + DZeta(0, 1) * DZeta(2, 2) * DZeta(3, 0) + DZeta(0, 2) * DZeta(2, 0) * DZeta(3, 1) + DZeta(0, 2) * DZeta(2, 1) * DZeta(3, 0) + DZeta(0, 0) * DZeta(0, 1) * DZeta(8, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(8, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(8, 0);
                DiBj3(8, 15) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(8, 0) + DZeta(0, 0) * DZeta(2, 2) * DZeta(3, 2) * 2.0 + DZeta(0, 2) * DZeta(2, 0) * DZeta(3, 2) * 2.0 + DZeta(0, 2) * DZeta(2, 2) * DZeta(3, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(8, 2) * 2.0;
                DiBj3(8, 16) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(8, 1) * 3.0 + DZeta(0, 1) * DZeta(2, 1) * DZeta(3, 1) * 6.0;
                DiBj3(8, 17) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(8, 2) + DZeta(0, 1) * DZeta(2, 1) * DZeta(3, 2) * 2.0 + DZeta(0, 1) * DZeta(2, 2) * DZeta(3, 1) * 2.0 + DZeta(0, 2) * DZeta(2, 1) * DZeta(3, 1) * 2.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(8, 1) * 2.0;
                DiBj3(8, 18) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(8, 1) + DZeta(0, 1) * DZeta(2, 2) * DZeta(3, 2) * 2.0 + DZeta(0, 2) * DZeta(2, 1) * DZeta(3, 2) * 2.0 + DZeta(0, 2) * DZeta(2, 2) * DZeta(3, 1) * 2.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(8, 2) * 2.0;
                DiBj3(8, 19) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(8, 2) * 3.0 + DZeta(0, 2) * DZeta(2, 2) * DZeta(3, 2) * 6.0;
                DiBj3(9, 1) = DZeta(9, 0);
                DiBj3(9, 2) = DZeta(9, 1);
                DiBj3(9, 3) = DZeta(9, 2);
                DiBj3(9, 4) = (DZeta(3, 0) * DZeta(3, 0)) * 2.0 + DZeta(0, 0) * DZeta(9, 0) * 2.0;
                DiBj3(9, 5) = DZeta(3, 0) * DZeta(3, 1) * 2.0 + DZeta(0, 0) * DZeta(9, 1) + DZeta(0, 1) * DZeta(9, 0);
                DiBj3(9, 6) = DZeta(3, 0) * DZeta(3, 2) * 2.0 + DZeta(0, 0) * DZeta(9, 2) + DZeta(0, 2) * DZeta(9, 0);
                DiBj3(9, 7) = (DZeta(3, 1) * DZeta(3, 1)) * 2.0 + DZeta(0, 1) * DZeta(9, 1) * 2.0;
                DiBj3(9, 8) = DZeta(3, 1) * DZeta(3, 2) * 2.0 + DZeta(0, 1) * DZeta(9, 2) + DZeta(0, 2) * DZeta(9, 1);
                DiBj3(9, 9) = (DZeta(3, 2) * DZeta(3, 2)) * 2.0 + DZeta(0, 2) * DZeta(9, 2) * 2.0;
                DiBj3(9, 10) = DZeta(0, 0) * (DZeta(3, 0) * DZeta(3, 0)) * 6.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(9, 0) * 3.0;
                DiBj3(9, 11) = DZeta(0, 1) * (DZeta(3, 0) * DZeta(3, 0)) * 2.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(9, 1) + DZeta(0, 0) * DZeta(3, 0) * DZeta(3, 1) * 4.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(9, 0) * 2.0;
                DiBj3(9, 12) = DZeta(0, 2) * (DZeta(3, 0) * DZeta(3, 0)) * 2.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(9, 2) + DZeta(0, 0) * DZeta(3, 0) * DZeta(3, 2) * 4.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(9, 0) * 2.0;
                DiBj3(9, 13) = DZeta(0, 0) * (DZeta(3, 1) * DZeta(3, 1)) * 2.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(9, 0) + DZeta(0, 1) * DZeta(3, 0) * DZeta(3, 1) * 4.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(9, 1) * 2.0;
                DiBj3(9, 14) = DZeta(0, 0) * DZeta(3, 1) * DZeta(3, 2) * 2.0 + DZeta(0, 1) * DZeta(3, 0) * DZeta(3, 2) * 2.0 + DZeta(0, 2) * DZeta(3, 0) * DZeta(3, 1) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(9, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(9, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(9, 0);
                DiBj3(9, 15) = DZeta(0, 0) * (DZeta(3, 2) * DZeta(3, 2)) * 2.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(9, 0) + DZeta(0, 2) * DZeta(3, 0) * DZeta(3, 2) * 4.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(9, 2) * 2.0;
                DiBj3(9, 16) = DZeta(0, 1) * (DZeta(3, 1) * DZeta(3, 1)) * 6.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(9, 1) * 3.0;
                DiBj3(9, 17) = DZeta(0, 2) * (DZeta(3, 1) * DZeta(3, 1)) * 2.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(9, 2) + DZeta(0, 1) * DZeta(3, 1) * DZeta(3, 2) * 4.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(9, 1) * 2.0;
                DiBj3(9, 18) = DZeta(0, 1) * (DZeta(3, 2) * DZeta(3, 2)) * 2.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(9, 1) + DZeta(0, 2) * DZeta(3, 1) * DZeta(3, 2) * 4.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(9, 2) * 2.0;
                DiBj3(9, 19) = DZeta(0, 2) * (DZeta(3, 2) * DZeta(3, 2)) * 6.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(9, 2) * 3.0;
                DiBj3(10, 1) = DZeta(10, 0);
                DiBj3(10, 2) = DZeta(10, 1);
                DiBj3(10, 3) = DZeta(10, 2);
                DiBj3(10, 4) = DZeta(1, 0) * DZeta(4, 0) * 6.0 + DZeta(0, 0) * DZeta(10, 0) * 2.0;
                DiBj3(10, 5) = DZeta(1, 0) * DZeta(4, 1) * 3.0 + DZeta(1, 1) * DZeta(4, 0) * 3.0 + DZeta(0, 0) * DZeta(10, 1) + DZeta(0, 1) * DZeta(10, 0);
                DiBj3(10, 6) = DZeta(1, 0) * DZeta(4, 2) * 3.0 + DZeta(1, 2) * DZeta(4, 0) * 3.0 + DZeta(0, 0) * DZeta(10, 2) + DZeta(0, 2) * DZeta(10, 0);
                DiBj3(10, 7) = DZeta(1, 1) * DZeta(4, 1) * 6.0 + DZeta(0, 1) * DZeta(10, 1) * 2.0;
                DiBj3(10, 8) = DZeta(1, 1) * DZeta(4, 2) * 3.0 + DZeta(1, 2) * DZeta(4, 1) * 3.0 + DZeta(0, 1) * DZeta(10, 2) + DZeta(0, 2) * DZeta(10, 1);
                DiBj3(10, 9) = DZeta(1, 2) * DZeta(4, 2) * 6.0 + DZeta(0, 2) * DZeta(10, 2) * 2.0;
                DiBj3(10, 10) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(10, 0) * 3.0 + (DZeta(1, 0) * DZeta(1, 0) * DZeta(1, 0)) * 6.0 + DZeta(0, 0) * DZeta(1, 0) * DZeta(4, 0) * 1.8E+1;
                DiBj3(10, 11) = (DZeta(1, 0) * DZeta(1, 0)) * DZeta(1, 1) * 6.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(10, 1) + DZeta(0, 0) * DZeta(1, 0) * DZeta(4, 1) * 6.0 + DZeta(0, 0) * DZeta(1, 1) * DZeta(4, 0) * 6.0 + DZeta(0, 1) * DZeta(1, 0) * DZeta(4, 0) * 6.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(10, 0) * 2.0;
                DiBj3(10, 12) = (DZeta(1, 0) * DZeta(1, 0)) * DZeta(1, 2) * 6.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(10, 2) + DZeta(0, 0) * DZeta(1, 0) * DZeta(4, 2) * 6.0 + DZeta(0, 0) * DZeta(1, 2) * DZeta(4, 0) * 6.0 + DZeta(0, 2) * DZeta(1, 0) * DZeta(4, 0) * 6.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(10, 0) * 2.0;
                DiBj3(10, 13) = DZeta(1, 0) * (DZeta(1, 1) * DZeta(1, 1)) * 6.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(10, 0) + DZeta(0, 0) * DZeta(1, 1) * DZeta(4, 1) * 6.0 + DZeta(0, 1) * DZeta(1, 0) * DZeta(4, 1) * 6.0 + DZeta(0, 1) * DZeta(1, 1) * DZeta(4, 0) * 6.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(10, 1) * 2.0;
                DiBj3(10, 14) = DZeta(1, 0) * DZeta(1, 1) * DZeta(1, 2) * 6.0 + DZeta(0, 0) * DZeta(1, 1) * DZeta(4, 2) * 3.0 + DZeta(0, 0) * DZeta(1, 2) * DZeta(4, 1) * 3.0 + DZeta(0, 1) * DZeta(1, 0) * DZeta(4, 2) * 3.0 + DZeta(0, 1) * DZeta(1, 2) * DZeta(4, 0) * 3.0 + DZeta(0, 2) * DZeta(1, 0) * DZeta(4, 1) * 3.0 + DZeta(0, 2) * DZeta(1, 1) * DZeta(4, 0) * 3.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(10, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(10, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(10, 0);
                DiBj3(10, 15) = DZeta(1, 0) * (DZeta(1, 2) * DZeta(1, 2)) * 6.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(10, 0) + DZeta(0, 0) * DZeta(1, 2) * DZeta(4, 2) * 6.0 + DZeta(0, 2) * DZeta(1, 0) * DZeta(4, 2) * 6.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(4, 0) * 6.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(10, 2) * 2.0;
                DiBj3(10, 16) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(10, 1) * 3.0 + (DZeta(1, 1) * DZeta(1, 1) * DZeta(1, 1)) * 6.0 + DZeta(0, 1) * DZeta(1, 1) * DZeta(4, 1) * 1.8E+1;
                DiBj3(10, 17) = (DZeta(1, 1) * DZeta(1, 1)) * DZeta(1, 2) * 6.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(10, 2) + DZeta(0, 1) * DZeta(1, 1) * DZeta(4, 2) * 6.0 + DZeta(0, 1) * DZeta(1, 2) * DZeta(4, 1) * 6.0 + DZeta(0, 2) * DZeta(1, 1) * DZeta(4, 1) * 6.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(10, 1) * 2.0;
                DiBj3(10, 18) = DZeta(1, 1) * (DZeta(1, 2) * DZeta(1, 2)) * 6.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(10, 1) + DZeta(0, 1) * DZeta(1, 2) * DZeta(4, 2) * 6.0 + DZeta(0, 2) * DZeta(1, 1) * DZeta(4, 2) * 6.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(4, 1) * 6.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(10, 2) * 2.0;
                DiBj3(10, 19) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(10, 2) * 3.0 + (DZeta(1, 2) * DZeta(1, 2) * DZeta(1, 2)) * 6.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(4, 2) * 1.8E+1;
                DiBj3(11, 1) = DZeta(11, 0);
                DiBj3(11, 2) = DZeta(11, 1);
                DiBj3(11, 3) = DZeta(11, 2);
                DiBj3(11, 4) = DZeta(1, 0) * DZeta(5, 0) * 4.0 + DZeta(2, 0) * DZeta(4, 0) * 2.0 + DZeta(0, 0) * DZeta(11, 0) * 2.0;
                DiBj3(11, 5) = DZeta(1, 0) * DZeta(5, 1) * 2.0 + DZeta(1, 1) * DZeta(5, 0) * 2.0 + DZeta(2, 0) * DZeta(4, 1) + DZeta(2, 1) * DZeta(4, 0) + DZeta(0, 0) * DZeta(11, 1) + DZeta(0, 1) * DZeta(11, 0);
                DiBj3(11, 6) = DZeta(1, 0) * DZeta(5, 2) * 2.0 + DZeta(1, 2) * DZeta(5, 0) * 2.0 + DZeta(2, 0) * DZeta(4, 2) + DZeta(2, 2) * DZeta(4, 0) + DZeta(0, 0) * DZeta(11, 2) + DZeta(0, 2) * DZeta(11, 0);
                DiBj3(11, 7) = DZeta(1, 1) * DZeta(5, 1) * 4.0 + DZeta(2, 1) * DZeta(4, 1) * 2.0 + DZeta(0, 1) * DZeta(11, 1) * 2.0;
                DiBj3(11, 8) = DZeta(1, 1) * DZeta(5, 2) * 2.0 + DZeta(1, 2) * DZeta(5, 1) * 2.0 + DZeta(2, 1) * DZeta(4, 2) + DZeta(2, 2) * DZeta(4, 1) + DZeta(0, 1) * DZeta(11, 2) + DZeta(0, 2) * DZeta(11, 1);
                DiBj3(11, 9) = DZeta(1, 2) * DZeta(5, 2) * 4.0 + DZeta(2, 2) * DZeta(4, 2) * 2.0 + DZeta(0, 2) * DZeta(11, 2) * 2.0;
                DiBj3(11, 10) = (DZeta(1, 0) * DZeta(1, 0)) * DZeta(2, 0) * 6.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(11, 0) * 3.0 + DZeta(0, 0) * DZeta(1, 0) * DZeta(5, 0) * 1.2E+1 + DZeta(0, 0) * DZeta(2, 0) * DZeta(4, 0) * 6.0;
                DiBj3(11, 11) = (DZeta(1, 0) * DZeta(1, 0)) * DZeta(2, 1) * 2.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(11, 1) + DZeta(1, 0) * DZeta(1, 1) * DZeta(2, 0) * 4.0 + DZeta(0, 0) * DZeta(1, 0) * DZeta(5, 1) * 4.0 + DZeta(0, 0) * DZeta(1, 1) * DZeta(5, 0) * 4.0 + DZeta(0, 0) * DZeta(2, 0) * DZeta(4, 1) * 2.0 + DZeta(0, 0) * DZeta(2, 1) * DZeta(4, 0) * 2.0 + DZeta(0, 1) * DZeta(1, 0) * DZeta(5, 0) * 4.0 + DZeta(0, 1) * DZeta(2, 0) * DZeta(4, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(11, 0) * 2.0;
                DiBj3(11, 12) = (DZeta(1, 0) * DZeta(1, 0)) * DZeta(2, 2) * 2.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(11, 2) + DZeta(1, 0) * DZeta(1, 2) * DZeta(2, 0) * 4.0 + DZeta(0, 0) * DZeta(1, 0) * DZeta(5, 2) * 4.0 + DZeta(0, 0) * DZeta(1, 2) * DZeta(5, 0) * 4.0 + DZeta(0, 0) * DZeta(2, 0) * DZeta(4, 2) * 2.0 + DZeta(0, 0) * DZeta(2, 2) * DZeta(4, 0) * 2.0 + DZeta(0, 2) * DZeta(1, 0) * DZeta(5, 0) * 4.0 + DZeta(0, 2) * DZeta(2, 0) * DZeta(4, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(11, 0) * 2.0;
                DiBj3(11, 13) = (DZeta(1, 1) * DZeta(1, 1)) * DZeta(2, 0) * 2.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(11, 0) + DZeta(1, 0) * DZeta(1, 1) * DZeta(2, 1) * 4.0 + DZeta(0, 0) * DZeta(1, 1) * DZeta(5, 1) * 4.0 + DZeta(0, 0) * DZeta(2, 1) * DZeta(4, 1) * 2.0 + DZeta(0, 1) * DZeta(1, 0) * DZeta(5, 1) * 4.0 + DZeta(0, 1) * DZeta(1, 1) * DZeta(5, 0) * 4.0 + DZeta(0, 1) * DZeta(2, 0) * DZeta(4, 1) * 2.0 + DZeta(0, 1) * DZeta(2, 1) * DZeta(4, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(11, 1) * 2.0;
                DiBj3(11, 14) = DZeta(1, 0) * DZeta(1, 1) * DZeta(2, 2) * 2.0 + DZeta(1, 0) * DZeta(1, 2) * DZeta(2, 1) * 2.0 + DZeta(1, 1) * DZeta(1, 2) * DZeta(2, 0) * 2.0 + DZeta(0, 0) * DZeta(1, 1) * DZeta(5, 2) * 2.0 + DZeta(0, 0) * DZeta(1, 2) * DZeta(5, 1) * 2.0 + DZeta(0, 0) * DZeta(2, 1) * DZeta(4, 2) + DZeta(0, 0) * DZeta(2, 2) * DZeta(4, 1) + DZeta(0, 1) * DZeta(1, 0) * DZeta(5, 2) * 2.0 + DZeta(0, 1) * DZeta(1, 2) * DZeta(5, 0) * 2.0 + DZeta(0, 1) * DZeta(2, 0) * DZeta(4, 2) + DZeta(0, 1) * DZeta(2, 2) * DZeta(4, 0) + DZeta(0, 2) * DZeta(1, 0) * DZeta(5, 1) * 2.0 + DZeta(0, 2) * DZeta(1, 1) * DZeta(5, 0) * 2.0 + DZeta(0, 2) * DZeta(2, 0) * DZeta(4, 1) + DZeta(0, 2) * DZeta(2, 1) * DZeta(4, 0) + DZeta(0, 0) * DZeta(0, 1) * DZeta(11, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(11, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(11, 0);
                DiBj3(11, 15) = (DZeta(1, 2) * DZeta(1, 2)) * DZeta(2, 0) * 2.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(11, 0) + DZeta(1, 0) * DZeta(1, 2) * DZeta(2, 2) * 4.0 + DZeta(0, 0) * DZeta(1, 2) * DZeta(5, 2) * 4.0 + DZeta(0, 0) * DZeta(2, 2) * DZeta(4, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 0) * DZeta(5, 2) * 4.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(5, 0) * 4.0 + DZeta(0, 2) * DZeta(2, 0) * DZeta(4, 2) * 2.0 + DZeta(0, 2) * DZeta(2, 2) * DZeta(4, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(11, 2) * 2.0;
                DiBj3(11, 16) = (DZeta(1, 1) * DZeta(1, 1)) * DZeta(2, 1) * 6.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(11, 1) * 3.0 + DZeta(0, 1) * DZeta(1, 1) * DZeta(5, 1) * 1.2E+1 + DZeta(0, 1) * DZeta(2, 1) * DZeta(4, 1) * 6.0;
                DiBj3(11, 17) = (DZeta(1, 1) * DZeta(1, 1)) * DZeta(2, 2) * 2.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(11, 2) + DZeta(1, 1) * DZeta(1, 2) * DZeta(2, 1) * 4.0 + DZeta(0, 1) * DZeta(1, 1) * DZeta(5, 2) * 4.0 + DZeta(0, 1) * DZeta(1, 2) * DZeta(5, 1) * 4.0 + DZeta(0, 1) * DZeta(2, 1) * DZeta(4, 2) * 2.0 + DZeta(0, 1) * DZeta(2, 2) * DZeta(4, 1) * 2.0 + DZeta(0, 2) * DZeta(1, 1) * DZeta(5, 1) * 4.0 + DZeta(0, 2) * DZeta(2, 1) * DZeta(4, 1) * 2.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(11, 1) * 2.0;
                DiBj3(11, 18) = (DZeta(1, 2) * DZeta(1, 2)) * DZeta(2, 1) * 2.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(11, 1) + DZeta(1, 1) * DZeta(1, 2) * DZeta(2, 2) * 4.0 + DZeta(0, 1) * DZeta(1, 2) * DZeta(5, 2) * 4.0 + DZeta(0, 1) * DZeta(2, 2) * DZeta(4, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 1) * DZeta(5, 2) * 4.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(5, 1) * 4.0 + DZeta(0, 2) * DZeta(2, 1) * DZeta(4, 2) * 2.0 + DZeta(0, 2) * DZeta(2, 2) * DZeta(4, 1) * 2.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(11, 2) * 2.0;
                DiBj3(11, 19) = (DZeta(1, 2) * DZeta(1, 2)) * DZeta(2, 2) * 6.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(11, 2) * 3.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(5, 2) * 1.2E+1 + DZeta(0, 2) * DZeta(2, 2) * DZeta(4, 2) * 6.0;
                DiBj3(12, 1) = DZeta(12, 0);
                DiBj3(12, 2) = DZeta(12, 1);
                DiBj3(12, 3) = DZeta(12, 2);
                DiBj3(12, 4) = DZeta(1, 0) * DZeta(6, 0) * 4.0 + DZeta(3, 0) * DZeta(4, 0) * 2.0 + DZeta(0, 0) * DZeta(12, 0) * 2.0;
                DiBj3(12, 5) = DZeta(1, 0) * DZeta(6, 1) * 2.0 + DZeta(1, 1) * DZeta(6, 0) * 2.0 + DZeta(3, 0) * DZeta(4, 1) + DZeta(3, 1) * DZeta(4, 0) + DZeta(0, 0) * DZeta(12, 1) + DZeta(0, 1) * DZeta(12, 0);
                DiBj3(12, 6) = DZeta(1, 0) * DZeta(6, 2) * 2.0 + DZeta(1, 2) * DZeta(6, 0) * 2.0 + DZeta(3, 0) * DZeta(4, 2) + DZeta(3, 2) * DZeta(4, 0) + DZeta(0, 0) * DZeta(12, 2) + DZeta(0, 2) * DZeta(12, 0);
                DiBj3(12, 7) = DZeta(1, 1) * DZeta(6, 1) * 4.0 + DZeta(3, 1) * DZeta(4, 1) * 2.0 + DZeta(0, 1) * DZeta(12, 1) * 2.0;
                DiBj3(12, 8) = DZeta(1, 1) * DZeta(6, 2) * 2.0 + DZeta(1, 2) * DZeta(6, 1) * 2.0 + DZeta(3, 1) * DZeta(4, 2) + DZeta(3, 2) * DZeta(4, 1) + DZeta(0, 1) * DZeta(12, 2) + DZeta(0, 2) * DZeta(12, 1);
                DiBj3(12, 9) = DZeta(1, 2) * DZeta(6, 2) * 4.0 + DZeta(3, 2) * DZeta(4, 2) * 2.0 + DZeta(0, 2) * DZeta(12, 2) * 2.0;
                DiBj3(12, 10) = (DZeta(1, 0) * DZeta(1, 0)) * DZeta(3, 0) * 6.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(12, 0) * 3.0 + DZeta(0, 0) * DZeta(1, 0) * DZeta(6, 0) * 1.2E+1 + DZeta(0, 0) * DZeta(3, 0) * DZeta(4, 0) * 6.0;
                DiBj3(12, 11) = (DZeta(1, 0) * DZeta(1, 0)) * DZeta(3, 1) * 2.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(12, 1) + DZeta(1, 0) * DZeta(1, 1) * DZeta(3, 0) * 4.0 + DZeta(0, 0) * DZeta(1, 0) * DZeta(6, 1) * 4.0 + DZeta(0, 0) * DZeta(1, 1) * DZeta(6, 0) * 4.0 + DZeta(0, 0) * DZeta(3, 0) * DZeta(4, 1) * 2.0 + DZeta(0, 0) * DZeta(3, 1) * DZeta(4, 0) * 2.0 + DZeta(0, 1) * DZeta(1, 0) * DZeta(6, 0) * 4.0 + DZeta(0, 1) * DZeta(3, 0) * DZeta(4, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(12, 0) * 2.0;
                DiBj3(12, 12) = (DZeta(1, 0) * DZeta(1, 0)) * DZeta(3, 2) * 2.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(12, 2) + DZeta(1, 0) * DZeta(1, 2) * DZeta(3, 0) * 4.0 + DZeta(0, 0) * DZeta(1, 0) * DZeta(6, 2) * 4.0 + DZeta(0, 0) * DZeta(1, 2) * DZeta(6, 0) * 4.0 + DZeta(0, 0) * DZeta(3, 0) * DZeta(4, 2) * 2.0 + DZeta(0, 0) * DZeta(3, 2) * DZeta(4, 0) * 2.0 + DZeta(0, 2) * DZeta(1, 0) * DZeta(6, 0) * 4.0 + DZeta(0, 2) * DZeta(3, 0) * DZeta(4, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(12, 0) * 2.0;
                DiBj3(12, 13) = (DZeta(1, 1) * DZeta(1, 1)) * DZeta(3, 0) * 2.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(12, 0) + DZeta(1, 0) * DZeta(1, 1) * DZeta(3, 1) * 4.0 + DZeta(0, 0) * DZeta(1, 1) * DZeta(6, 1) * 4.0 + DZeta(0, 0) * DZeta(3, 1) * DZeta(4, 1) * 2.0 + DZeta(0, 1) * DZeta(1, 0) * DZeta(6, 1) * 4.0 + DZeta(0, 1) * DZeta(1, 1) * DZeta(6, 0) * 4.0 + DZeta(0, 1) * DZeta(3, 0) * DZeta(4, 1) * 2.0 + DZeta(0, 1) * DZeta(3, 1) * DZeta(4, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(12, 1) * 2.0;
                DiBj3(12, 14) = DZeta(1, 0) * DZeta(1, 1) * DZeta(3, 2) * 2.0 + DZeta(1, 0) * DZeta(1, 2) * DZeta(3, 1) * 2.0 + DZeta(1, 1) * DZeta(1, 2) * DZeta(3, 0) * 2.0 + DZeta(0, 0) * DZeta(1, 1) * DZeta(6, 2) * 2.0 + DZeta(0, 0) * DZeta(1, 2) * DZeta(6, 1) * 2.0 + DZeta(0, 0) * DZeta(3, 1) * DZeta(4, 2) + DZeta(0, 0) * DZeta(3, 2) * DZeta(4, 1) + DZeta(0, 1) * DZeta(1, 0) * DZeta(6, 2) * 2.0 + DZeta(0, 1) * DZeta(1, 2) * DZeta(6, 0) * 2.0 + DZeta(0, 1) * DZeta(3, 0) * DZeta(4, 2) + DZeta(0, 1) * DZeta(3, 2) * DZeta(4, 0) + DZeta(0, 2) * DZeta(1, 0) * DZeta(6, 1) * 2.0 + DZeta(0, 2) * DZeta(1, 1) * DZeta(6, 0) * 2.0 + DZeta(0, 2) * DZeta(3, 0) * DZeta(4, 1) + DZeta(0, 2) * DZeta(3, 1) * DZeta(4, 0) + DZeta(0, 0) * DZeta(0, 1) * DZeta(12, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(12, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(12, 0);
                DiBj3(12, 15) = (DZeta(1, 2) * DZeta(1, 2)) * DZeta(3, 0) * 2.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(12, 0) + DZeta(1, 0) * DZeta(1, 2) * DZeta(3, 2) * 4.0 + DZeta(0, 0) * DZeta(1, 2) * DZeta(6, 2) * 4.0 + DZeta(0, 0) * DZeta(3, 2) * DZeta(4, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 0) * DZeta(6, 2) * 4.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(6, 0) * 4.0 + DZeta(0, 2) * DZeta(3, 0) * DZeta(4, 2) * 2.0 + DZeta(0, 2) * DZeta(3, 2) * DZeta(4, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(12, 2) * 2.0;
                DiBj3(12, 16) = (DZeta(1, 1) * DZeta(1, 1)) * DZeta(3, 1) * 6.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(12, 1) * 3.0 + DZeta(0, 1) * DZeta(1, 1) * DZeta(6, 1) * 1.2E+1 + DZeta(0, 1) * DZeta(3, 1) * DZeta(4, 1) * 6.0;
                DiBj3(12, 17) = (DZeta(1, 1) * DZeta(1, 1)) * DZeta(3, 2) * 2.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(12, 2) + DZeta(1, 1) * DZeta(1, 2) * DZeta(3, 1) * 4.0 + DZeta(0, 1) * DZeta(1, 1) * DZeta(6, 2) * 4.0 + DZeta(0, 1) * DZeta(1, 2) * DZeta(6, 1) * 4.0 + DZeta(0, 1) * DZeta(3, 1) * DZeta(4, 2) * 2.0 + DZeta(0, 1) * DZeta(3, 2) * DZeta(4, 1) * 2.0 + DZeta(0, 2) * DZeta(1, 1) * DZeta(6, 1) * 4.0 + DZeta(0, 2) * DZeta(3, 1) * DZeta(4, 1) * 2.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(12, 1) * 2.0;
                DiBj3(12, 18) = (DZeta(1, 2) * DZeta(1, 2)) * DZeta(3, 1) * 2.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(12, 1) + DZeta(1, 1) * DZeta(1, 2) * DZeta(3, 2) * 4.0 + DZeta(0, 1) * DZeta(1, 2) * DZeta(6, 2) * 4.0 + DZeta(0, 1) * DZeta(3, 2) * DZeta(4, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 1) * DZeta(6, 2) * 4.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(6, 1) * 4.0 + DZeta(0, 2) * DZeta(3, 1) * DZeta(4, 2) * 2.0 + DZeta(0, 2) * DZeta(3, 2) * DZeta(4, 1) * 2.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(12, 2) * 2.0;
                DiBj3(12, 19) = (DZeta(1, 2) * DZeta(1, 2)) * DZeta(3, 2) * 6.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(12, 2) * 3.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(6, 2) * 1.2E+1 + DZeta(0, 2) * DZeta(3, 2) * DZeta(4, 2) * 6.0;
                DiBj3(13, 1) = DZeta(13, 0);
                DiBj3(13, 2) = DZeta(13, 1);
                DiBj3(13, 3) = DZeta(13, 2);
                DiBj3(13, 4) = DZeta(2, 0) * DZeta(5, 0) * 4.0 + DZeta(1, 0) * DZeta(7, 0) * 2.0 + DZeta(0, 0) * DZeta(13, 0) * 2.0;
                DiBj3(13, 5) = DZeta(2, 0) * DZeta(5, 1) * 2.0 + DZeta(2, 1) * DZeta(5, 0) * 2.0 + DZeta(1, 0) * DZeta(7, 1) + DZeta(1, 1) * DZeta(7, 0) + DZeta(0, 0) * DZeta(13, 1) + DZeta(0, 1) * DZeta(13, 0);
                DiBj3(13, 6) = DZeta(2, 0) * DZeta(5, 2) * 2.0 + DZeta(2, 2) * DZeta(5, 0) * 2.0 + DZeta(1, 0) * DZeta(7, 2) + DZeta(1, 2) * DZeta(7, 0) + DZeta(0, 0) * DZeta(13, 2) + DZeta(0, 2) * DZeta(13, 0);
                DiBj3(13, 7) = DZeta(2, 1) * DZeta(5, 1) * 4.0 + DZeta(1, 1) * DZeta(7, 1) * 2.0 + DZeta(0, 1) * DZeta(13, 1) * 2.0;
                DiBj3(13, 8) = DZeta(2, 1) * DZeta(5, 2) * 2.0 + DZeta(2, 2) * DZeta(5, 1) * 2.0 + DZeta(1, 1) * DZeta(7, 2) + DZeta(1, 2) * DZeta(7, 1) + DZeta(0, 1) * DZeta(13, 2) + DZeta(0, 2) * DZeta(13, 1);
                DiBj3(13, 9) = DZeta(2, 2) * DZeta(5, 2) * 4.0 + DZeta(1, 2) * DZeta(7, 2) * 2.0 + DZeta(0, 2) * DZeta(13, 2) * 2.0;
                DiBj3(13, 10) = DZeta(1, 0) * (DZeta(2, 0) * DZeta(2, 0)) * 6.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(13, 0) * 3.0 + DZeta(0, 0) * DZeta(2, 0) * DZeta(5, 0) * 1.2E+1 + DZeta(0, 0) * DZeta(1, 0) * DZeta(7, 0) * 6.0;
                DiBj3(13, 11) = DZeta(1, 1) * (DZeta(2, 0) * DZeta(2, 0)) * 2.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(13, 1) + DZeta(1, 0) * DZeta(2, 0) * DZeta(2, 1) * 4.0 + DZeta(0, 0) * DZeta(2, 0) * DZeta(5, 1) * 4.0 + DZeta(0, 0) * DZeta(2, 1) * DZeta(5, 0) * 4.0 + DZeta(0, 1) * DZeta(2, 0) * DZeta(5, 0) * 4.0 + DZeta(0, 0) * DZeta(1, 0) * DZeta(7, 1) * 2.0 + DZeta(0, 0) * DZeta(1, 1) * DZeta(7, 0) * 2.0 + DZeta(0, 1) * DZeta(1, 0) * DZeta(7, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(13, 0) * 2.0;
                DiBj3(13, 12) = DZeta(1, 2) * (DZeta(2, 0) * DZeta(2, 0)) * 2.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(13, 2) + DZeta(1, 0) * DZeta(2, 0) * DZeta(2, 2) * 4.0 + DZeta(0, 0) * DZeta(2, 0) * DZeta(5, 2) * 4.0 + DZeta(0, 0) * DZeta(2, 2) * DZeta(5, 0) * 4.0 + DZeta(0, 2) * DZeta(2, 0) * DZeta(5, 0) * 4.0 + DZeta(0, 0) * DZeta(1, 0) * DZeta(7, 2) * 2.0 + DZeta(0, 0) * DZeta(1, 2) * DZeta(7, 0) * 2.0 + DZeta(0, 2) * DZeta(1, 0) * DZeta(7, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(13, 0) * 2.0;
                DiBj3(13, 13) = DZeta(1, 0) * (DZeta(2, 1) * DZeta(2, 1)) * 2.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(13, 0) + DZeta(1, 1) * DZeta(2, 0) * DZeta(2, 1) * 4.0 + DZeta(0, 0) * DZeta(2, 1) * DZeta(5, 1) * 4.0 + DZeta(0, 1) * DZeta(2, 0) * DZeta(5, 1) * 4.0 + DZeta(0, 1) * DZeta(2, 1) * DZeta(5, 0) * 4.0 + DZeta(0, 0) * DZeta(1, 1) * DZeta(7, 1) * 2.0 + DZeta(0, 1) * DZeta(1, 0) * DZeta(7, 1) * 2.0 + DZeta(0, 1) * DZeta(1, 1) * DZeta(7, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(13, 1) * 2.0;
                DiBj3(13, 14) = DZeta(1, 0) * DZeta(2, 1) * DZeta(2, 2) * 2.0 + DZeta(1, 1) * DZeta(2, 0) * DZeta(2, 2) * 2.0 + DZeta(1, 2) * DZeta(2, 0) * DZeta(2, 1) * 2.0 + DZeta(0, 0) * DZeta(2, 1) * DZeta(5, 2) * 2.0 + DZeta(0, 0) * DZeta(2, 2) * DZeta(5, 1) * 2.0 + DZeta(0, 1) * DZeta(2, 0) * DZeta(5, 2) * 2.0 + DZeta(0, 1) * DZeta(2, 2) * DZeta(5, 0) * 2.0 + DZeta(0, 2) * DZeta(2, 0) * DZeta(5, 1) * 2.0 + DZeta(0, 2) * DZeta(2, 1) * DZeta(5, 0) * 2.0 + DZeta(0, 0) * DZeta(1, 1) * DZeta(7, 2) + DZeta(0, 0) * DZeta(1, 2) * DZeta(7, 1) + DZeta(0, 1) * DZeta(1, 0) * DZeta(7, 2) + DZeta(0, 1) * DZeta(1, 2) * DZeta(7, 0) + DZeta(0, 2) * DZeta(1, 0) * DZeta(7, 1) + DZeta(0, 2) * DZeta(1, 1) * DZeta(7, 0) + DZeta(0, 0) * DZeta(0, 1) * DZeta(13, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(13, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(13, 0);
                DiBj3(13, 15) = DZeta(1, 0) * (DZeta(2, 2) * DZeta(2, 2)) * 2.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(13, 0) + DZeta(1, 2) * DZeta(2, 0) * DZeta(2, 2) * 4.0 + DZeta(0, 0) * DZeta(2, 2) * DZeta(5, 2) * 4.0 + DZeta(0, 2) * DZeta(2, 0) * DZeta(5, 2) * 4.0 + DZeta(0, 2) * DZeta(2, 2) * DZeta(5, 0) * 4.0 + DZeta(0, 0) * DZeta(1, 2) * DZeta(7, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 0) * DZeta(7, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(7, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(13, 2) * 2.0;
                DiBj3(13, 16) = DZeta(1, 1) * (DZeta(2, 1) * DZeta(2, 1)) * 6.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(13, 1) * 3.0 + DZeta(0, 1) * DZeta(2, 1) * DZeta(5, 1) * 1.2E+1 + DZeta(0, 1) * DZeta(1, 1) * DZeta(7, 1) * 6.0;
                DiBj3(13, 17) = DZeta(1, 2) * (DZeta(2, 1) * DZeta(2, 1)) * 2.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(13, 2) + DZeta(1, 1) * DZeta(2, 1) * DZeta(2, 2) * 4.0 + DZeta(0, 1) * DZeta(2, 1) * DZeta(5, 2) * 4.0 + DZeta(0, 1) * DZeta(2, 2) * DZeta(5, 1) * 4.0 + DZeta(0, 2) * DZeta(2, 1) * DZeta(5, 1) * 4.0 + DZeta(0, 1) * DZeta(1, 1) * DZeta(7, 2) * 2.0 + DZeta(0, 1) * DZeta(1, 2) * DZeta(7, 1) * 2.0 + DZeta(0, 2) * DZeta(1, 1) * DZeta(7, 1) * 2.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(13, 1) * 2.0;
                DiBj3(13, 18) = DZeta(1, 1) * (DZeta(2, 2) * DZeta(2, 2)) * 2.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(13, 1) + DZeta(1, 2) * DZeta(2, 1) * DZeta(2, 2) * 4.0 + DZeta(0, 1) * DZeta(2, 2) * DZeta(5, 2) * 4.0 + DZeta(0, 2) * DZeta(2, 1) * DZeta(5, 2) * 4.0 + DZeta(0, 2) * DZeta(2, 2) * DZeta(5, 1) * 4.0 + DZeta(0, 1) * DZeta(1, 2) * DZeta(7, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 1) * DZeta(7, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(7, 1) * 2.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(13, 2) * 2.0;
                DiBj3(13, 19) = DZeta(1, 2) * (DZeta(2, 2) * DZeta(2, 2)) * 6.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(13, 2) * 3.0 + DZeta(0, 2) * DZeta(2, 2) * DZeta(5, 2) * 1.2E+1 + DZeta(0, 2) * DZeta(1, 2) * DZeta(7, 2) * 6.0;
                DiBj3(14, 1) = DZeta(14, 0);
                DiBj3(14, 2) = DZeta(14, 1);
                DiBj3(14, 3) = DZeta(14, 2);
                DiBj3(14, 4) = DZeta(2, 0) * DZeta(6, 0) * 2.0 + DZeta(3, 0) * DZeta(5, 0) * 2.0 + DZeta(1, 0) * DZeta(8, 0) * 2.0 + DZeta(0, 0) * DZeta(14, 0) * 2.0;
                DiBj3(14, 5) = DZeta(2, 0) * DZeta(6, 1) + DZeta(2, 1) * DZeta(6, 0) + DZeta(3, 0) * DZeta(5, 1) + DZeta(3, 1) * DZeta(5, 0) + DZeta(1, 0) * DZeta(8, 1) + DZeta(1, 1) * DZeta(8, 0) + DZeta(0, 0) * DZeta(14, 1) + DZeta(0, 1) * DZeta(14, 0);
                DiBj3(14, 6) = DZeta(2, 0) * DZeta(6, 2) + DZeta(2, 2) * DZeta(6, 0) + DZeta(3, 0) * DZeta(5, 2) + DZeta(3, 2) * DZeta(5, 0) + DZeta(1, 0) * DZeta(8, 2) + DZeta(1, 2) * DZeta(8, 0) + DZeta(0, 0) * DZeta(14, 2) + DZeta(0, 2) * DZeta(14, 0);
                DiBj3(14, 7) = DZeta(2, 1) * DZeta(6, 1) * 2.0 + DZeta(3, 1) * DZeta(5, 1) * 2.0 + DZeta(1, 1) * DZeta(8, 1) * 2.0 + DZeta(0, 1) * DZeta(14, 1) * 2.0;
                DiBj3(14, 8) = DZeta(2, 1) * DZeta(6, 2) + DZeta(2, 2) * DZeta(6, 1) + DZeta(3, 1) * DZeta(5, 2) + DZeta(3, 2) * DZeta(5, 1) + DZeta(1, 1) * DZeta(8, 2) + DZeta(1, 2) * DZeta(8, 1) + DZeta(0, 1) * DZeta(14, 2) + DZeta(0, 2) * DZeta(14, 1);
                DiBj3(14, 9) = DZeta(2, 2) * DZeta(6, 2) * 2.0 + DZeta(3, 2) * DZeta(5, 2) * 2.0 + DZeta(1, 2) * DZeta(8, 2) * 2.0 + DZeta(0, 2) * DZeta(14, 2) * 2.0;
                DiBj3(14, 10) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(14, 0) * 3.0 + DZeta(1, 0) * DZeta(2, 0) * DZeta(3, 0) * 6.0 + DZeta(0, 0) * DZeta(2, 0) * DZeta(6, 0) * 6.0 + DZeta(0, 0) * DZeta(3, 0) * DZeta(5, 0) * 6.0 + DZeta(0, 0) * DZeta(1, 0) * DZeta(8, 0) * 6.0;
                DiBj3(14, 11) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(14, 1) + DZeta(1, 0) * DZeta(2, 0) * DZeta(3, 1) * 2.0 + DZeta(1, 0) * DZeta(2, 1) * DZeta(3, 0) * 2.0 + DZeta(1, 1) * DZeta(2, 0) * DZeta(3, 0) * 2.0 + DZeta(0, 0) * DZeta(2, 0) * DZeta(6, 1) * 2.0 + DZeta(0, 0) * DZeta(2, 1) * DZeta(6, 0) * 2.0 + DZeta(0, 0) * DZeta(3, 0) * DZeta(5, 1) * 2.0 + DZeta(0, 0) * DZeta(3, 1) * DZeta(5, 0) * 2.0 + DZeta(0, 1) * DZeta(2, 0) * DZeta(6, 0) * 2.0 + DZeta(0, 1) * DZeta(3, 0) * DZeta(5, 0) * 2.0 + DZeta(0, 0) * DZeta(1, 0) * DZeta(8, 1) * 2.0 + DZeta(0, 0) * DZeta(1, 1) * DZeta(8, 0) * 2.0 + DZeta(0, 1) * DZeta(1, 0) * DZeta(8, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(14, 0) * 2.0;
                DiBj3(14, 12) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(14, 2) + DZeta(1, 0) * DZeta(2, 0) * DZeta(3, 2) * 2.0 + DZeta(1, 0) * DZeta(2, 2) * DZeta(3, 0) * 2.0 + DZeta(1, 2) * DZeta(2, 0) * DZeta(3, 0) * 2.0 + DZeta(0, 0) * DZeta(2, 0) * DZeta(6, 2) * 2.0 + DZeta(0, 0) * DZeta(2, 2) * DZeta(6, 0) * 2.0 + DZeta(0, 0) * DZeta(3, 0) * DZeta(5, 2) * 2.0 + DZeta(0, 0) * DZeta(3, 2) * DZeta(5, 0) * 2.0 + DZeta(0, 2) * DZeta(2, 0) * DZeta(6, 0) * 2.0 + DZeta(0, 2) * DZeta(3, 0) * DZeta(5, 0) * 2.0 + DZeta(0, 0) * DZeta(1, 0) * DZeta(8, 2) * 2.0 + DZeta(0, 0) * DZeta(1, 2) * DZeta(8, 0) * 2.0 + DZeta(0, 2) * DZeta(1, 0) * DZeta(8, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(14, 0) * 2.0;
                DiBj3(14, 13) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(14, 0) + DZeta(1, 0) * DZeta(2, 1) * DZeta(3, 1) * 2.0 + DZeta(1, 1) * DZeta(2, 0) * DZeta(3, 1) * 2.0 + DZeta(1, 1) * DZeta(2, 1) * DZeta(3, 0) * 2.0 + DZeta(0, 0) * DZeta(2, 1) * DZeta(6, 1) * 2.0 + DZeta(0, 0) * DZeta(3, 1) * DZeta(5, 1) * 2.0 + DZeta(0, 1) * DZeta(2, 0) * DZeta(6, 1) * 2.0 + DZeta(0, 1) * DZeta(2, 1) * DZeta(6, 0) * 2.0 + DZeta(0, 1) * DZeta(3, 0) * DZeta(5, 1) * 2.0 + DZeta(0, 1) * DZeta(3, 1) * DZeta(5, 0) * 2.0 + DZeta(0, 0) * DZeta(1, 1) * DZeta(8, 1) * 2.0 + DZeta(0, 1) * DZeta(1, 0) * DZeta(8, 1) * 2.0 + DZeta(0, 1) * DZeta(1, 1) * DZeta(8, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(14, 1) * 2.0;
                DiBj3(14, 14) = DZeta(1, 0) * DZeta(2, 1) * DZeta(3, 2) + DZeta(1, 0) * DZeta(2, 2) * DZeta(3, 1) + DZeta(1, 1) * DZeta(2, 0) * DZeta(3, 2) + DZeta(1, 1) * DZeta(2, 2) * DZeta(3, 0) + DZeta(1, 2) * DZeta(2, 0) * DZeta(3, 1) + DZeta(1, 2) * DZeta(2, 1) * DZeta(3, 0) + DZeta(0, 0) * DZeta(2, 1) * DZeta(6, 2) + DZeta(0, 0) * DZeta(2, 2) * DZeta(6, 1) + DZeta(0, 0) * DZeta(3, 1) * DZeta(5, 2) + DZeta(0, 0) * DZeta(3, 2) * DZeta(5, 1) + DZeta(0, 1) * DZeta(2, 0) * DZeta(6, 2) + DZeta(0, 1) * DZeta(2, 2) * DZeta(6, 0) + DZeta(0, 1) * DZeta(3, 0) * DZeta(5, 2) + DZeta(0, 1) * DZeta(3, 2) * DZeta(5, 0) + DZeta(0, 2) * DZeta(2, 0) * DZeta(6, 1) + DZeta(0, 2) * DZeta(2, 1) * DZeta(6, 0) + DZeta(0, 2) * DZeta(3, 0) * DZeta(5, 1) + DZeta(0, 2) * DZeta(3, 1) * DZeta(5, 0) + DZeta(0, 0) * DZeta(1, 1) * DZeta(8, 2) + DZeta(0, 0) * DZeta(1, 2) * DZeta(8, 1) + DZeta(0, 1) * DZeta(1, 0) * DZeta(8, 2) + DZeta(0, 1) * DZeta(1, 2) * DZeta(8, 0) + DZeta(0, 2) * DZeta(1, 0) * DZeta(8, 1) + DZeta(0, 2) * DZeta(1, 1) * DZeta(8, 0) + DZeta(0, 0) * DZeta(0, 1) * DZeta(14, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(14, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(14, 0);
                DiBj3(14, 15) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(14, 0) + DZeta(1, 0) * DZeta(2, 2) * DZeta(3, 2) * 2.0 + DZeta(1, 2) * DZeta(2, 0) * DZeta(3, 2) * 2.0 + DZeta(1, 2) * DZeta(2, 2) * DZeta(3, 0) * 2.0 + DZeta(0, 0) * DZeta(2, 2) * DZeta(6, 2) * 2.0 + DZeta(0, 0) * DZeta(3, 2) * DZeta(5, 2) * 2.0 + DZeta(0, 2) * DZeta(2, 0) * DZeta(6, 2) * 2.0 + DZeta(0, 2) * DZeta(2, 2) * DZeta(6, 0) * 2.0 + DZeta(0, 2) * DZeta(3, 0) * DZeta(5, 2) * 2.0 + DZeta(0, 2) * DZeta(3, 2) * DZeta(5, 0) * 2.0 + DZeta(0, 0) * DZeta(1, 2) * DZeta(8, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 0) * DZeta(8, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(8, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(14, 2) * 2.0;
                DiBj3(14, 16) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(14, 1) * 3.0 + DZeta(1, 1) * DZeta(2, 1) * DZeta(3, 1) * 6.0 + DZeta(0, 1) * DZeta(2, 1) * DZeta(6, 1) * 6.0 + DZeta(0, 1) * DZeta(3, 1) * DZeta(5, 1) * 6.0 + DZeta(0, 1) * DZeta(1, 1) * DZeta(8, 1) * 6.0;
                DiBj3(14, 17) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(14, 2) + DZeta(1, 1) * DZeta(2, 1) * DZeta(3, 2) * 2.0 + DZeta(1, 1) * DZeta(2, 2) * DZeta(3, 1) * 2.0 + DZeta(1, 2) * DZeta(2, 1) * DZeta(3, 1) * 2.0 + DZeta(0, 1) * DZeta(2, 1) * DZeta(6, 2) * 2.0 + DZeta(0, 1) * DZeta(2, 2) * DZeta(6, 1) * 2.0 + DZeta(0, 1) * DZeta(3, 1) * DZeta(5, 2) * 2.0 + DZeta(0, 1) * DZeta(3, 2) * DZeta(5, 1) * 2.0 + DZeta(0, 2) * DZeta(2, 1) * DZeta(6, 1) * 2.0 + DZeta(0, 2) * DZeta(3, 1) * DZeta(5, 1) * 2.0 + DZeta(0, 1) * DZeta(1, 1) * DZeta(8, 2) * 2.0 + DZeta(0, 1) * DZeta(1, 2) * DZeta(8, 1) * 2.0 + DZeta(0, 2) * DZeta(1, 1) * DZeta(8, 1) * 2.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(14, 1) * 2.0;
                DiBj3(14, 18) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(14, 1) + DZeta(1, 1) * DZeta(2, 2) * DZeta(3, 2) * 2.0 + DZeta(1, 2) * DZeta(2, 1) * DZeta(3, 2) * 2.0 + DZeta(1, 2) * DZeta(2, 2) * DZeta(3, 1) * 2.0 + DZeta(0, 1) * DZeta(2, 2) * DZeta(6, 2) * 2.0 + DZeta(0, 1) * DZeta(3, 2) * DZeta(5, 2) * 2.0 + DZeta(0, 2) * DZeta(2, 1) * DZeta(6, 2) * 2.0 + DZeta(0, 2) * DZeta(2, 2) * DZeta(6, 1) * 2.0 + DZeta(0, 2) * DZeta(3, 1) * DZeta(5, 2) * 2.0 + DZeta(0, 2) * DZeta(3, 2) * DZeta(5, 1) * 2.0 + DZeta(0, 1) * DZeta(1, 2) * DZeta(8, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 1) * DZeta(8, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(8, 1) * 2.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(14, 2) * 2.0;
                DiBj3(14, 19) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(14, 2) * 3.0 + DZeta(1, 2) * DZeta(2, 2) * DZeta(3, 2) * 6.0 + DZeta(0, 2) * DZeta(2, 2) * DZeta(6, 2) * 6.0 + DZeta(0, 2) * DZeta(3, 2) * DZeta(5, 2) * 6.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(8, 2) * 6.0;
                DiBj3(15, 1) = DZeta(15, 0);
                DiBj3(15, 2) = DZeta(15, 1);
                DiBj3(15, 3) = DZeta(15, 2);
                DiBj3(15, 4) = DZeta(3, 0) * DZeta(6, 0) * 4.0 + DZeta(1, 0) * DZeta(9, 0) * 2.0 + DZeta(0, 0) * DZeta(15, 0) * 2.0;
                DiBj3(15, 5) = DZeta(3, 0) * DZeta(6, 1) * 2.0 + DZeta(3, 1) * DZeta(6, 0) * 2.0 + DZeta(1, 0) * DZeta(9, 1) + DZeta(1, 1) * DZeta(9, 0) + DZeta(0, 0) * DZeta(15, 1) + DZeta(0, 1) * DZeta(15, 0);
                DiBj3(15, 6) = DZeta(3, 0) * DZeta(6, 2) * 2.0 + DZeta(3, 2) * DZeta(6, 0) * 2.0 + DZeta(1, 0) * DZeta(9, 2) + DZeta(1, 2) * DZeta(9, 0) + DZeta(0, 0) * DZeta(15, 2) + DZeta(0, 2) * DZeta(15, 0);
                DiBj3(15, 7) = DZeta(3, 1) * DZeta(6, 1) * 4.0 + DZeta(1, 1) * DZeta(9, 1) * 2.0 + DZeta(0, 1) * DZeta(15, 1) * 2.0;
                DiBj3(15, 8) = DZeta(3, 1) * DZeta(6, 2) * 2.0 + DZeta(3, 2) * DZeta(6, 1) * 2.0 + DZeta(1, 1) * DZeta(9, 2) + DZeta(1, 2) * DZeta(9, 1) + DZeta(0, 1) * DZeta(15, 2) + DZeta(0, 2) * DZeta(15, 1);
                DiBj3(15, 9) = DZeta(3, 2) * DZeta(6, 2) * 4.0 + DZeta(1, 2) * DZeta(9, 2) * 2.0 + DZeta(0, 2) * DZeta(15, 2) * 2.0;
                DiBj3(15, 10) = DZeta(1, 0) * (DZeta(3, 0) * DZeta(3, 0)) * 6.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(15, 0) * 3.0 + DZeta(0, 0) * DZeta(3, 0) * DZeta(6, 0) * 1.2E+1 + DZeta(0, 0) * DZeta(1, 0) * DZeta(9, 0) * 6.0;
                DiBj3(15, 11) = DZeta(1, 1) * (DZeta(3, 0) * DZeta(3, 0)) * 2.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(15, 1) + DZeta(1, 0) * DZeta(3, 0) * DZeta(3, 1) * 4.0 + DZeta(0, 0) * DZeta(3, 0) * DZeta(6, 1) * 4.0 + DZeta(0, 0) * DZeta(3, 1) * DZeta(6, 0) * 4.0 + DZeta(0, 1) * DZeta(3, 0) * DZeta(6, 0) * 4.0 + DZeta(0, 0) * DZeta(1, 0) * DZeta(9, 1) * 2.0 + DZeta(0, 0) * DZeta(1, 1) * DZeta(9, 0) * 2.0 + DZeta(0, 1) * DZeta(1, 0) * DZeta(9, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(15, 0) * 2.0;
                DiBj3(15, 12) = DZeta(1, 2) * (DZeta(3, 0) * DZeta(3, 0)) * 2.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(15, 2) + DZeta(1, 0) * DZeta(3, 0) * DZeta(3, 2) * 4.0 + DZeta(0, 0) * DZeta(3, 0) * DZeta(6, 2) * 4.0 + DZeta(0, 0) * DZeta(3, 2) * DZeta(6, 0) * 4.0 + DZeta(0, 2) * DZeta(3, 0) * DZeta(6, 0) * 4.0 + DZeta(0, 0) * DZeta(1, 0) * DZeta(9, 2) * 2.0 + DZeta(0, 0) * DZeta(1, 2) * DZeta(9, 0) * 2.0 + DZeta(0, 2) * DZeta(1, 0) * DZeta(9, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(15, 0) * 2.0;
                DiBj3(15, 13) = DZeta(1, 0) * (DZeta(3, 1) * DZeta(3, 1)) * 2.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(15, 0) + DZeta(1, 1) * DZeta(3, 0) * DZeta(3, 1) * 4.0 + DZeta(0, 0) * DZeta(3, 1) * DZeta(6, 1) * 4.0 + DZeta(0, 1) * DZeta(3, 0) * DZeta(6, 1) * 4.0 + DZeta(0, 1) * DZeta(3, 1) * DZeta(6, 0) * 4.0 + DZeta(0, 0) * DZeta(1, 1) * DZeta(9, 1) * 2.0 + DZeta(0, 1) * DZeta(1, 0) * DZeta(9, 1) * 2.0 + DZeta(0, 1) * DZeta(1, 1) * DZeta(9, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(15, 1) * 2.0;
                DiBj3(15, 14) = DZeta(1, 0) * DZeta(3, 1) * DZeta(3, 2) * 2.0 + DZeta(1, 1) * DZeta(3, 0) * DZeta(3, 2) * 2.0 + DZeta(1, 2) * DZeta(3, 0) * DZeta(3, 1) * 2.0 + DZeta(0, 0) * DZeta(3, 1) * DZeta(6, 2) * 2.0 + DZeta(0, 0) * DZeta(3, 2) * DZeta(6, 1) * 2.0 + DZeta(0, 1) * DZeta(3, 0) * DZeta(6, 2) * 2.0 + DZeta(0, 1) * DZeta(3, 2) * DZeta(6, 0) * 2.0 + DZeta(0, 2) * DZeta(3, 0) * DZeta(6, 1) * 2.0 + DZeta(0, 2) * DZeta(3, 1) * DZeta(6, 0) * 2.0 + DZeta(0, 0) * DZeta(1, 1) * DZeta(9, 2) + DZeta(0, 0) * DZeta(1, 2) * DZeta(9, 1) + DZeta(0, 1) * DZeta(1, 0) * DZeta(9, 2) + DZeta(0, 1) * DZeta(1, 2) * DZeta(9, 0) + DZeta(0, 2) * DZeta(1, 0) * DZeta(9, 1) + DZeta(0, 2) * DZeta(1, 1) * DZeta(9, 0) + DZeta(0, 0) * DZeta(0, 1) * DZeta(15, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(15, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(15, 0);
                DiBj3(15, 15) = DZeta(1, 0) * (DZeta(3, 2) * DZeta(3, 2)) * 2.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(15, 0) + DZeta(1, 2) * DZeta(3, 0) * DZeta(3, 2) * 4.0 + DZeta(0, 0) * DZeta(3, 2) * DZeta(6, 2) * 4.0 + DZeta(0, 2) * DZeta(3, 0) * DZeta(6, 2) * 4.0 + DZeta(0, 2) * DZeta(3, 2) * DZeta(6, 0) * 4.0 + DZeta(0, 0) * DZeta(1, 2) * DZeta(9, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 0) * DZeta(9, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(9, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(15, 2) * 2.0;
                DiBj3(15, 16) = DZeta(1, 1) * (DZeta(3, 1) * DZeta(3, 1)) * 6.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(15, 1) * 3.0 + DZeta(0, 1) * DZeta(3, 1) * DZeta(6, 1) * 1.2E+1 + DZeta(0, 1) * DZeta(1, 1) * DZeta(9, 1) * 6.0;
                DiBj3(15, 17) = DZeta(1, 2) * (DZeta(3, 1) * DZeta(3, 1)) * 2.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(15, 2) + DZeta(1, 1) * DZeta(3, 1) * DZeta(3, 2) * 4.0 + DZeta(0, 1) * DZeta(3, 1) * DZeta(6, 2) * 4.0 + DZeta(0, 1) * DZeta(3, 2) * DZeta(6, 1) * 4.0 + DZeta(0, 2) * DZeta(3, 1) * DZeta(6, 1) * 4.0 + DZeta(0, 1) * DZeta(1, 1) * DZeta(9, 2) * 2.0 + DZeta(0, 1) * DZeta(1, 2) * DZeta(9, 1) * 2.0 + DZeta(0, 2) * DZeta(1, 1) * DZeta(9, 1) * 2.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(15, 1) * 2.0;
                DiBj3(15, 18) = DZeta(1, 1) * (DZeta(3, 2) * DZeta(3, 2)) * 2.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(15, 1) + DZeta(1, 2) * DZeta(3, 1) * DZeta(3, 2) * 4.0 + DZeta(0, 1) * DZeta(3, 2) * DZeta(6, 2) * 4.0 + DZeta(0, 2) * DZeta(3, 1) * DZeta(6, 2) * 4.0 + DZeta(0, 2) * DZeta(3, 2) * DZeta(6, 1) * 4.0 + DZeta(0, 1) * DZeta(1, 2) * DZeta(9, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 1) * DZeta(9, 2) * 2.0 + DZeta(0, 2) * DZeta(1, 2) * DZeta(9, 1) * 2.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(15, 2) * 2.0;
                DiBj3(15, 19) = DZeta(1, 2) * (DZeta(3, 2) * DZeta(3, 2)) * 6.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(15, 2) * 3.0 + DZeta(0, 2) * DZeta(3, 2) * DZeta(6, 2) * 1.2E+1 + DZeta(0, 2) * DZeta(1, 2) * DZeta(9, 2) * 6.0;
                DiBj3(16, 1) = DZeta(16, 0);
                DiBj3(16, 2) = DZeta(16, 1);
                DiBj3(16, 3) = DZeta(16, 2);
                DiBj3(16, 4) = DZeta(2, 0) * DZeta(7, 0) * 6.0 + DZeta(0, 0) * DZeta(16, 0) * 2.0;
                DiBj3(16, 5) = DZeta(2, 0) * DZeta(7, 1) * 3.0 + DZeta(2, 1) * DZeta(7, 0) * 3.0 + DZeta(0, 0) * DZeta(16, 1) + DZeta(0, 1) * DZeta(16, 0);
                DiBj3(16, 6) = DZeta(2, 0) * DZeta(7, 2) * 3.0 + DZeta(2, 2) * DZeta(7, 0) * 3.0 + DZeta(0, 0) * DZeta(16, 2) + DZeta(0, 2) * DZeta(16, 0);
                DiBj3(16, 7) = DZeta(2, 1) * DZeta(7, 1) * 6.0 + DZeta(0, 1) * DZeta(16, 1) * 2.0;
                DiBj3(16, 8) = DZeta(2, 1) * DZeta(7, 2) * 3.0 + DZeta(2, 2) * DZeta(7, 1) * 3.0 + DZeta(0, 1) * DZeta(16, 2) + DZeta(0, 2) * DZeta(16, 1);
                DiBj3(16, 9) = DZeta(2, 2) * DZeta(7, 2) * 6.0 + DZeta(0, 2) * DZeta(16, 2) * 2.0;
                DiBj3(16, 10) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(16, 0) * 3.0 + (DZeta(2, 0) * DZeta(2, 0) * DZeta(2, 0)) * 6.0 + DZeta(0, 0) * DZeta(2, 0) * DZeta(7, 0) * 1.8E+1;
                DiBj3(16, 11) = (DZeta(2, 0) * DZeta(2, 0)) * DZeta(2, 1) * 6.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(16, 1) + DZeta(0, 0) * DZeta(2, 0) * DZeta(7, 1) * 6.0 + DZeta(0, 0) * DZeta(2, 1) * DZeta(7, 0) * 6.0 + DZeta(0, 1) * DZeta(2, 0) * DZeta(7, 0) * 6.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(16, 0) * 2.0;
                DiBj3(16, 12) = (DZeta(2, 0) * DZeta(2, 0)) * DZeta(2, 2) * 6.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(16, 2) + DZeta(0, 0) * DZeta(2, 0) * DZeta(7, 2) * 6.0 + DZeta(0, 0) * DZeta(2, 2) * DZeta(7, 0) * 6.0 + DZeta(0, 2) * DZeta(2, 0) * DZeta(7, 0) * 6.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(16, 0) * 2.0;
                DiBj3(16, 13) = DZeta(2, 0) * (DZeta(2, 1) * DZeta(2, 1)) * 6.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(16, 0) + DZeta(0, 0) * DZeta(2, 1) * DZeta(7, 1) * 6.0 + DZeta(0, 1) * DZeta(2, 0) * DZeta(7, 1) * 6.0 + DZeta(0, 1) * DZeta(2, 1) * DZeta(7, 0) * 6.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(16, 1) * 2.0;
                DiBj3(16, 14) = DZeta(2, 0) * DZeta(2, 1) * DZeta(2, 2) * 6.0 + DZeta(0, 0) * DZeta(2, 1) * DZeta(7, 2) * 3.0 + DZeta(0, 0) * DZeta(2, 2) * DZeta(7, 1) * 3.0 + DZeta(0, 1) * DZeta(2, 0) * DZeta(7, 2) * 3.0 + DZeta(0, 1) * DZeta(2, 2) * DZeta(7, 0) * 3.0 + DZeta(0, 2) * DZeta(2, 0) * DZeta(7, 1) * 3.0 + DZeta(0, 2) * DZeta(2, 1) * DZeta(7, 0) * 3.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(16, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(16, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(16, 0);
                DiBj3(16, 15) = DZeta(2, 0) * (DZeta(2, 2) * DZeta(2, 2)) * 6.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(16, 0) + DZeta(0, 0) * DZeta(2, 2) * DZeta(7, 2) * 6.0 + DZeta(0, 2) * DZeta(2, 0) * DZeta(7, 2) * 6.0 + DZeta(0, 2) * DZeta(2, 2) * DZeta(7, 0) * 6.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(16, 2) * 2.0;
                DiBj3(16, 16) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(16, 1) * 3.0 + (DZeta(2, 1) * DZeta(2, 1) * DZeta(2, 1)) * 6.0 + DZeta(0, 1) * DZeta(2, 1) * DZeta(7, 1) * 1.8E+1;
                DiBj3(16, 17) = (DZeta(2, 1) * DZeta(2, 1)) * DZeta(2, 2) * 6.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(16, 2) + DZeta(0, 1) * DZeta(2, 1) * DZeta(7, 2) * 6.0 + DZeta(0, 1) * DZeta(2, 2) * DZeta(7, 1) * 6.0 + DZeta(0, 2) * DZeta(2, 1) * DZeta(7, 1) * 6.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(16, 1) * 2.0;
                DiBj3(16, 18) = DZeta(2, 1) * (DZeta(2, 2) * DZeta(2, 2)) * 6.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(16, 1) + DZeta(0, 1) * DZeta(2, 2) * DZeta(7, 2) * 6.0 + DZeta(0, 2) * DZeta(2, 1) * DZeta(7, 2) * 6.0 + DZeta(0, 2) * DZeta(2, 2) * DZeta(7, 1) * 6.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(16, 2) * 2.0;
                DiBj3(16, 19) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(16, 2) * 3.0 + (DZeta(2, 2) * DZeta(2, 2) * DZeta(2, 2)) * 6.0 + DZeta(0, 2) * DZeta(2, 2) * DZeta(7, 2) * 1.8E+1;
                DiBj3(17, 1) = DZeta(17, 0);
                DiBj3(17, 2) = DZeta(17, 1);
                DiBj3(17, 3) = DZeta(17, 2);
                DiBj3(17, 4) = DZeta(2, 0) * DZeta(8, 0) * 4.0 + DZeta(3, 0) * DZeta(7, 0) * 2.0 + DZeta(0, 0) * DZeta(17, 0) * 2.0;
                DiBj3(17, 5) = DZeta(2, 0) * DZeta(8, 1) * 2.0 + DZeta(2, 1) * DZeta(8, 0) * 2.0 + DZeta(3, 0) * DZeta(7, 1) + DZeta(3, 1) * DZeta(7, 0) + DZeta(0, 0) * DZeta(17, 1) + DZeta(0, 1) * DZeta(17, 0);
                DiBj3(17, 6) = DZeta(2, 0) * DZeta(8, 2) * 2.0 + DZeta(2, 2) * DZeta(8, 0) * 2.0 + DZeta(3, 0) * DZeta(7, 2) + DZeta(3, 2) * DZeta(7, 0) + DZeta(0, 0) * DZeta(17, 2) + DZeta(0, 2) * DZeta(17, 0);
                DiBj3(17, 7) = DZeta(2, 1) * DZeta(8, 1) * 4.0 + DZeta(3, 1) * DZeta(7, 1) * 2.0 + DZeta(0, 1) * DZeta(17, 1) * 2.0;
                DiBj3(17, 8) = DZeta(2, 1) * DZeta(8, 2) * 2.0 + DZeta(2, 2) * DZeta(8, 1) * 2.0 + DZeta(3, 1) * DZeta(7, 2) + DZeta(3, 2) * DZeta(7, 1) + DZeta(0, 1) * DZeta(17, 2) + DZeta(0, 2) * DZeta(17, 1);
                DiBj3(17, 9) = DZeta(2, 2) * DZeta(8, 2) * 4.0 + DZeta(3, 2) * DZeta(7, 2) * 2.0 + DZeta(0, 2) * DZeta(17, 2) * 2.0;
                DiBj3(17, 10) = (DZeta(2, 0) * DZeta(2, 0)) * DZeta(3, 0) * 6.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(17, 0) * 3.0 + DZeta(0, 0) * DZeta(2, 0) * DZeta(8, 0) * 1.2E+1 + DZeta(0, 0) * DZeta(3, 0) * DZeta(7, 0) * 6.0;
                DiBj3(17, 11) = (DZeta(2, 0) * DZeta(2, 0)) * DZeta(3, 1) * 2.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(17, 1) + DZeta(2, 0) * DZeta(2, 1) * DZeta(3, 0) * 4.0 + DZeta(0, 0) * DZeta(2, 0) * DZeta(8, 1) * 4.0 + DZeta(0, 0) * DZeta(2, 1) * DZeta(8, 0) * 4.0 + DZeta(0, 0) * DZeta(3, 0) * DZeta(7, 1) * 2.0 + DZeta(0, 0) * DZeta(3, 1) * DZeta(7, 0) * 2.0 + DZeta(0, 1) * DZeta(2, 0) * DZeta(8, 0) * 4.0 + DZeta(0, 1) * DZeta(3, 0) * DZeta(7, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(17, 0) * 2.0;
                DiBj3(17, 12) = (DZeta(2, 0) * DZeta(2, 0)) * DZeta(3, 2) * 2.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(17, 2) + DZeta(2, 0) * DZeta(2, 2) * DZeta(3, 0) * 4.0 + DZeta(0, 0) * DZeta(2, 0) * DZeta(8, 2) * 4.0 + DZeta(0, 0) * DZeta(2, 2) * DZeta(8, 0) * 4.0 + DZeta(0, 0) * DZeta(3, 0) * DZeta(7, 2) * 2.0 + DZeta(0, 0) * DZeta(3, 2) * DZeta(7, 0) * 2.0 + DZeta(0, 2) * DZeta(2, 0) * DZeta(8, 0) * 4.0 + DZeta(0, 2) * DZeta(3, 0) * DZeta(7, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(17, 0) * 2.0;
                DiBj3(17, 13) = (DZeta(2, 1) * DZeta(2, 1)) * DZeta(3, 0) * 2.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(17, 0) + DZeta(2, 0) * DZeta(2, 1) * DZeta(3, 1) * 4.0 + DZeta(0, 0) * DZeta(2, 1) * DZeta(8, 1) * 4.0 + DZeta(0, 0) * DZeta(3, 1) * DZeta(7, 1) * 2.0 + DZeta(0, 1) * DZeta(2, 0) * DZeta(8, 1) * 4.0 + DZeta(0, 1) * DZeta(2, 1) * DZeta(8, 0) * 4.0 + DZeta(0, 1) * DZeta(3, 0) * DZeta(7, 1) * 2.0 + DZeta(0, 1) * DZeta(3, 1) * DZeta(7, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(17, 1) * 2.0;
                DiBj3(17, 14) = DZeta(2, 0) * DZeta(2, 1) * DZeta(3, 2) * 2.0 + DZeta(2, 0) * DZeta(2, 2) * DZeta(3, 1) * 2.0 + DZeta(2, 1) * DZeta(2, 2) * DZeta(3, 0) * 2.0 + DZeta(0, 0) * DZeta(2, 1) * DZeta(8, 2) * 2.0 + DZeta(0, 0) * DZeta(2, 2) * DZeta(8, 1) * 2.0 + DZeta(0, 0) * DZeta(3, 1) * DZeta(7, 2) + DZeta(0, 0) * DZeta(3, 2) * DZeta(7, 1) + DZeta(0, 1) * DZeta(2, 0) * DZeta(8, 2) * 2.0 + DZeta(0, 1) * DZeta(2, 2) * DZeta(8, 0) * 2.0 + DZeta(0, 1) * DZeta(3, 0) * DZeta(7, 2) + DZeta(0, 1) * DZeta(3, 2) * DZeta(7, 0) + DZeta(0, 2) * DZeta(2, 0) * DZeta(8, 1) * 2.0 + DZeta(0, 2) * DZeta(2, 1) * DZeta(8, 0) * 2.0 + DZeta(0, 2) * DZeta(3, 0) * DZeta(7, 1) + DZeta(0, 2) * DZeta(3, 1) * DZeta(7, 0) + DZeta(0, 0) * DZeta(0, 1) * DZeta(17, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(17, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(17, 0);
                DiBj3(17, 15) = (DZeta(2, 2) * DZeta(2, 2)) * DZeta(3, 0) * 2.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(17, 0) + DZeta(2, 0) * DZeta(2, 2) * DZeta(3, 2) * 4.0 + DZeta(0, 0) * DZeta(2, 2) * DZeta(8, 2) * 4.0 + DZeta(0, 0) * DZeta(3, 2) * DZeta(7, 2) * 2.0 + DZeta(0, 2) * DZeta(2, 0) * DZeta(8, 2) * 4.0 + DZeta(0, 2) * DZeta(2, 2) * DZeta(8, 0) * 4.0 + DZeta(0, 2) * DZeta(3, 0) * DZeta(7, 2) * 2.0 + DZeta(0, 2) * DZeta(3, 2) * DZeta(7, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(17, 2) * 2.0;
                DiBj3(17, 16) = (DZeta(2, 1) * DZeta(2, 1)) * DZeta(3, 1) * 6.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(17, 1) * 3.0 + DZeta(0, 1) * DZeta(2, 1) * DZeta(8, 1) * 1.2E+1 + DZeta(0, 1) * DZeta(3, 1) * DZeta(7, 1) * 6.0;
                DiBj3(17, 17) = (DZeta(2, 1) * DZeta(2, 1)) * DZeta(3, 2) * 2.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(17, 2) + DZeta(2, 1) * DZeta(2, 2) * DZeta(3, 1) * 4.0 + DZeta(0, 1) * DZeta(2, 1) * DZeta(8, 2) * 4.0 + DZeta(0, 1) * DZeta(2, 2) * DZeta(8, 1) * 4.0 + DZeta(0, 1) * DZeta(3, 1) * DZeta(7, 2) * 2.0 + DZeta(0, 1) * DZeta(3, 2) * DZeta(7, 1) * 2.0 + DZeta(0, 2) * DZeta(2, 1) * DZeta(8, 1) * 4.0 + DZeta(0, 2) * DZeta(3, 1) * DZeta(7, 1) * 2.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(17, 1) * 2.0;
                DiBj3(17, 18) = (DZeta(2, 2) * DZeta(2, 2)) * DZeta(3, 1) * 2.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(17, 1) + DZeta(2, 1) * DZeta(2, 2) * DZeta(3, 2) * 4.0 + DZeta(0, 1) * DZeta(2, 2) * DZeta(8, 2) * 4.0 + DZeta(0, 1) * DZeta(3, 2) * DZeta(7, 2) * 2.0 + DZeta(0, 2) * DZeta(2, 1) * DZeta(8, 2) * 4.0 + DZeta(0, 2) * DZeta(2, 2) * DZeta(8, 1) * 4.0 + DZeta(0, 2) * DZeta(3, 1) * DZeta(7, 2) * 2.0 + DZeta(0, 2) * DZeta(3, 2) * DZeta(7, 1) * 2.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(17, 2) * 2.0;
                DiBj3(17, 19) = (DZeta(2, 2) * DZeta(2, 2)) * DZeta(3, 2) * 6.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(17, 2) * 3.0 + DZeta(0, 2) * DZeta(2, 2) * DZeta(8, 2) * 1.2E+1 + DZeta(0, 2) * DZeta(3, 2) * DZeta(7, 2) * 6.0;
                DiBj3(18, 1) = DZeta(18, 0);
                DiBj3(18, 2) = DZeta(18, 1);
                DiBj3(18, 3) = DZeta(18, 2);
                DiBj3(18, 4) = DZeta(2, 0) * DZeta(9, 0) * 2.0 + DZeta(3, 0) * DZeta(8, 0) * 4.0 + DZeta(0, 0) * DZeta(18, 0) * 2.0;
                DiBj3(18, 5) = DZeta(2, 0) * DZeta(9, 1) + DZeta(2, 1) * DZeta(9, 0) + DZeta(3, 0) * DZeta(8, 1) * 2.0 + DZeta(3, 1) * DZeta(8, 0) * 2.0 + DZeta(0, 0) * DZeta(18, 1) + DZeta(0, 1) * DZeta(18, 0);
                DiBj3(18, 6) = DZeta(2, 0) * DZeta(9, 2) + DZeta(2, 2) * DZeta(9, 0) + DZeta(3, 0) * DZeta(8, 2) * 2.0 + DZeta(3, 2) * DZeta(8, 0) * 2.0 + DZeta(0, 0) * DZeta(18, 2) + DZeta(0, 2) * DZeta(18, 0);
                DiBj3(18, 7) = DZeta(2, 1) * DZeta(9, 1) * 2.0 + DZeta(3, 1) * DZeta(8, 1) * 4.0 + DZeta(0, 1) * DZeta(18, 1) * 2.0;
                DiBj3(18, 8) = DZeta(2, 1) * DZeta(9, 2) + DZeta(2, 2) * DZeta(9, 1) + DZeta(3, 1) * DZeta(8, 2) * 2.0 + DZeta(3, 2) * DZeta(8, 1) * 2.0 + DZeta(0, 1) * DZeta(18, 2) + DZeta(0, 2) * DZeta(18, 1);
                DiBj3(18, 9) = DZeta(2, 2) * DZeta(9, 2) * 2.0 + DZeta(3, 2) * DZeta(8, 2) * 4.0 + DZeta(0, 2) * DZeta(18, 2) * 2.0;
                DiBj3(18, 10) = DZeta(2, 0) * (DZeta(3, 0) * DZeta(3, 0)) * 6.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(18, 0) * 3.0 + DZeta(0, 0) * DZeta(2, 0) * DZeta(9, 0) * 6.0 + DZeta(0, 0) * DZeta(3, 0) * DZeta(8, 0) * 1.2E+1;
                DiBj3(18, 11) = DZeta(2, 1) * (DZeta(3, 0) * DZeta(3, 0)) * 2.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(18, 1) + DZeta(2, 0) * DZeta(3, 0) * DZeta(3, 1) * 4.0 + DZeta(0, 0) * DZeta(2, 0) * DZeta(9, 1) * 2.0 + DZeta(0, 0) * DZeta(2, 1) * DZeta(9, 0) * 2.0 + DZeta(0, 0) * DZeta(3, 0) * DZeta(8, 1) * 4.0 + DZeta(0, 0) * DZeta(3, 1) * DZeta(8, 0) * 4.0 + DZeta(0, 1) * DZeta(2, 0) * DZeta(9, 0) * 2.0 + DZeta(0, 1) * DZeta(3, 0) * DZeta(8, 0) * 4.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(18, 0) * 2.0;
                DiBj3(18, 12) = DZeta(2, 2) * (DZeta(3, 0) * DZeta(3, 0)) * 2.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(18, 2) + DZeta(2, 0) * DZeta(3, 0) * DZeta(3, 2) * 4.0 + DZeta(0, 0) * DZeta(2, 0) * DZeta(9, 2) * 2.0 + DZeta(0, 0) * DZeta(2, 2) * DZeta(9, 0) * 2.0 + DZeta(0, 0) * DZeta(3, 0) * DZeta(8, 2) * 4.0 + DZeta(0, 0) * DZeta(3, 2) * DZeta(8, 0) * 4.0 + DZeta(0, 2) * DZeta(2, 0) * DZeta(9, 0) * 2.0 + DZeta(0, 2) * DZeta(3, 0) * DZeta(8, 0) * 4.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(18, 0) * 2.0;
                DiBj3(18, 13) = DZeta(2, 0) * (DZeta(3, 1) * DZeta(3, 1)) * 2.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(18, 0) + DZeta(2, 1) * DZeta(3, 0) * DZeta(3, 1) * 4.0 + DZeta(0, 0) * DZeta(2, 1) * DZeta(9, 1) * 2.0 + DZeta(0, 0) * DZeta(3, 1) * DZeta(8, 1) * 4.0 + DZeta(0, 1) * DZeta(2, 0) * DZeta(9, 1) * 2.0 + DZeta(0, 1) * DZeta(2, 1) * DZeta(9, 0) * 2.0 + DZeta(0, 1) * DZeta(3, 0) * DZeta(8, 1) * 4.0 + DZeta(0, 1) * DZeta(3, 1) * DZeta(8, 0) * 4.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(18, 1) * 2.0;
                DiBj3(18, 14) = DZeta(2, 0) * DZeta(3, 1) * DZeta(3, 2) * 2.0 + DZeta(2, 1) * DZeta(3, 0) * DZeta(3, 2) * 2.0 + DZeta(2, 2) * DZeta(3, 0) * DZeta(3, 1) * 2.0 + DZeta(0, 0) * DZeta(2, 1) * DZeta(9, 2) + DZeta(0, 0) * DZeta(2, 2) * DZeta(9, 1) + DZeta(0, 0) * DZeta(3, 1) * DZeta(8, 2) * 2.0 + DZeta(0, 0) * DZeta(3, 2) * DZeta(8, 1) * 2.0 + DZeta(0, 1) * DZeta(2, 0) * DZeta(9, 2) + DZeta(0, 1) * DZeta(2, 2) * DZeta(9, 0) + DZeta(0, 1) * DZeta(3, 0) * DZeta(8, 2) * 2.0 + DZeta(0, 1) * DZeta(3, 2) * DZeta(8, 0) * 2.0 + DZeta(0, 2) * DZeta(2, 0) * DZeta(9, 1) + DZeta(0, 2) * DZeta(2, 1) * DZeta(9, 0) + DZeta(0, 2) * DZeta(3, 0) * DZeta(8, 1) * 2.0 + DZeta(0, 2) * DZeta(3, 1) * DZeta(8, 0) * 2.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(18, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(18, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(18, 0);
                DiBj3(18, 15) = DZeta(2, 0) * (DZeta(3, 2) * DZeta(3, 2)) * 2.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(18, 0) + DZeta(2, 2) * DZeta(3, 0) * DZeta(3, 2) * 4.0 + DZeta(0, 0) * DZeta(2, 2) * DZeta(9, 2) * 2.0 + DZeta(0, 0) * DZeta(3, 2) * DZeta(8, 2) * 4.0 + DZeta(0, 2) * DZeta(2, 0) * DZeta(9, 2) * 2.0 + DZeta(0, 2) * DZeta(2, 2) * DZeta(9, 0) * 2.0 + DZeta(0, 2) * DZeta(3, 0) * DZeta(8, 2) * 4.0 + DZeta(0, 2) * DZeta(3, 2) * DZeta(8, 0) * 4.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(18, 2) * 2.0;
                DiBj3(18, 16) = DZeta(2, 1) * (DZeta(3, 1) * DZeta(3, 1)) * 6.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(18, 1) * 3.0 + DZeta(0, 1) * DZeta(2, 1) * DZeta(9, 1) * 6.0 + DZeta(0, 1) * DZeta(3, 1) * DZeta(8, 1) * 1.2E+1;
                DiBj3(18, 17) = DZeta(2, 2) * (DZeta(3, 1) * DZeta(3, 1)) * 2.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(18, 2) + DZeta(2, 1) * DZeta(3, 1) * DZeta(3, 2) * 4.0 + DZeta(0, 1) * DZeta(2, 1) * DZeta(9, 2) * 2.0 + DZeta(0, 1) * DZeta(2, 2) * DZeta(9, 1) * 2.0 + DZeta(0, 1) * DZeta(3, 1) * DZeta(8, 2) * 4.0 + DZeta(0, 1) * DZeta(3, 2) * DZeta(8, 1) * 4.0 + DZeta(0, 2) * DZeta(2, 1) * DZeta(9, 1) * 2.0 + DZeta(0, 2) * DZeta(3, 1) * DZeta(8, 1) * 4.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(18, 1) * 2.0;
                DiBj3(18, 18) = DZeta(2, 1) * (DZeta(3, 2) * DZeta(3, 2)) * 2.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(18, 1) + DZeta(2, 2) * DZeta(3, 1) * DZeta(3, 2) * 4.0 + DZeta(0, 1) * DZeta(2, 2) * DZeta(9, 2) * 2.0 + DZeta(0, 1) * DZeta(3, 2) * DZeta(8, 2) * 4.0 + DZeta(0, 2) * DZeta(2, 1) * DZeta(9, 2) * 2.0 + DZeta(0, 2) * DZeta(2, 2) * DZeta(9, 1) * 2.0 + DZeta(0, 2) * DZeta(3, 1) * DZeta(8, 2) * 4.0 + DZeta(0, 2) * DZeta(3, 2) * DZeta(8, 1) * 4.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(18, 2) * 2.0;
                DiBj3(18, 19) = DZeta(2, 2) * (DZeta(3, 2) * DZeta(3, 2)) * 6.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(18, 2) * 3.0 + DZeta(0, 2) * DZeta(2, 2) * DZeta(9, 2) * 6.0 + DZeta(0, 2) * DZeta(3, 2) * DZeta(8, 2) * 1.2E+1;
                DiBj3(19, 1) = DZeta(19, 0);
                DiBj3(19, 2) = DZeta(19, 1);
                DiBj3(19, 3) = DZeta(19, 2);
                DiBj3(19, 4) = DZeta(3, 0) * DZeta(9, 0) * 6.0 + DZeta(0, 0) * DZeta(19, 0) * 2.0;
                DiBj3(19, 5) = DZeta(3, 0) * DZeta(9, 1) * 3.0 + DZeta(3, 1) * DZeta(9, 0) * 3.0 + DZeta(0, 0) * DZeta(19, 1) + DZeta(0, 1) * DZeta(19, 0);
                DiBj3(19, 6) = DZeta(3, 0) * DZeta(9, 2) * 3.0 + DZeta(3, 2) * DZeta(9, 0) * 3.0 + DZeta(0, 0) * DZeta(19, 2) + DZeta(0, 2) * DZeta(19, 0);
                DiBj3(19, 7) = DZeta(3, 1) * DZeta(9, 1) * 6.0 + DZeta(0, 1) * DZeta(19, 1) * 2.0;
                DiBj3(19, 8) = DZeta(3, 1) * DZeta(9, 2) * 3.0 + DZeta(3, 2) * DZeta(9, 1) * 3.0 + DZeta(0, 1) * DZeta(19, 2) + DZeta(0, 2) * DZeta(19, 1);
                DiBj3(19, 9) = DZeta(3, 2) * DZeta(9, 2) * 6.0 + DZeta(0, 2) * DZeta(19, 2) * 2.0;
                DiBj3(19, 10) = (DZeta(0, 0) * DZeta(0, 0)) * DZeta(19, 0) * 3.0 + (DZeta(3, 0) * DZeta(3, 0) * DZeta(3, 0)) * 6.0 + DZeta(0, 0) * DZeta(3, 0) * DZeta(9, 0) * 1.8E+1;
                DiBj3(19, 11) = (DZeta(3, 0) * DZeta(3, 0)) * DZeta(3, 1) * 6.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(19, 1) + DZeta(0, 0) * DZeta(3, 0) * DZeta(9, 1) * 6.0 + DZeta(0, 0) * DZeta(3, 1) * DZeta(9, 0) * 6.0 + DZeta(0, 1) * DZeta(3, 0) * DZeta(9, 0) * 6.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(19, 0) * 2.0;
                DiBj3(19, 12) = (DZeta(3, 0) * DZeta(3, 0)) * DZeta(3, 2) * 6.0 + (DZeta(0, 0) * DZeta(0, 0)) * DZeta(19, 2) + DZeta(0, 0) * DZeta(3, 0) * DZeta(9, 2) * 6.0 + DZeta(0, 0) * DZeta(3, 2) * DZeta(9, 0) * 6.0 + DZeta(0, 2) * DZeta(3, 0) * DZeta(9, 0) * 6.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(19, 0) * 2.0;
                DiBj3(19, 13) = DZeta(3, 0) * (DZeta(3, 1) * DZeta(3, 1)) * 6.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(19, 0) + DZeta(0, 0) * DZeta(3, 1) * DZeta(9, 1) * 6.0 + DZeta(0, 1) * DZeta(3, 0) * DZeta(9, 1) * 6.0 + DZeta(0, 1) * DZeta(3, 1) * DZeta(9, 0) * 6.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(19, 1) * 2.0;
                DiBj3(19, 14) = DZeta(3, 0) * DZeta(3, 1) * DZeta(3, 2) * 6.0 + DZeta(0, 0) * DZeta(3, 1) * DZeta(9, 2) * 3.0 + DZeta(0, 0) * DZeta(3, 2) * DZeta(9, 1) * 3.0 + DZeta(0, 1) * DZeta(3, 0) * DZeta(9, 2) * 3.0 + DZeta(0, 1) * DZeta(3, 2) * DZeta(9, 0) * 3.0 + DZeta(0, 2) * DZeta(3, 0) * DZeta(9, 1) * 3.0 + DZeta(0, 2) * DZeta(3, 1) * DZeta(9, 0) * 3.0 + DZeta(0, 0) * DZeta(0, 1) * DZeta(19, 2) + DZeta(0, 0) * DZeta(0, 2) * DZeta(19, 1) + DZeta(0, 1) * DZeta(0, 2) * DZeta(19, 0);
                DiBj3(19, 15) = DZeta(3, 0) * (DZeta(3, 2) * DZeta(3, 2)) * 6.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(19, 0) + DZeta(0, 0) * DZeta(3, 2) * DZeta(9, 2) * 6.0 + DZeta(0, 2) * DZeta(3, 0) * DZeta(9, 2) * 6.0 + DZeta(0, 2) * DZeta(3, 2) * DZeta(9, 0) * 6.0 + DZeta(0, 0) * DZeta(0, 2) * DZeta(19, 2) * 2.0;
                DiBj3(19, 16) = (DZeta(0, 1) * DZeta(0, 1)) * DZeta(19, 1) * 3.0 + (DZeta(3, 1) * DZeta(3, 1) * DZeta(3, 1)) * 6.0 + DZeta(0, 1) * DZeta(3, 1) * DZeta(9, 1) * 1.8E+1;
                DiBj3(19, 17) = (DZeta(3, 1) * DZeta(3, 1)) * DZeta(3, 2) * 6.0 + (DZeta(0, 1) * DZeta(0, 1)) * DZeta(19, 2) + DZeta(0, 1) * DZeta(3, 1) * DZeta(9, 2) * 6.0 + DZeta(0, 1) * DZeta(3, 2) * DZeta(9, 1) * 6.0 + DZeta(0, 2) * DZeta(3, 1) * DZeta(9, 1) * 6.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(19, 1) * 2.0;
                DiBj3(19, 18) = DZeta(3, 1) * (DZeta(3, 2) * DZeta(3, 2)) * 6.0 + (DZeta(0, 2) * DZeta(0, 2)) * DZeta(19, 1) + DZeta(0, 1) * DZeta(3, 2) * DZeta(9, 2) * 6.0 + DZeta(0, 2) * DZeta(3, 1) * DZeta(9, 2) * 6.0 + DZeta(0, 2) * DZeta(3, 2) * DZeta(9, 1) * 6.0 + DZeta(0, 1) * DZeta(0, 2) * DZeta(19, 2) * 2.0;
                DiBj3(19, 19) = (DZeta(0, 2) * DZeta(0, 2)) * DZeta(19, 2) * 3.0 + (DZeta(3, 2) * DZeta(3, 2) * DZeta(3, 2)) * 6.0 + DZeta(0, 2) * DZeta(3, 2) * DZeta(9, 2) * 1.8E+1;

                Eigen::MatrixXd DiBj2 = DiBj3({0, 1, 2, 4, 5, 7, 10, 11, 13, 16}, {0, 1, 2, 4, 5, 7, 10, 11, 13, 16});
                // std::cout << DiBj << std::endl;
                // std::cout << D1zeta1 << D1zeta0 << std::endl;
                DiBj = DiBj2.topLeftCorner(DiBj.rows(), DiBj.cols());
                // std::cout << "\nNEW\n\n"
                //           << DiBj << std::endl;
                // abort();
            }
#endif

            // here DiBj's direvatives are dBdxi-s, where xi is defined by xi = invPJacobi * (x - xc)
            if (!options.disableLocalCoord2GlobalDiffs)
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

            if (Weights.size() == 10) // ! warning : NORM_FUNCTIONAL only implemented for 4th order
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
                Eigen::Vector2d fNorm = iGauss >= 0 ? faceNorms[iFace][iGauss]({0, 1}).stableNormalized() * length
                                                    : faceNormCenter[iFace]({0, 1}).stableNormalized() * length;

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
            else if (Weights.size() == 6) // ! warning : NORM_FUNCTIONAL only implemented for 4th order
            {
                // std::cout << "2d 3rd order" << std::endl;
                // std::abort();
                const real epsR = 1e-50;
                real w2r, length;
                if (std::fabs(Weights[1]) >= epsR || std::fabs(Weights[2]) >= epsR)
                {
                    w2r = std::fabs(Weights[1]) >= std::fabs(Weights[2]) ? Weights[3] / (Weights[1] * Weights[1]) : Weights[5] / (Weights[2] * Weights[2]);
                    length = std::sqrt(Weights[1] * Weights[1] + Weights[2] * Weights[2]);
                }
                else
                {
                    w2r = 0.0;
                    length = 0;
                }
                Eigen::Vector2d fNorm = iGauss >= 0 ? faceNorms[iFace][iGauss]({0, 1}).stableNormalized() * length
                                                    : faceNormCenter[iFace]({0, 1}).stableNormalized() * length;

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

        void initBaseDiffCache()
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
                            Eigen::MatrixXd ZeroDs = DiBj({0}, Eigen::seq(1, Eigen::last));
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

                        cellDiBjCenterBatchElem.m(0)(Eigen::seq(1, Eigen::last), Eigen::seq(1, Eigen::last)) *= LOrth.transpose();
                        for (int ig = 0; ig < eCell.getNInt(); ig++)
                        {
                            cellDiBjGaussBatchElem.m(ig)(Eigen::seq(1, Eigen::last), Eigen::seq(1, Eigen::last)) *= LOrth.transpose();
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
                    else if (faceAtr.iPhy == BoundaryType::Farfield || faceAtr.iPhy == BoundaryType::Special_DMRFar)
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
                        GW = 1. / std::sqrt(D);
                        break;
                    case Setting::WeightSchemeGeom::S:
                        GW = 1. / std::sqrt(S);
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
                    else if (faceAttribute.iPhy == BoundaryType::Wall ||
                             faceAttribute.iPhy == BoundaryType::Farfield ||
                             faceAttribute.iPhy == BoundaryType::Wall_Euler ||
                             faceAttribute.iPhy == BoundaryType::Wall_NoSlip ||
                             faceAttribute.iPhy == BoundaryType::Special_DMRFar)
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
                                 faceAttribute.iPhy == BoundaryType::Special_DMRFar)
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
                        return vsize * (cellRecAtrLocal[i][0].NDOF - 1); // !!note - 1!!
                    },
                    nCellDist),
                mpi);
            uR.CreateGhostCopyComm(mesh->cellAtrLocal);
            uR.InitPersistentPullClean();
        }

        void BuildRec(ArrayRecV &uR, index nvars)
        {
            index nCellDist = mesh->cell2nodeLocal.dist->size();

            uR.resize(
                nCellDist, mpi, [&](index i)
                { return cellRecAtrLocal[i][0].NDOF - 1; },
                nvars);
            uR.CreateGhostCopyComm(mesh->cellAtrLocal);
            uR.InitPersistentPullClean();
        }

        void initUcurve()
        {
            index nCellDist = mesh->cell2nodeLocal.dist->size();

            uCurve.dist = std::make_shared<decltype(uCurve.dist)::element_type>(
                decltype(uCurve.dist)::element_type::tContext(
                    [&](index i) -> rowsize
                    {
                        switch (setting.curvilinearOrder)
                        {
                        case 1:
                            return 2 * (3 - 1); // !!note - 1!! !!note the 2 sized row!!
                            break;
                        case 2:
                            return 2 * (6 - 1);
                            break;
                        case 3:
                            return 2 * (10 - 1);
                            break;
                        default:
                            assert(false);
                            return -1;
                        }
                    },
                    nCellDist),
                mpi);
            uCurve.CreateGhostCopyComm(mesh->cellAtrLocal);
            uCurve.InitPersistentPullClean();
            forEachInArrayPair(
                *uCurve.pair,
                [&](decltype(uCurve.dist)::element_type::tComponent &e, index iCell)
                {
                    e.m().setZero();
                    e.m()(0, 0) = 1;
                    e.m()(1, 1) = 1;

                    // e.m()(1, 0) = 1.1;
                    // e.m()(0, 1) = -1.1;
                    // e.m()(1, 1) = -1000.1;
                });
        }

        void SOR_InitRedBlack()
        {
            const index nCell = mesh->cell2nodeLocal.dist->size();
            SOR_iScan2iCell = std::make_shared<std::vector<index>>(nCell);
            SOR_iCell2iScan = std::make_shared<std::vector<index>>(nCell);
            if (!(setting.SOR_RedBlack && setting.SOR_Instead))
            {
                for (index iCell = 0; iCell < nCell; iCell++)
                    (*SOR_iScan2iCell)[iCell] = (*SOR_iCell2iScan)[iCell] = iCell;
                return;
            }

            // for (index iCell = 0; iCell < nCell; iCell++)
            //     (*SOR_iScan2iCell)[iCell] = -1;
            // // ? seeding
            // (*SOR_iScan2iCell)[0] = 0;

            std::vector<index> color(nCell, -1);
            std::vector<index> currentBoundary, nextBoundary;
            index nUnknown = nCell;

            //?seeding
            index cColor = 0;
            color[0] = cColor;
            nUnknown--;
            currentBoundary.push_back(0);

            while (nUnknown > 0)
            {

                if (currentBoundary.size() == 0)
                {
                    abort();
                }
                // if (mpi.rank == 0)
                // {
                //     std::cout << "cB ====== " << std::endl;
                //     for (auto i : currentBoundary)
                //         std::cout << "cB:" << i << std::endl;
                // }

                cColor = 1 - cColor; //? 2 colors
                nextBoundary.clear();
                for (auto iCell : currentBoundary)
                {
                    auto &c2f = mesh->cell2faceLocal[iCell];
                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                    {
                        index iFace = c2f[ic2f];
                        auto &f2c = (*mesh->face2cellPair)[iFace];
                        index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                        if (iCellOther != FACE_2_VOL_EMPTY && iCellOther < nCell) // * note that ghost cells unwanted
                            if (color[iCellOther] == -1)
                                nextBoundary.push_back(iCellOther), nUnknown--, color[iCellOther] = cColor;
                    }
                }
                currentBoundary = std::move(nextBoundary);
            }
            // if (mpi.rank == 0)
            // {
            //     for (auto i : color)
            //         std::cout << "C: " << i << std::endl;

            //     abort();
            // }
            std::vector<index> colorSizes(2, 0);
            for (auto c : color)
            {
                assert(c == 0 || c == 1);
                colorSizes[c]++;
            }
            std::vector<index> colorStarts(3, 0);
            for (index iC = 0; iC < 2; iC++)
                colorStarts[iC + 1] = colorStarts[iC] + colorSizes[iC];

            colorSizes.assign(colorSizes.size(), 0);
            for (index iCell = 0; iCell < nCell; iCell++)
            {
                index c = color[iCell];
                (*SOR_iScan2iCell)[colorSizes[c] + colorStarts[c]] = iCell;
                (*SOR_iCell2iScan)[iCell] = colorSizes[c] + colorStarts[c];
                colorSizes[c]++;
            }

            // if (mpi.rank == 0)
            // {
            //     for (auto i : (*SOR_iScan2iCell))
            //         std::cout << i << std::endl;

            //     abort();
            // }
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

            bool inverseScan = setting.SOR_InverseScanning && setting.SOR_Instead;

            for (index iScan = 0; iScan < uRec.dist->size() * (inverseScan ? 2 : 1); iScan++)
            {

                index iCell = iScan;
                if (inverseScan && iCell >= uRec.dist->size())
                    iCell = 2 * uRec.dist->size() - iScan - 1;
                if (setting.SOR_Instead)
                    iCell = (*SOR_iScan2iCell)[iCell];

                if (setting.SOR_InverseScanning)
                    auto &uRecE = uRec[iCell];
                real relax = cellRecAtrLocal[iCell][0].relax;
                auto &c2f = mesh->cell2faceLocal[iCell];
                if (!setting.SOR_Instead)
                    uRecNewBuf[iCell].m() = (1 - relax) * uRec[iCell].m();
                else
                    uRec[iCell].m() = (1 - relax) * uRec[iCell].m();

                auto matrixBatchElem = (*matrixBatch)[iCell];
                auto vectorBatchElem = (*vectorBatch)[iCell];

                // Eigen::MatrixXd coords;
                // mesh->LoadCoords(mesh->cell2nodeLocal[iCell], coords);
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
                    auto faceDiBjGaussBatchElem = (*faceDiBjGaussBatch)[iFace];

                    if (iCellOther != FACE_2_VOL_EMPTY)
                    {
                        if (!setting.SOR_Instead)
                            uRecNewBuf[iCell].m() +=
                                relax * ((matrixBatchElem.m(ic2f + 1) * uRec[iCellOther].m()) +
                                         ((u[iCellOther].p() - u[iCell].p()) * vectorBatchElem.m(0).row(ic2f)).transpose());
                        else
                            uRec[iCell].m() +=
                                relax * ((matrixBatchElem.m(ic2f + 1) * uRec[iCellOther].m()) +
                                         ((u[iCellOther].p() - u[iCell].p()) * vectorBatchElem.m(0).row(ic2f)).transpose());
                    }
                    else
                    {
                        if (faceAttribute.iPhy == BoundaryType::Wall)
                        {
                            Eigen::Vector<real, vsize> uBV;
                            uBV.setZero();
                            if (!setting.SOR_Instead)
                                uRecNewBuf[iCell].m() +=
                                    relax * (((uBV - u[iCell].p()) * vectorBatchElem.m(0).row(ic2f)).transpose());
                            else
                                uRec[iCell].m() +=
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
                        else if (faceAttribute.iPhy == BoundaryType::Farfield ||
                                 faceAttribute.iPhy == BoundaryType::Special_DMRFar)
                        {
                            if (!setting.SOR_Instead)
                                uRecNewBuf[iCell].m() +=
                                    relax * ((matrixBatchElem.m(ic2f + 1) * uRec[iCell].m()));
                            else
                                uRec[iCell].m() +=
                                    relax * ((matrixBatchElem.m(ic2f + 1) * uRec[iCell].m()));
                        }
                        else if (faceAttribute.iPhy == BoundaryType::Wall_Euler)
                        {
                            Eigen::MatrixXd BCCorrection;
                            BCCorrection.resizeLike(uRec[iCell].m());
                            BCCorrection.setZero();
                            eFace.Integration(
                                BCCorrection,
                                [&](Eigen::MatrixXd &corInc, int ig, Elem::tPoint &p, Elem::tDiFj &iDiNj)
                                {
                                    // auto &diffI = faceDiBjGaussCache[iFace][ig * 2 + 0]; // must be left
                                    auto diffI = faceDiBjGaussBatchElem.m(ig * 2 + 0);
                                    Eigen::Vector<real, vsize> uBV;
                                    Eigen::Vector<real, vsize> uBL = (diffI.row(0).rightCols(uRec[iCell].m().rows()) *
                                                                      uRec[iCell].m())
                                                                         .transpose();
                                    uBL += u[iCell].p().transpose();
                                    uBV.setZero();
                                    uBV(0) = uBL(0);
                                    Elem::tPoint normOut = faceNorms[iFace][ig].stableNormalized();
                                    uBV({1, 2, 3}) = uBL({1, 2, 3}) - normOut * (normOut.dot(uBV({1, 2, 3})));
                                    uBV(4) = uBL(4);

                                    Eigen::MatrixXd rowUD = (uBV - u[iCell].p()).transpose();
                                    Eigen::MatrixXd rowDiffI = diffI.row(0).rightCols(uRec[iCell].m().rows());
                                    FFaceFunctional(iFace, ig, rowDiffI, rowUD, (*faceWeights)[iFace]({0}), corInc);
                                    corInc *= faceNorms[iFace][ig].norm();
                                });
                            if (!setting.SOR_Instead)
                                uRecNewBuf[iCell].m() +=
                                    relax * matrixBatchElem.m(0) * BCCorrection;
                            else
                                uRec[iCell].m() +=
                                    relax * matrixBatchElem.m(0) * BCCorrection;
                        }
                        else if (faceAttribute.iPhy == BoundaryType::Wall_NoSlip)
                        {
                            Eigen::MatrixXd BCCorrection;
                            BCCorrection.resizeLike(uRec[iCell].m());
                            BCCorrection.setZero();
                            eFace.Integration(
                                BCCorrection,
                                [&](Eigen::MatrixXd &corInc, int ig, Elem::tPoint &p, Elem::tDiFj &iDiNj)
                                {
                                    // auto &diffI = faceDiBjGaussCache[iFace][ig * 2 + 0]; // must be left
                                    auto diffI = faceDiBjGaussBatchElem.m(ig * 2 + 0);
                                    Eigen::Vector<real, vsize> uBV;
                                    Eigen::Vector<real, vsize> uBL = (diffI.row(0).rightCols(uRec[iCell].m().rows()) *
                                                                      uRec[iCell].m())
                                                                         .transpose();
                                    uBL += u[iCell].p().transpose();
                                    uBV.setZero();
                                    uBV(0) = uBL(0);
                                    // Elem::tPoint normOut = faceNorms[iFace][ig].stableNormalized();
                                    // auto uBLMomentum = uBL({1, 2, 3});
                                    // uBV({1, 2, 3}) = uBLMomentum - normOut * (normOut.dot(uBLMomentum));
                                    // uBV({1, 2, 3}).setZero();
                                    uBV({1, 2, 3}) = -uBL({1, 2, 3});
                                    uBV(4) = uBL(4);

                                    Eigen::MatrixXd rowUD = (uBV - u[iCell].p()).transpose();
                                    Eigen::MatrixXd rowDiffI = diffI.row(0).rightCols(uRec[iCell].m().rows());
                                    FFaceFunctional(iFace, ig, rowDiffI, rowUD, (*faceWeights)[iFace]({0}), corInc);
                                    corInc *= faceNorms[iFace][ig].norm();
                                });
                            if (!setting.SOR_Instead)
                                uRecNewBuf[iCell].m() +=
                                    relax * matrixBatchElem.m(0) * BCCorrection;
                            else
                                uRec[iCell].m() +=
                                    relax * matrixBatchElem.m(0) * BCCorrection;
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

#ifdef PRINT_EVERY_VR_JACOBI_ITER_INCREMENT
            real vall = 0;
            real nall = 0;
            real vallR, nallR;
            for (index iCell = 0; iCell < uRec.dist->size(); iCell++)
            {
                auto &uRecE = uRec[iCell];
                nall += (uRecE.m() - uRecNewBuf[iCell].m()).squaredNorm();
                vall += 1;
            }
            // // std::cout << "NEW\n"
            // //           << uRecNewBuf[0] << std::endl;
            MPI_Allreduce(&vall, &vallR, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm); //? remove at release
            MPI_Allreduce(&nall, &nallR, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            real res = nallR / vallR;
            if (mpi.rank == 0)
            {
                auto fmt = log().flags();
                log() << " Rec RES " << std::scientific << std::setprecision(10) << res << std::endl;
                log().setf(fmt);
            }
#endif

            if (!setting.SOR_Instead)
            {
                for (index iCell = 0; iCell < uRec.dist->size(); iCell++)
                {
                    auto &uRecE = uRec[iCell];
                    uRecE.m() = uRecNewBuf[iCell].m();
                }
            }
            icount++;
        }

        void ReconstructionJacobiStep(ArrayLocal<VecStaticBatch<1>> &u,
                                      ArrayLocal<SemiVarMatrix<1>> &uRec,
                                      ArrayLocal<SemiVarMatrix<1>> &uRecNewBuf);

        /**
         * @brief input vector<Eigen::Array-like>
         */
        template <typename TinOthers, typename Tout>
        static inline void FWBAP_L2_Multiway_Polynomial2D(const TinOthers &uOthers, int Nother, Tout &uOut)
        {
            using namespace DNDS;
            static const int p = 4;
            static const real verySmallReal_pDiP = std::pow(verySmallReal, 1.0 / p);

            Eigen::ArrayXXd uUp; //* copy!
            uUp.resizeLike(uOthers[0]);
            uUp.setZero();
            Eigen::ArrayXd uDown;
            uDown.resize(uOthers[0].cols());
            uDown.setZero();
            Eigen::ArrayXXd uMax = uUp + verySmallReal;
            for (int iOther = 0; iOther < Nother; iOther++)
                uMax = uMax.max(uOthers[iOther].abs());
            uMax.rowwise() = uMax.colwise().maxCoeff();
            uOut = uMax;

            for (int iOther = 0; iOther < Nother; iOther++)
            {
                Eigen::ArrayXd thetaNorm;
                Eigen::ArrayXXd theta = uOthers[iOther] / uMax;
                switch (theta.rows())
                {
                case 2:
                    thetaNorm =
                        theta(0, Eigen::all).pow(2) +
                        theta(1, Eigen::all).pow(2);
                    break;
                case 3:
                    thetaNorm =
                        theta(0, Eigen::all).pow(2) +
                        theta(1, Eigen::all).pow(2) * 0.5 +
                        theta(2, Eigen::all).pow(2);
                    break;
                case 4:
                    thetaNorm =
                        theta(0, Eigen::all).pow(2) +
                        theta(1, Eigen::all).pow(2) * (1. / 3.) +
                        theta(2, Eigen::all).pow(2) * (1. / 3.) +
                        theta(3, Eigen::all).pow(2);
                    break;

                default:
                    assert(false);
                    break;
                }
                thetaNorm += verySmallReal_pDiP;
                thetaNorm = thetaNorm.pow(-p / 2);

                uDown += thetaNorm;
                uUp += theta.rowwise() * thetaNorm.transpose();
            }

            // std::cout << uUp << std::endl;
            // std::cout << uDown << std::endl;
            uOut *= uUp.rowwise() / (uDown.transpose() + verySmallReal);

            // // * Do cut off
            // for (int iOther = 0; iOther < Nother; iOther++)
            // {
            //     Eigen::ArrayXd uDotuOut;
            //     switch (uOut.rows())
            //     {
            //     case 2:
            //         uDotuOut =
            //             uOthers[iOther](0, Eigen::all) * uOut(0, Eigen::all) +
            //             uOthers[iOther](1, Eigen::all) * uOut(1, Eigen::all);
            //         break;
            //     case 3:
            //         uDotuOut =
            //             uOthers[iOther](0, Eigen::all) * uOut(0, Eigen::all) +
            //             uOthers[iOther](1, Eigen::all) * uOut(1, Eigen::all) * 0.5 +
            //             uOthers[iOther](2, Eigen::all) * uOut(2, Eigen::all);
            //         break;
            //     case 4:
            //         uDotuOut =
            //             uOthers[iOther](0, Eigen::all) * uOut(0, Eigen::all) +
            //             uOthers[iOther](1, Eigen::all) * uOut(1, Eigen::all) * (1. / 3.) +
            //             uOthers[iOther](2, Eigen::all) * uOut(2, Eigen::all) * (1. / 3.) +
            //             uOthers[iOther](3, Eigen::all) * uOut(3, Eigen::all);
            //         break;

            //     default:
            //         assert(false);
            //         break;
            //     }

            //     uOut.rowwise() *= 0.5 * (uDotuOut.sign().transpose() + 1);
            // }
            // // * Do cut off

            if (uOut.hasNaN())
            {
                std::cout << "Limiter FWBAP_L2_Multiway Failed" << std::endl;
                std::cout << uMax.transpose() << std::endl;
                std::cout << uUp.transpose() << std::endl;
                std::cout << uDown.transpose() << std::endl;
                abort();
            }
        }

        /**
         * @brief input vector<Eigen::Array-like>
         */
        template <typename Tcenter, typename TinOthers, typename Tout>
        static inline void FMEMM_Multiway_Polynomial2D(const Tcenter &u, const TinOthers &uOthers, int Nother, Tout &uOut)
        {
            using namespace DNDS;
            static const int p = 4;

            Eigen::ArrayXXd umax = u.abs();
            umax.rowwise() = umax.colwise().maxCoeff() + verySmallReal;

            Eigen::ArrayXXd theta0 = u / umax;
            Eigen::ArrayXXd thetaMinNorm = theta0;

            for (int iOther = 0; iOther < Nother; iOther++)
            {
                Eigen::ArrayXd thetaMinNormNorm;
                Eigen::ArrayXd thetaNorm;
                Eigen::ArrayXXd theta = uOthers[iOther] / umax;
                Eigen::ArrayXd theta0DotTheta;
                switch (theta.rows())
                {
                case 2:
                    thetaNorm =
                        theta(0, Eigen::all).pow(2) +
                        theta(1, Eigen::all).pow(2);
                    thetaMinNormNorm =
                        thetaMinNorm(0, Eigen::all).pow(2) +
                        thetaMinNorm(1, Eigen::all).pow(2);
                    theta0DotTheta =
                        theta(0, Eigen::all) * theta0(0, Eigen::all) +
                        theta(1, Eigen::all) * theta0(1, Eigen::all);
                    break;
                case 3:
                    thetaNorm =
                        theta(0, Eigen::all).pow(2) +
                        theta(1, Eigen::all).pow(2) * 0.5 +
                        theta(2, Eigen::all).pow(2);
                    thetaMinNormNorm =
                        thetaMinNorm(0, Eigen::all).pow(2) +
                        thetaMinNorm(1, Eigen::all).pow(2) * 0.5 +
                        thetaMinNorm(2, Eigen::all).pow(2);
                    theta0DotTheta =
                        theta(0, Eigen::all) * theta0(0, Eigen::all) +
                        theta(1, Eigen::all) * theta0(1, Eigen::all) * 0.5 +
                        theta(2, Eigen::all) * theta0(2, Eigen::all);
                    break;
                case 4:
                    thetaNorm =
                        theta(0, Eigen::all).pow(2) +
                        theta(1, Eigen::all).pow(2) * (1. / 3.) +
                        theta(2, Eigen::all).pow(2) * (1. / 3.) +
                        theta(3, Eigen::all).pow(2);
                    thetaMinNormNorm =
                        thetaMinNorm(0, Eigen::all).pow(2) +
                        thetaMinNorm(1, Eigen::all).pow(2) * (1. / 3.) +
                        thetaMinNorm(2, Eigen::all).pow(2) * (1. / 3.) +
                        thetaMinNorm(3, Eigen::all).pow(2);
                    theta0DotTheta =
                        theta(0, Eigen::all) * theta0(0, Eigen::all) +
                        theta(1, Eigen::all) * theta0(1, Eigen::all) * (1. / 3.) +
                        theta(2, Eigen::all) * theta0(2, Eigen::all) * (1. / 3.) +
                        theta(3, Eigen::all) * theta0(3, Eigen::all);
                    break;

                default:
                    assert(false);
                    break;
                }
                Eigen::ArrayXd selection = (thetaNorm - thetaMinNormNorm).sign() * 0.5 + 0.5; //! need eliminate one side?
                thetaMinNorm = theta.rowwise() * (1 - selection).transpose() +
                               thetaMinNorm.rowwise() * selection.transpose();
                // //! cutting
                // theta0 = theta0.rowwise() * (theta0DotTheta.sign() + 1).transpose() * 0.5;
            }
            Eigen::ArrayXd thetaNorm;
            Eigen::ArrayXd thetaMinNormNorm;
            switch (theta0.rows())
            {
            case 2:
                thetaNorm =
                    theta0(0, Eigen::all).pow(2) +
                    theta0(1, Eigen::all).pow(2);
                thetaMinNormNorm =
                    thetaMinNorm(0, Eigen::all).pow(2) +
                    thetaMinNorm(1, Eigen::all).pow(2);
                break;
            case 3:
                thetaNorm =
                    theta0(0, Eigen::all).pow(2) +
                    theta0(1, Eigen::all).pow(2) * 0.5 +
                    theta0(2, Eigen::all).pow(2);
                thetaMinNormNorm =
                    thetaMinNorm(0, Eigen::all).pow(2) +
                    thetaMinNorm(1, Eigen::all).pow(2) * 0.5 +
                    thetaMinNorm(2, Eigen::all).pow(2);
                break;
            case 4:
                thetaNorm =
                    theta0(0, Eigen::all).pow(2) +
                    theta0(1, Eigen::all).pow(2) * (1. / 3.) +
                    theta0(2, Eigen::all).pow(2) * (1. / 3.) +
                    theta0(3, Eigen::all).pow(2);
                thetaMinNormNorm =
                    thetaMinNorm(0, Eigen::all).pow(2) +
                    thetaMinNorm(1, Eigen::all).pow(2) * (1. / 3.) +
                    thetaMinNorm(2, Eigen::all).pow(2) * (1. / 3.) +
                    thetaMinNorm(3, Eigen::all).pow(2);
                break;
            default:
                assert(false);
                break;
            }
            Eigen::ArrayXd replaceLoc = ((thetaNorm / (thetaMinNormNorm + verySmallReal)).sqrt() - 1).max(verySmallReal);
            // Eigen::ArrayXd replaceFactor = 2 - (-replaceLoc).exp();
            // Eigen::ArrayXd replaceFactor = 2 - (replaceLoc * p + 1).pow(-1. / p);
            Eigen::ArrayXd replaceFactor = replaceLoc * 0 + 1;
            // Eigen::ArrayXd replaceFactor = 1+(1 - (replaceLoc * p + 1).pow(-1. / p)) / (replaceLoc/10+1 );

            replaceFactor = (replaceFactor - 1) / replaceLoc;

            // !safety?
            Eigen::ArrayXd ifReplace = (thetaNorm - thetaMinNormNorm).sign() * 0.5 + 0.5;
            replaceFactor = ifReplace * replaceFactor + (1 - ifReplace);

            uOut = u.rowwise() * replaceFactor.transpose() + (thetaMinNorm * umax).rowwise() * (1 - replaceFactor).transpose();

            if (uOut.hasNaN())
            {
                std::cout << "Limiter FMEMM_L2_Multiway Failed" << std::endl;
                std::cout << umax.transpose() << std::endl;
                std::cout << uOut.transpose() << std::endl;
                std::cout << replaceFactor << std::endl;
                std::cout << replaceLoc << std::endl;
                abort();
            }
        }

        /**
         * @brief input vector<Eigen::Array-like>
         */
        template <typename TinOthers, typename Tout>
        static inline void FWBAP_L2_Multiway_PolynomialOrth(const TinOthers &uOthers, int Nother, Tout &uOut)
        {
            using namespace DNDS;
            static const int p = 4;
            static const real verySmallReal_pDiP = std::pow(verySmallReal, 1.0 / p);

            Eigen::ArrayXXd uUp; //* copy!
            uUp.resizeLike(uOthers[0]);
            uUp.setZero();
            Eigen::ArrayXd uDown;
            uDown.resize(uOthers[0].cols());
            uDown.setZero();
            Eigen::ArrayXXd uMax = uUp + verySmallReal;
            for (int iOther = 0; iOther < Nother; iOther++)
                uMax = uMax.max(uOthers[iOther].abs());
            uMax.rowwise() = uMax.colwise().maxCoeff();
            uOut = uMax;

            for (int iOther = 0; iOther < Nother; iOther++)
            {
                Eigen::ArrayXd thetaNorm;
                Eigen::ArrayXXd theta = uOthers[iOther] / uMax;
                thetaNorm = (theta * theta).colwise().sum();
                thetaNorm += verySmallReal_pDiP;
                thetaNorm = thetaNorm.pow(-p / 2);

                uDown += thetaNorm;
                uUp += theta.rowwise() * thetaNorm.transpose();
            }
            // std::cout << uUp << std::endl;
            // std::cout << uDown << std::endl;
            uOut *= uUp.rowwise() / (uDown.transpose() + verySmallReal);
            if (uOut.hasNaN())
            {
                std::cout << "Limiter FWBAP_L2_Multiway Failed" << std::endl;
                std::cout << uMax.transpose() << std::endl;
                std::cout << uUp.transpose() << std::endl;
                std::cout << uDown.transpose() << std::endl;
                abort();
            }
        }

        /**
         * @brief input vector<Eigen::Array-like>
         */
        template <typename TinOthers, typename Tout>
        inline void FWBAP_L2_Multiway(const TinOthers &uOthers, int Nother, Tout &uOut)
        {
            static const int p = 4;
            static const real verySmallReal_pDiP = std::pow(verySmallReal, 1.0 / p);

            Eigen::ArrayXXd uUp; //* copy!
            uUp.resizeLike(uOthers[0]);
            uUp.setZero();
            Eigen::ArrayXXd uDown = uUp; //* copy!
            Eigen::ArrayXXd uMax = uDown + verySmallReal;

            for (int iOther = 0; iOther < Nother; iOther++)
                uMax = uMax.max(uOthers[iOther].abs());
            uOut = uMax;

            for (int iOther = 0; iOther < Nother; iOther++)
            {
                auto thetaInverse = uMax / (uOthers[iOther].sign() * (uOthers[iOther].abs() + verySmallReal_pDiP) +
                                            verySmallReal_pDiP * (-2));
                uDown += thetaInverse.pow(p);
                uUp += thetaInverse.pow(p - 1);
            }
            uOut *= uUp / (uDown + verySmallReal);

            if (uOut.hasNaN())
            {
                std::cout << "Limiter FWBAP_L2_Multiway Failed" << std::endl;
                std::cout << uMax.transpose() << std::endl;
                std::cout << uUp.transpose() << std::endl;
                std::cout << uDown.transpose() << std::endl;
                abort();
            }
        }

        /**
         * @brief input eigen arrays
         */
        template <typename Tin1, typename Tin2, typename Tout>
        inline void FWBAP_L2_Biway(const Tin1 &u1, const Tin2 &u2, Tout &uOut, real n)
        {
            static const int p = 4;
            // static const real n = 10.0;
            static const real verySmallReal_pDiP = std::pow(verySmallReal, 1.0 / p);
            auto uMax = u1.abs().max(u2.abs()) + verySmallReal_pDiP;
            auto u1p = (u1 / uMax).pow(p);
            auto u2p = (u2 / uMax).pow(p);
            // std::cout << u1 << std::endl;

            uOut = (u1p * u2 + n * u2p * u1) / ((u1p + n * u2p) + verySmallReal);
            // uOut *= (u1.sign() + u2.sign()).abs() * 0.5; //! cutting below zero!!!
            // std::cout << u2 << std::endl;
        }

        template <typename Tin1, typename Tin2, typename Tout>
        inline void FWBAP_L2_Biway_Polynomial2D(const Tin1 &u1, const Tin2 &u2, Tout &uOut, real n)
        {
            static const int p = 4;
            // static const real n = 10.0;
            static const real verySmallReal_pDiP = std::pow(verySmallReal, 1.0 / p);
            Eigen::ArrayXXd uMax = u1.abs().max(u2.abs()) + verySmallReal_pDiP;
            uMax.rowwise() = uMax.colwise().maxCoeff();
            Eigen::ArrayXd u1p, u2p;
            Eigen::ArrayXXd theta1 = u1 / uMax;
            Eigen::ArrayXXd theta2 = u2 / uMax;
            switch (u1.rows())
            {
            case 2:
                u1p =
                    theta1(0, Eigen::all).pow(2) +
                    theta1(1, Eigen::all).pow(2);
                u2p =
                    theta2(0, Eigen::all).pow(2) +
                    theta2(1, Eigen::all).pow(2);
                break;
            case 3:
                u1p =
                    theta1(0, Eigen::all).pow(2) +
                    theta1(1, Eigen::all).pow(2) * 0.5 +
                    theta1(2, Eigen::all).pow(2);
                u2p =
                    theta2(0, Eigen::all).pow(2) +
                    theta2(1, Eigen::all).pow(2) * 0.5 +
                    theta2(2, Eigen::all).pow(2);
                break;
            case 4:
                u1p =
                    theta1(0, Eigen::all).pow(2) +
                    theta1(1, Eigen::all).pow(2) * (1. / 3.) +
                    theta1(2, Eigen::all).pow(2) * (1. / 3.) +
                    theta1(3, Eigen::all).pow(2);
                u2p =
                    theta2(0, Eigen::all).pow(2) +
                    theta2(1, Eigen::all).pow(2) * (1. / 3.) +
                    theta2(2, Eigen::all).pow(2) * (1. / 3.) +
                    theta2(3, Eigen::all).pow(2);
                break;

            default:
                assert(false);
                break;
            }
            u1p = u1p.pow(p / 2);
            u2p = u2p.pow(p / 2);

            uOut = (u2.rowwise() * u1p.transpose() + n * (u1.rowwise() * u2p.transpose())).rowwise() / ((u1p + n * u2p) + verySmallReal).transpose();

            // std::cout << u2 << std::endl;
        }

        template <typename Tin1, typename Tin2, typename Tout>
        inline void FMEMM_Biway_Polynomial2D(const Tin1 &u1, const Tin2 &u2, Tout &uOut, real n)
        {
            static const int p = 4;
            // static const real n = 10.0;
            static const real verySmallReal_pDiP = std::pow(verySmallReal, 1.0 / p);
            Eigen::ArrayXXd uMax = u1.abs().max(u2.abs()) + verySmallReal_pDiP;
            uMax.rowwise() = uMax.colwise().maxCoeff();
            Eigen::ArrayXd u1p, u2p, u1u2;
            Eigen::ArrayXXd theta1 = u1 / uMax;
            Eigen::ArrayXXd theta2 = u2 / uMax;
            switch (u1.rows())
            {
            case 2:
                u1p =
                    theta1(0, Eigen::all).pow(2) +
                    theta1(1, Eigen::all).pow(2);
                u2p =
                    theta2(0, Eigen::all).pow(2) +
                    theta2(1, Eigen::all).pow(2);
                u1u2 =
                    theta2(0, Eigen::all) * theta1(0, Eigen::all) +
                    theta2(1, Eigen::all) * theta1(1, Eigen::all);
                break;
            case 3:
                u1p =
                    theta1(0, Eigen::all).pow(2) +
                    theta1(1, Eigen::all).pow(2) * 0.5 +
                    theta1(2, Eigen::all).pow(2);
                u2p =
                    theta2(0, Eigen::all).pow(2) +
                    theta2(1, Eigen::all).pow(2) * 0.5 +
                    theta2(2, Eigen::all).pow(2);
                u1u2 =
                    theta2(0, Eigen::all) * theta1(0, Eigen::all) +
                    theta2(1, Eigen::all) * theta1(1, Eigen::all) * 0.5 +
                    theta2(2, Eigen::all) * theta1(2, Eigen::all);
                break;
            case 4:
                u1p =
                    theta1(0, Eigen::all).pow(2) +
                    theta1(1, Eigen::all).pow(2) * (1. / 3.) +
                    theta1(2, Eigen::all).pow(2) * (1. / 3.) +
                    theta1(3, Eigen::all).pow(2);
                u2p =
                    theta2(0, Eigen::all).pow(2) +
                    theta2(1, Eigen::all).pow(2) * (1. / 3.) +
                    theta2(2, Eigen::all).pow(2) * (1. / 3.) +
                    theta2(3, Eigen::all).pow(2);
                u1u2 =
                    theta2(0, Eigen::all) * theta1(0, Eigen::all) +
                    theta2(1, Eigen::all) * theta1(1, Eigen::all) * (1. / 3.) +
                    theta2(2, Eigen::all) * theta1(2, Eigen::all) * (1. / 3.) +
                    theta2(3, Eigen::all) * theta1(3, Eigen::all);
                break;

            default:
                assert(false);
                break;
            }
            u1p = u1p.sqrt();
            u2p = u2p.sqrt();

            Eigen::ArrayXd replaceLoc = (u1p / (u2p + verySmallReal) - 1).max(verySmallReal);
            // Eigen::ArrayXd replaceFactor = 2 - (-replaceLoc).exp();
            // Eigen::ArrayXd replaceFactor = 2 - (replaceLoc * p + 1).pow(-1. / p);
            Eigen::ArrayXd replaceFactor = replaceLoc * 0 + 1;
            // Eigen::ArrayXd replaceFactor = 1+(1 - (replaceLoc * p + 1).pow(-1. / p)) / (replaceLoc/10+1 );

            replaceFactor = (replaceFactor - 1) / replaceLoc;

            // !safety?
            Eigen::ArrayXd ifReplace = (u1p - u2p).sign() * 0.5 + 0.5;
            replaceFactor = ifReplace * replaceFactor + (1 - ifReplace);

            uOut = u1.rowwise() * replaceFactor.transpose() + u2.rowwise() * (1 - replaceFactor).transpose();
            // //! cutting
            // uOut = uOut.rowwise() * (u1u2.sign() + 1).transpose() * 0.5;

            // std::cout << u2 << std::endl;
        }

        template <typename Tin1, typename Tin2, typename Tout>
        inline void FWBAP_L2_Biway_PolynomialOrth(const Tin1 &u1, const Tin2 &u2, Tout &uOut, real n)
        {
            static const int p = 4;
            // static const real n = 10.0;
            static const real verySmallReal_pDiP = std::pow(verySmallReal, 1.0 / p);
            Eigen::ArrayXXd uMax = u1.abs().max(u2.abs()) + verySmallReal_pDiP;
            uMax.rowwise() = uMax.colwise().maxCoeff();
            Eigen::ArrayXd u1p, u2p;
            Eigen::ArrayXXd theta1 = u1 / uMax;
            Eigen::ArrayXXd theta2 = u2 / uMax;
            u1p = (theta1 * theta1).colwise().sum();
            u2p = (theta2 * theta2).colwise().sum();
            u1p = u1p.pow(p / 2);
            u2p = u2p.pow(p / 2);

            uOut = (u2.rowwise() * u1p.transpose() + n * (u1.rowwise() * u2p.transpose())).rowwise() / ((u1p + n * u2p) + verySmallReal).transpose();

            // std::cout << u2 << std::endl;
        }

        template <typename TinC, typename TinOthers, typename Tout>
        inline void FWBAP_LE_Multiway(const TinC &uC, const TinOthers &uOthers, int Nother, Tout &uOut)
        {
            static const int p = 4;
            static const real n = 100.0;
            static const real verySmallReal_pDiP = std::pow(verySmallReal, 1.0 / p);
            static const real eps = 5;

            //! TODO:
            // static_assert(false, "Incomplete Implementation");
        }

        template <uint32_t vsize>
        // static const int vsize = 1; // intellisense helper: give example...
        void BuildRecFacial(ArrayLocal<SemiVarMatrix<vsize>> &uR)
        {
            index nCellDist = mesh->cell2nodeLocal.dist->size();
            // InsertCheck(mpi, "BuildRecFacial Start");
            uR.dist = std::make_shared<typename decltype(uR.dist)::element_type>(
                typename decltype(uR.dist)::element_type::tContext(
                    [&](index i) -> rowsize
                    {
                        index nface = mesh->cell2faceLocal[i].size();
                        return vsize * (cellRecAtrLocal[i][0].NDOF - 1) * nface; // !!note - 1!!
                    },
                    nCellDist),
                mpi);
            uR.CreateGhostCopyComm(mesh->cellAtrLocal);
            uR.InitPersistentPullClean();
        }

        void BuildIfUseLimiter(ArrayLocal<Batch<real, 1>> &ifUseLimiter)
        {
            index nCellDist = mesh->cell2nodeLocal.dist->size();
            ifUseLimiter.dist = std::make_shared<typename decltype(ifUseLimiter.dist)::element_type>(
                decltype(ifUseLimiter.dist)::element_type::tContext(
                    nCellDist),
                mpi);
            ifUseLimiter.CreateGhostCopyComm(mesh->cellAtrLocal);
            ifUseLimiter.InitPersistentPullClean();
        }

        /**
         * @brief FM(uLeft,uRight,norm) gives vsize * vsize mat of Left Eigen Vectors
         *
         */
        template <uint32_t vsize = 1, typename TFM, typename TFMI>
        // static const int vsize = 1; // intellisense helper: give example...
        void ReconstructionWBAPLimitFacial(ArrayLocal<VecStaticBatch<vsize>> &u,
                                           ArrayLocal<SemiVarMatrix<vsize>> &uRec,
                                           ArrayLocal<SemiVarMatrix<vsize>> &uRecNewBuf,
                                           ArrayLocal<SemiVarMatrix<vsize>> &uRecFacialBuf,
                                           ArrayLocal<SemiVarMatrix<vsize>> &uRecFacialNewBuf,
                                           ArrayLocal<Batch<real, 1>> &ifUseLimiter,
                                           TFM &&FM, TFMI &&FMI)
        {
            static int icount = 0;

            // InsertCheck(mpi, "ReconstructionWBAPLimitFacial Start");
            //* Step 0: prepare the facial buf
            for (index iScan = 0; iScan < uRec.dist->size(); iScan++)
            {
                index iCell = iScan;
                index NRecDOF = cellRecAtrLocal[iCell][0].NDOF - 1; // ! not good ! TODO
                auto &c2f = mesh->cell2faceLocal[iCell];

                Eigen::Matrix<real, 2, 2> IJIISIsum;
                IJIISIsum.setZero();
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    auto &faceRecAtr = faceRecAtrLocal[iFace][0];
                    auto &faceAtr = mesh->faceAtrLocal[iFace][0];
                    auto f2c = mesh->face2cellLocal[iFace];
                    auto faceDiBjGaussBatchElemVR = (*faceDiBjGaussBatch)[iFace];
                    // Elem::ElementManager eFace(faceAtr.type, faceRecAtr.intScheme);
                    Elem::ElementManager eFace(faceAtr.type, Elem::INT_SCHEME_LINE_1); // !using faster integration
                    index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                    index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                    if (iCellOther != FACE_2_VOL_EMPTY)
                    {
                        auto uR = iCellAtFace ? u[iCell].p() : u[iCellOther].p();
                        auto uL = iCellAtFace ? u[iCellOther].p() : u[iCell].p();
                        auto M = FM(uL, uR, faceNormCenter[iFace].stableNormalized());
                        Eigen::MatrixXd V = uRec[iCell].m();
                        V = (M * V.transpose()).transpose();
                        uRecFacialBuf[iCell]
                            .m()(
                                Eigen::seq(ic2f * NRecDOF, ic2f * NRecDOF + NRecDOF - 1),
                                Eigen::all) = V;
                    }
                    else
                    {
                        uRecFacialBuf[iCell]
                            .m()(
                                Eigen::seq(ic2f * NRecDOF, ic2f * NRecDOF + NRecDOF - 1),
                                Eigen::all)
                            .setConstant(UnInitReal);
                    }
                    Eigen::Matrix<real, 2, 2> IJIISI;
                    IJIISI.setZero();
                    eFace.Integration(
                        IJIISI,
                        [&](Eigen::Matrix<real, 2, 2> &finc, int ig, Elem::tPoint &p, Elem::tDiFj &DiNj)
                        {
                            int nDiff = faceWeights->operator[](iFace).size();
                            Elem::tPoint unitNorm = faceNorms[iFace][ig].normalized();
                            //! Taking only rho and E
                            Eigen::MatrixXd uRecVal(nDiff, 2), uRecValL(nDiff, 2), uRecValR(nDiff, 2), uRecValJump(nDiff, 2);
                            uRecVal.setZero(), uRecValJump.setZero();
                            uRecValL = faceDiBjGaussBatchElemVR.m(ig * 2 + 0).rightCols(uRec[iCell].m().rows()) * uRec[iCell].m()(Eigen::all, {0, 4});
                            uRecValL(0, Eigen::all) += u[iCell].p()({0, 4}).transpose();

                            if (iCellOther != FACE_2_VOL_EMPTY)
                            {
                                uRecValR = faceDiBjGaussBatchElemVR.m(ig * 2 + 1).rightCols(uRec[iCellOther].m().rows()) * uRec[iCellOther].m()(Eigen::all, {0, 4});
                                uRecValR(0, Eigen::all) += u[iCellOther].p()({0, 4}).transpose();
                                uRecVal = (uRecValL + uRecValR) * 0.5;
                                uRecValJump = (uRecValL - uRecValR) * 0.5;
                            }

                            Eigen::MatrixXd IJI, ISI;
                            FFaceFunctional(iFace, ig, uRecVal, uRecVal, (*faceWeights)[iFace], ISI);
                            FFaceFunctional(iFace, ig, uRecValJump, uRecValJump, (*faceWeights)[iFace], IJI);
                            finc({0, 1}, 0) = IJI.diagonal();
                            finc({0, 1}, 1) = ISI.diagonal();

                            finc *= faceNorms[iFace][ig].norm(); // don't forget this
                        });
                    IJIISIsum += IJIISI;
                }
                Eigen::Vector<real, 2> smoothIndicator =
                    (IJIISIsum({0, 1}, 0).array() /
                     (IJIISIsum({0, 1}, 1).array() + verySmallReal))
                        .matrix();
                real sImax = smoothIndicator.array().abs().maxCoeff();
                // sImax = smoothIndicator(0);
                // ifUseLimiter[iCell] = (ifUseLimiter[iCell] << 1) |
                //                       (std::sqrt(sImax) > setting.WBAP_SmoothIndicatorScale / (P_ORDER * P_ORDER)
                //                            ? 0x00000001U
                //                            : 0x00000000U);
                ifUseLimiter[iCell][0] = std::sqrt(sImax) * (P_ORDER * P_ORDER);
            }
            // assert(u.ghost->commStat.hasPersistentPullReqs);
            // assert(uRecFacialBuf.ghost->commStat.hasPersistentPullReqs);
            // exit(0);
            uRecFacialBuf.StartPersistentPullClean();
            uRecFacialBuf.WaitPersistentPullClean();
            ifUseLimiter.StartPersistentPullClean();
            ifUseLimiter.WaitPersistentPullClean();

            // InsertCheck(mpi, "ReconstructionWBAPLimitFacial Step 1");
            //* Step 1: facial hierachical limiting
            int cPOrder = P_ORDER;
            for (; cPOrder >= 1; cPOrder--) //! 2d here
            {
                int LimStart, LimEnd; // End is inclusive
                switch (cPOrder)
                {
                case 3:
                    LimStart = 5;
                    LimEnd = 8;
                    break;
                case 2:
                    LimStart = 2;
                    LimEnd = 4;
                    break;
                case 1:
                    LimStart = 0;
                    LimEnd = 1;
                    break;

                default:
                    assert(false);
                    break;
                }

                for (index iScan = 0; iScan < uRec.dist->size(); iScan++)
                {
                    index iCell = iScan;

                    index NRecDOF = cellRecAtrLocal[iCell][0].NDOF - 1;
                    auto &c2f = mesh->cell2faceLocal[iCell];

                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                    {
                        index iFace = c2f[ic2f];
                        auto &f2c = (*mesh->face2cellPair)[iFace];
                        index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                        index iCellAtFace = f2c[0] == iCell ? 0 : 1;

                        auto matrixSecondaryBatchElem = (*matrixSecondaryBatch)[iFace];

                        if (iCellOther != FACE_2_VOL_EMPTY)
                        {
                            index NRecDOFOther = cellRecAtrLocal[iCellOther][0].NDOF - 1;
                            if (NRecDOFOther < (LimEnd + 1) || NRecDOF < (LimEnd + 1))
                                continue; // reserved for p-adaption
                            // if (!(ifUseLimiter[iCell] & 0x0000000FU))
                            //     continue;
                            if (ifUseLimiter[iCell][0] < setting.WBAP_SmoothIndicatorScale &&
                                ifUseLimiter[iCellOther][0] < setting.WBAP_SmoothIndicatorScale)
                                continue;
                            auto &cOther2f = mesh->cell2faceLocal[iCellOther];
                            index icOther2f = 0;
                            //* find icOther2f
                            for (; icOther2f < cOther2f.size(); icOther2f++)
                                if (iFace == cOther2f[icOther2f])
                                    break;
                            assert(icOther2f < cOther2f.size());

                            const auto &matrixSecondary =
                                iCellAtFace
                                    ? matrixSecondaryBatchElem.m(1)
                                    : matrixSecondaryBatchElem.m(0);
                            //! note that when false == bool(iCellAtFace), this cell is at left of the face

                            auto uOtherIn =
                                (matrixSecondary *
                                 uRecFacialBuf[iCellOther].m()(
                                     Eigen::seq(
                                         icOther2f * NRecDOFOther + 0,
                                         icOther2f * NRecDOFOther + NRecDOFOther - 1),
                                     Eigen::all))(
                                    Eigen::seq(
                                        LimStart,
                                        LimEnd),
                                    Eigen::all);
                            auto uThisIn =
                                uRecFacialBuf[iCell].m()(
                                    Eigen::seq(
                                        ic2f * NRecDOF + LimStart,
                                        ic2f * NRecDOF + LimEnd),
                                    Eigen::all);

                            Eigen::ArrayXXd uLimOutArray;
                            real n = setting.WBAP_nStd;
                            // if (ifUseLimiter[iCell] < 2 * setting.WBAP_SmoothIndicatorScale)
                            // {
                            //     real eIS = (ifUseLimiter[iCell] - setting.WBAP_SmoothIndicatorScale) / (setting.WBAP_SmoothIndicatorScale);
                            //     n *= std::exp((1 - eIS) * 10);
                            // }
                            FWBAP_L2_Biway(uThisIn.array(), uOtherIn.array(), uLimOutArray, n);
                            if (uLimOutArray.hasNaN())
                            {
                                std::cout << uThisIn.array().transpose() << std::endl;
                                std::cout << uOtherIn.array().transpose() << std::endl;
                                std::cout << uLimOutArray.transpose() << std::endl;
                                std::abort();
                            }

                            uRecFacialNewBuf[iCell]
                                .m()(
                                    Eigen::seq(
                                        ic2f * NRecDOF + LimStart,
                                        ic2f * NRecDOF + LimEnd),
                                    Eigen::all) = uLimOutArray.matrix();
                        }
                        else
                        {
                        }
                    }
                }
                uRecFacialNewBuf.StartPersistentPullClean();
                uRecFacialNewBuf.WaitPersistentPullClean();
                //! why ????? need a InitPersistentPullClean() !

                for (index iScan = 0; iScan < uRec.dist->size(); iScan++)
                {
                    index iCell = iScan;
                    // if (!(ifUseLimiter[iCell] & 0x0000000FU))
                    //     continue;
                    index NRecDOF = cellRecAtrLocal[iCell][0].NDOF - 1;
                    auto &c2f = mesh->cell2faceLocal[iCell];

                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                    {
                        index iFace = c2f[ic2f];
                        auto &f2c = (*mesh->face2cellPair)[iFace];
                        index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                        index iCellAtFace = f2c[0] == iCell ? 0 : 1;

                        auto matrixSecondaryBatchElem = (*matrixSecondaryBatch)[iFace];

                        if (iCellOther != FACE_2_VOL_EMPTY)
                        {
                            if (ifUseLimiter[iCell][0] < setting.WBAP_SmoothIndicatorScale &&
                                ifUseLimiter[iCellOther][0] < setting.WBAP_SmoothIndicatorScale)
                                continue;
                            index NRecDOFOther = cellRecAtrLocal[iCellOther][0].NDOF - 1;
                            if (NRecDOFOther < (LimEnd + 1) || NRecDOF < (LimEnd + 1))
                                continue; // reserved for p-adaption

                            uRecFacialBuf[iCell].m()(
                                Eigen::seq(
                                    ic2f * NRecDOF + LimStart,
                                    ic2f * NRecDOF + LimEnd),
                                Eigen::all) =
                                uRecFacialNewBuf[iCell].m()(
                                    Eigen::seq(
                                        ic2f * NRecDOF + LimStart,
                                        ic2f * NRecDOF + LimEnd),
                                    Eigen::all);
                        }
                        else
                        {
                        }
                    }
                }

                uRecFacialBuf.StartPersistentPullClean();
                uRecFacialBuf.WaitPersistentPullClean();
            }

            // InsertCheck(mpi, "ReconstructionWBAPLimitFacial Step 2");
            //* Step 2: facial V 2 U
            for (index iScan = 0; iScan < uRec.dist->size(); iScan++)
            {
                index iCell = iScan;
                // if (!(ifUseLimiter[iCell] & 0x0000000FU))
                //     continue;
                index NRecDOF = cellRecAtrLocal[iCell][0].NDOF - 1;
                auto &c2f = mesh->cell2faceLocal[iCell];

                bool ifOtherUse = false;
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    auto &f2c = (*mesh->face2cellPair)[iFace];
                    index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                    index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                    if (iCellOther != FACE_2_VOL_EMPTY)
                    {
                        if (ifUseLimiter[iCellOther][0] >= setting.WBAP_SmoothIndicatorScale)
                            ifOtherUse = true;
                    }
                }
                if (ifUseLimiter[iCell][0] < setting.WBAP_SmoothIndicatorScale && (!ifOtherUse))
                    continue;

                std::vector<Eigen::Array<real, -1, vsize>> uOthers;
                // InsertCheck(mpi, "ReconstructionWBAPLimitFacial Step 2-0");
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    auto &f2c = (*mesh->face2cellPair)[iFace];
                    index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                    index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                    if (iCellOther != FACE_2_VOL_EMPTY)
                    {

                        auto uR = iCellAtFace ? u[iCell].p() : u[iCellOther].p();
                        auto uL = iCellAtFace ? u[iCellOther].p() : u[iCell].p();
                        auto MI = FMI(uL, uR, faceNormCenter[iFace].stableNormalized());
                        Eigen::ArrayXXd uOther;
                        uOther = (MI * uRecFacialBuf[iCell]
                                           .m()(
                                               Eigen::seq(ic2f * NRecDOF, ic2f * NRecDOF + NRecDOF - 1),
                                               Eigen::all)
                                           .transpose())
                                     .transpose()
                                     .array();
                        // InsertCheck(mpi, "ReconstructionWBAPLimitFacial Step 2-0-0");
                        uOthers.push_back(uOther);
                        // InsertCheck(mpi, "ReconstructionWBAPLimitFacial Step 2-0-1");
                    }
                    else
                    {
                    }
                }
                // InsertCheck(mpi, "ReconstructionWBAPLimitFacial Step 2-1");
                Eigen::ArrayXXd uLimOutArray;

                Eigen::ArrayXXd uBefore = uRec[iCell].m();

                FWBAP_L2_Multiway(uOthers, uOthers.size(), uLimOutArray);
                if (uLimOutArray.hasNaN())
                {
                    // std::cout << uRec[iCell].m().array().transpose() << std::endl;
                    for (auto e : uOthers)
                        std::cout << "E: \n"
                                  << e.transpose() << std::endl;
                    std::cout << uLimOutArray.transpose() << std::endl;
                    std::abort();
                }

                real relax = 1;
                // if (ifUseLimiter[iCell][0] < 2 * setting.WBAP_SmoothIndicatorScale)
                // {
                //     real eIS = (ifUseLimiter[iCell][0] - setting.WBAP_SmoothIndicatorScale) / (setting.WBAP_SmoothIndicatorScale);
                //     relax = eIS;
                // } //! relaxation
                uRecNewBuf[iCell].m() = uLimOutArray.matrix() * relax + uRec[iCell].m() * (1 - relax);

                // std::cout << "new Old" << std::endl;
                // std::cout << std::setprecision(10) << uRecNewBuf[iCell].m().transpose() << std::endl;
                // std::cout << std::setprecision(10) << uBefore.transpose() << std::endl;
                // std::cout << std::setprecision(10) << (uBefore - uRecNewBuf[iCell].m()).transpose() << std::endl;
                // InsertCheck(mpi, "ReconstructionWBAPLimitFacial Step 2-2");
            }
        }

        /**
         * @brief FM(uLeft,uRight,norm) gives vsize * vsize mat of Left Eigen Vectors
         *
         */
        template <uint32_t vsize = 1, typename TFM, typename TFMI>
        // static const int vsize = 1; // intellisense helper: give example...
        void ReconstructionWBAPLimitFacialV2(ArrayLocal<VecStaticBatch<vsize>> &u,
                                             ArrayLocal<SemiVarMatrix<vsize>> &uRec,
                                             ArrayLocal<SemiVarMatrix<vsize>> &uRecNewBuf,
                                             ArrayLocal<SemiVarMatrix<vsize>> &uRecNewBuf1,
                                             ArrayLocal<Batch<real, 1>> &ifUseLimiter,
                                             bool ifAll,
                                             TFM &&FM, TFMI &&FMI)
        {
            static int icount = 0;

            // InsertCheck(mpi, "ReconstructionWBAPLimitFacial Start");
            //* Step 0: prepare the facial buf
            for (index iScan = 0; iScan < uRec.dist->size(); iScan++)
            {
                index iCell = iScan;
                index NRecDOF = cellRecAtrLocal[iCell][0].NDOF - 1; // ! not good ! TODO
                auto &c2f = mesh->cell2faceLocal[iCell];
                // uRecNewBuf[iCell].m() = uRec[iCell].m();

                Eigen::Matrix<real, 2, 2> IJIISIsum;
                IJIISIsum.setZero();
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    auto &faceRecAtr = faceRecAtrLocal[iFace][0];
                    auto &faceAtr = mesh->faceAtrLocal[iFace][0];
                    auto f2c = mesh->face2cellLocal[iFace];
                    auto faceDiBjGaussBatchElemVR = (*faceDiBjGaussBatch)[iFace];
                    // Elem::ElementManager eFace(faceAtr.type, faceRecAtr.intScheme);
                    Elem::ElementManager eFace(faceAtr.type, Elem::INT_SCHEME_LINE_1); // !using faster integration
                    index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                    index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                    Eigen::Matrix<real, 2, 2> IJIISI;
                    IJIISI.setZero();
                    eFace.Integration(
                        IJIISI,
                        [&](Eigen::Matrix<real, 2, 2> &finc, int ig, Elem::tPoint &p, Elem::tDiFj &DiNj)
                        {
                            int nDiff = faceWeights->operator[](iFace).size();
                            Elem::tPoint unitNorm = faceNorms[iFace][ig].normalized();
                            //! Taking only rho and E
                            Eigen::MatrixXd uRecVal(nDiff, 2), uRecValL(nDiff, 2), uRecValR(nDiff, 2), uRecValJump(nDiff, 2);
                            uRecVal.setZero(), uRecValJump.setZero();
                            uRecValL = faceDiBjGaussBatchElemVR.m(ig * 2 + 0).rightCols(uRec[iCell].m().rows()) * uRec[iCell].m()(Eigen::all, {0, 4});
                            uRecValL(0, Eigen::all) += u[iCell].p()({0, 4}).transpose();

                            if (iCellOther != FACE_2_VOL_EMPTY)
                            {
                                uRecValR = faceDiBjGaussBatchElemVR.m(ig * 2 + 1).rightCols(uRec[iCellOther].m().rows()) * uRec[iCellOther].m()(Eigen::all, {0, 4});
                                uRecValR(0, Eigen::all) += u[iCellOther].p()({0, 4}).transpose();
                                uRecVal = (uRecValL + uRecValR) * 0.5;
                                uRecValJump = (uRecValL - uRecValR) * 0.5;
                            }

                            Eigen::MatrixXd IJI, ISI;
                            FFaceFunctional(iFace, ig, uRecVal, uRecVal, (*faceWeights)[iFace], ISI);
                            FFaceFunctional(iFace, ig, uRecValJump, uRecValJump, (*faceWeights)[iFace], IJI);
                            finc({0, 1}, 0) = IJI.diagonal();
                            finc({0, 1}, 1) = ISI.diagonal();

                            finc *= faceNorms[iFace][ig].norm(); // don't forget this
                        });
                    IJIISIsum += IJIISI;
                }
                Eigen::Vector<real, 2> smoothIndicator =
                    (IJIISIsum({0, 1}, 0).array() /
                     (IJIISIsum({0, 1}, 1).array() + verySmallReal))
                        .matrix();
                real sImax = smoothIndicator.array().abs().maxCoeff();
                // sImax = smoothIndicator(0);
                // ifUseLimiter[iCell] = (ifUseLimiter[iCell] << 1) |
                //                       (std::sqrt(sImax) > setting.WBAP_SmoothIndicatorScale / (P_ORDER * P_ORDER)
                //                            ? 0x00000001U
                //                            : 0x00000000U);
                ifUseLimiter[iCell][0] = std::sqrt(sImax) * (P_ORDER * P_ORDER);
            }
            // assert(u.ghost->commStat.hasPersistentPullReqs);
            // assert(uRecFacialBuf.ghost->commStat.hasPersistentPullReqs);
            // exit(0);
            ifUseLimiter.StartPersistentPullClean();
            ifUseLimiter.WaitPersistentPullClean();

            // uRecNewBuf.StartPersistentPullClean();
            // uRecNewBuf.WaitPersistentPullClean();
            for (index iCell = 0; iCell < uRec.size(); iCell++)
            {
                uRecNewBuf[iCell].m() = uRec[iCell].m();
            }

            // InsertCheck(mpi, "ReconstructionWBAPLimitFacial Step 1");
            //* Step 1: facial hierachical limiting
            int cPOrder = P_ORDER;
            for (; cPOrder >= 1; cPOrder--) //! 2d here
            {
                int LimStart, LimEnd; // End is inclusive
                switch (cPOrder)
                {
                case 3:
                    LimStart = 5;
                    LimEnd = 8;
                    break;
                case 2:
                    LimStart = 2;
                    LimEnd = 4;
                    break;
                case 1:
                    LimStart = 0;
                    LimEnd = 1;
                    break;

                default:
                    assert(false);
                    break;
                }

                for (index iScan = 0; iScan < uRec.dist->size(); iScan++)
                {
                    index iCell = iScan;
                    if (ifUseLimiter[iCell][0] < setting.WBAP_SmoothIndicatorScale && (!ifAll))
                        continue;
                    index NRecDOF = cellRecAtrLocal[iCell][0].NDOF - 1;
                    auto &c2f = mesh->cell2faceLocal[iCell];

                    std::vector<Eigen::ArrayXXd> uOthers;
                    Eigen::ArrayXXd uC = uRecNewBuf[iCell].m()(
                        Eigen::seq(
                            LimStart,
                            LimEnd),
                        Eigen::all);
                    auto &c2n = mesh->cell2nodeLocal[iCell];
                    Eigen::MatrixXd coords;
                    mesh->LoadCoords(c2n, coords);
                    Elem::tPoint sScaleThis = CoordMinMaxScale(coords);

                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                    {
                        index iFace = c2f[ic2f];
                        auto &f2c = (*mesh->face2cellPair)[iFace];
                        index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                        index iCellAtFace = f2c[0] == iCell ? 0 : 1;

                        auto matrixSecondaryBatchElem = (*matrixSecondaryBatch)[iFace];

                        if (iCellOther != FACE_2_VOL_EMPTY)
                        {
                            index NRecDOFOther = cellRecAtrLocal[iCellOther][0].NDOF - 1;
                            if (NRecDOFOther < (LimEnd + 1) || NRecDOF < (LimEnd + 1))
                                continue; // reserved for p-adaption
                            // if (!(ifUseLimiter[iCell] & 0x0000000FU))
                            //     continue;

                            auto &c2n = mesh->cell2nodeLocal[iCellOther];
                            Eigen::MatrixXd coords;
                            mesh->LoadCoords(c2n, coords);
                            Elem::tPoint sScaleOther = CoordMinMaxScale(coords);

                            Elem::tPoint unitNorm = faceNormCenter[iFace].stableNormalized();

                            auto &cOther2f = mesh->cell2faceLocal[iCellOther];
                            index icOther2f = 0;
                            //* find icOther2f
                            for (; icOther2f < cOther2f.size(); icOther2f++)
                                if (iFace == cOther2f[icOther2f])
                                    break;
                            assert(icOther2f < cOther2f.size());

                            const auto &matrixSecondary =
                                iCellAtFace
                                    ? matrixSecondaryBatchElem.m(1)
                                    : matrixSecondaryBatchElem.m(0);
                            //! note that when false == bool(iCellAtFace), this cell is at left of the face

                            // std::cout << sScaleThis.transpose() << std::endl;
                            // std::cout << sScaleOther.transpose() << std::endl;
                            // std::cout << ((getCellCenter(iCell)-getCellCenter(iCellOther)).array()/sScaleOther.array()).transpose() << std::endl;
                            // std::cout << matrixSecondary << std::endl;
                            // assert(false);

                            Eigen::MatrixXd uOtherIn =
                                (matrixSecondary *
                                 uRecNewBuf[iCellOther].m())(
                                    Eigen::seq(
                                        LimStart,
                                        LimEnd),
                                    Eigen::all);
                            Eigen::MatrixXd uThisIn =
                                uC.matrix();
                            // 2 char space :
                            auto uR = iCellAtFace ? u[iCell].p() : u[iCellOther].p();
                            auto uL = iCellAtFace ? u[iCellOther].p() : u[iCell].p();
                            auto M = FM(uL, uR, unitNorm);

                            uOtherIn = (M * uOtherIn.transpose()).transpose();
                            uThisIn = (M * uThisIn.transpose()).transpose();

                            Eigen::ArrayXXd uLimOutArray;
                            real n = setting.WBAP_nStd;
                            // if (ifUseLimiter[iCell] < 2 * setting.WBAP_SmoothIndicatorScale)
                            // {
                            //     real eIS = (ifUseLimiter[iCell] - setting.WBAP_SmoothIndicatorScale) / (setting.WBAP_SmoothIndicatorScale);
                            //     n *= std::exp((1 - eIS) * 10);
                            // }

                            if (setting.normWBAP)
                            {
                                if (setting.orthogonalizeBase)
                                    FWBAP_L2_Biway_PolynomialOrth(uThisIn.array(), uOtherIn.array(), uLimOutArray, n);
                                else
                                    // FMEMM_Biway_Polynomial2D(uThisIn.array(), uOtherIn.array(), uLimOutArray, n);
                                    FWBAP_L2_Biway_Polynomial2D(uThisIn.array(), uOtherIn.array(), uLimOutArray, n);
                            }
                            else
                                FWBAP_L2_Biway(uThisIn.array(), uOtherIn.array(), uLimOutArray, n);

                            if (uLimOutArray.hasNaN())
                            {
                                std::cout << uThisIn.array().transpose() << std::endl;
                                std::cout << uOtherIn.array().transpose() << std::endl;
                                std::cout << uLimOutArray.transpose() << std::endl;
                                assert(false);
                            }

                            // to phys space
                            auto MI = FMI(uL, uR, unitNorm);
                            uLimOutArray = (MI * uLimOutArray.matrix().transpose()).transpose().array();
                            uOthers.push_back(uLimOutArray);
                        }
                        else
                        {
                        }
                    }
                    Eigen::ArrayXXd uLimOutArray;

                    if (setting.normWBAP)
                    {
                        if (setting.orthogonalizeBase)
                            FWBAP_L2_Multiway_PolynomialOrth(uOthers, uOthers.size(), uLimOutArray);
                        else
                            // FMEMM_Multiway_Polynomial2D(uC, uOthers, uOthers.size(), uLimOutArray);
                            FWBAP_L2_Multiway_Polynomial2D(uOthers, uOthers.size(), uLimOutArray);
                    }
                    else
                        FWBAP_L2_Multiway(uOthers, uOthers.size(), uLimOutArray);

                    if (uLimOutArray.hasNaN())
                    {
                        // std::cout << uRec[iCell].m().array().transpose() << std::endl;
                        for (auto e : uOthers)
                            std::cout << "E: \n"
                                      << e.transpose() << std::endl;
                        std::cout << uLimOutArray.transpose() << std::endl;
                        assert(false);
                    }
                    uRecNewBuf1[iCell].m()(
                        Eigen::seq(
                            LimStart,
                            LimEnd),
                        Eigen::all) = uLimOutArray.matrix();
                }
                // uRecFacialNewBuf.StartPersistentPullClean();
                // uRecFacialNewBuf.WaitPersistentPullClean();
                uRecNewBuf1.StartPersistentPullClean();
                uRecNewBuf1.WaitPersistentPullClean();

                for (index iScan = 0; iScan < uRec.size(); iScan++)
                {
                    index iCell = iScan;
                    uRecNewBuf[iCell].m()(
                        Eigen::seq(
                            LimStart,
                            LimEnd),
                        Eigen::all) =
                        uRecNewBuf1[iCell].m()(
                            Eigen::seq(
                                LimStart,
                                LimEnd),
                            Eigen::all);
                }
            }
            for (index iScan = 0; iScan < uRec.dist->size(); iScan++)
            {
            }
        }

        /**
         * @brief FM(uLeft,uRight,norm) gives vsize * vsize mat of Left Eigen Vectors
         *
         */
        template <uint32_t vsize = 1, typename TFM, typename TFMI>
        // static const int vsize = 1; // intellisense helper: give example...
        void ReconstructionWBAPLimitFacialV3(ArrayLocal<VecStaticBatch<vsize>> &u,
                                             ArrayLocal<SemiVarMatrix<vsize>> &uRec,
                                             ArrayLocal<SemiVarMatrix<vsize>> &uRecNewBuf,
                                             ArrayLocal<SemiVarMatrix<vsize>> &uRecNewBuf1,
                                             ArrayLocal<Batch<real, 1>> &ifUseLimiter,
                                             bool ifAll,
                                             TFM &&FM, TFMI &&FMI)
        {
            static int icount = 0;

            // InsertCheck(mpi, "ReconstructionWBAPLimitFacial Start");
            //* Step 0: prepare the facial buf
            for (index iScan = 0; iScan < uRec.dist->size(); iScan++)
            {
                index iCell = iScan;
                index NRecDOF = cellRecAtrLocal[iCell][0].NDOF - 1; // ! not good ! TODO
                auto &c2f = mesh->cell2faceLocal[iCell];

                Eigen::Matrix<real, 2, 2> IJIISIsum;
                IJIISIsum.setZero();
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    auto &faceRecAtr = faceRecAtrLocal[iFace][0];
                    auto &faceAtr = mesh->faceAtrLocal[iFace][0];
                    auto f2c = mesh->face2cellLocal[iFace];
                    auto faceDiBjGaussBatchElemVR = (*faceDiBjGaussBatch)[iFace];
                    // Elem::ElementManager eFace(faceAtr.type, faceRecAtr.intScheme);
                    Elem::ElementManager eFace(faceAtr.type, Elem::INT_SCHEME_LINE_1); // !using faster integration
                    index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                    index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                    Eigen::Matrix<real, 2, 2> IJIISI;
                    IJIISI.setZero();
                    eFace.Integration(
                        IJIISI,
                        [&](Eigen::Matrix<real, 2, 2> &finc, int ig, Elem::tPoint &p, Elem::tDiFj &DiNj)
                        {
                            int nDiff = faceWeights->operator[](iFace).size();
                            Elem::tPoint unitNorm = faceNorms[iFace][ig].normalized();
                            //! Taking only rho and E
                            Eigen::MatrixXd uRecVal(nDiff, 2), uRecValL(nDiff, 2), uRecValR(nDiff, 2), uRecValJump(nDiff, 2);
                            uRecVal.setZero(), uRecValJump.setZero();
                            uRecValL = faceDiBjGaussBatchElemVR.m(ig * 2 + 0).rightCols(uRec[iCell].m().rows()) * uRec[iCell].m()(Eigen::all, {0, 4});
                            uRecValL(0, Eigen::all) += u[iCell].p()({0, 4}).transpose();

                            if (iCellOther != FACE_2_VOL_EMPTY)
                            {
                                uRecValR = faceDiBjGaussBatchElemVR.m(ig * 2 + 1).rightCols(uRec[iCellOther].m().rows()) * uRec[iCellOther].m()(Eigen::all, {0, 4});
                                uRecValR(0, Eigen::all) += u[iCellOther].p()({0, 4}).transpose();
                                uRecVal = (uRecValL + uRecValR) * 0.5;
                                uRecValJump = (uRecValL - uRecValR) * 0.5;
                            }

                            Eigen::MatrixXd IJI, ISI;
                            FFaceFunctional(iFace, ig, uRecVal, uRecVal, (*faceWeights)[iFace], ISI);
                            FFaceFunctional(iFace, ig, uRecValJump, uRecValJump, (*faceWeights)[iFace], IJI);
                            finc({0, 1}, 0) = IJI.diagonal();
                            finc({0, 1}, 1) = ISI.diagonal();

                            finc *= faceNorms[iFace][ig].norm(); // don't forget this
                        });
                    IJIISIsum += IJIISI;
                }
                Eigen::Vector<real, 2> smoothIndicator =
                    (IJIISIsum({0, 1}, 0).array() /
                     (IJIISIsum({0, 1}, 1).array() + verySmallReal))
                        .matrix();
                real sImax = smoothIndicator.array().abs().maxCoeff();
                // sImax = smoothIndicator(0);
                // ifUseLimiter[iCell] = (ifUseLimiter[iCell] << 1) |
                //                       (std::sqrt(sImax) > setting.WBAP_SmoothIndicatorScale / (P_ORDER * P_ORDER)
                //                            ? 0x00000001U
                //                            : 0x00000000U);
                ifUseLimiter[iCell][0] = std::sqrt(sImax) * (P_ORDER * P_ORDER);
            }
            // assert(u.ghost->commStat.hasPersistentPullReqs);
            // assert(uRecFacialBuf.ghost->commStat.hasPersistentPullReqs);
            // exit(0);
            ifUseLimiter.StartPersistentPullClean();
            ifUseLimiter.WaitPersistentPullClean();

            for (index iCell = 0; iCell < uRec.size(); iCell++)
            {
                uRecNewBuf[iCell].m() = uRec[iCell].m();
            }

            // InsertCheck(mpi, "ReconstructionWBAPLimitFacial Step 1");

            for (index iScan = 0; iScan < uRec.dist->size(); iScan++)
            {
                index iCell = iScan;
                if (ifUseLimiter[iCell][0] < setting.WBAP_SmoothIndicatorScale && (!ifAll))
                    continue;
                index NRecDOF = cellRecAtrLocal[iCell][0].NDOF - 1;
                auto &c2f = mesh->cell2faceLocal[iCell];
                std::vector<Eigen::MatrixXd> uFaces(c2f.size());
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    auto &f2c = (*mesh->face2cellPair)[iFace];
                    index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                    index iCellAtFace = f2c[0] == iCell ? 0 : 1;

                    auto matrixSecondaryBatchElem = (*matrixSecondaryBatch)[iFace];

                    if (iCellOther != FACE_2_VOL_EMPTY)
                        uFaces[ic2f] = uRecNewBuf[iCellOther].m() * 1e100;
                }

                int cPOrder = P_ORDER;
                for (; cPOrder >= 1; cPOrder--) //! 2d here
                {
                    int LimStart, LimEnd; // End is inclusive
                    switch (cPOrder)
                    {
                    case 3:
                        LimStart = 5;
                        LimEnd = 8;
                        break;
                    case 2:
                        LimStart = 2;
                        LimEnd = 4;
                        break;
                    case 1:
                        LimStart = 0;
                        LimEnd = 1;
                        break;

                    default:
                        assert(false);
                        break;
                    }

                    std::vector<Eigen::ArrayXXd> uOthers;
                    Eigen::ArrayXXd uC = uRecNewBuf[iCell].m()(
                        Eigen::seq(
                            LimStart,
                            LimEnd),
                        Eigen::all);

                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                    {
                        index iFace = c2f[ic2f];
                        auto &f2c = (*mesh->face2cellPair)[iFace];
                        index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                        index iCellAtFace = f2c[0] == iCell ? 0 : 1;

                        auto matrixSecondaryBatchElem = (*matrixSecondaryBatch)[iFace];

                        if (iCellOther != FACE_2_VOL_EMPTY)
                        {
                            index NRecDOFOther = cellRecAtrLocal[iCellOther][0].NDOF - 1;
                            index NRecDOFLim = std::min(NRecDOFOther, NRecDOF);
                            if (NRecDOFLim < (LimEnd + 1))
                                continue; // reserved for p-adaption
                            // if (!(ifUseLimiter[iCell] & 0x0000000FU))
                            //     continue;

                            auto &c2n = mesh->cell2nodeLocal[iCellOther];
                            Eigen::MatrixXd coords;
                            mesh->LoadCoords(c2n, coords);
                            Elem::tPoint sScaleOther = CoordMinMaxScale(coords);

                            Elem::tPoint unitNorm = faceNormCenter[iFace].stableNormalized();

                            auto &cOther2f = mesh->cell2faceLocal[iCellOther];
                            index icOther2f = 0;
                            //* find icOther2f
                            for (; icOther2f < cOther2f.size(); icOther2f++)
                                if (iFace == cOther2f[icOther2f])
                                    break;
                            assert(icOther2f < cOther2f.size());

                            const auto &matrixSecondary =
                                iCellAtFace
                                    ? matrixSecondaryBatchElem.m(1)
                                    : matrixSecondaryBatchElem.m(0);

                            const auto &matrixSecondaryOther =
                                iCellAtFace
                                    ? matrixSecondaryBatchElem.m(0)
                                    : matrixSecondaryBatchElem.m(1);
                            // std::cout << "A"<<std::endl;
                            //! note that when false == bool(iCellAtFace), this cell is at left of the face
                            Eigen::MatrixXd uOtherOther = uRecNewBuf[iCellOther].m()(Eigen::seq(0, NRecDOFLim - 1), Eigen::all);
                            if (LimEnd < uOtherOther.rows() - 1)
                                uOtherOther(Eigen::seq(LimEnd + 1, NRecDOFLim - 1), Eigen::all) =
                                    matrixSecondaryOther(Eigen::seq(LimEnd + 1, NRecDOFLim - 1), Eigen::seq(LimEnd + 1, NRecDOFLim - 1)) *
                                    uFaces[ic2f](Eigen::seq(LimEnd + 1, NRecDOFLim - 1), Eigen::all);
                            // std::cout << "B" << std::endl;
                            Eigen::MatrixXd uOtherIn =
                                matrixSecondary(Eigen::seq(LimStart, LimEnd), Eigen::all) * uOtherOther;

                            Eigen::MatrixXd uThisIn =
                                uC.matrix();

                            // 2 char space :
                            auto uR = iCellAtFace ? u[iCell].p() : u[iCellOther].p();
                            auto uL = iCellAtFace ? u[iCellOther].p() : u[iCell].p();
                            auto M = FM(uL, uR, unitNorm);

                            uOtherIn = (M * uOtherIn.transpose()).transpose();
                            uThisIn = (M * uThisIn.transpose()).transpose();

                            Eigen::ArrayXXd uLimOutArray;
                            real n = setting.WBAP_nStd;

                            if (setting.normWBAP)
                            {
                                if (setting.orthogonalizeBase)
                                    FWBAP_L2_Biway_PolynomialOrth(uThisIn.array(), uOtherIn.array(), uLimOutArray, n);
                                else
                                    // FMEMM_Biway_Polynomial2D(uThisIn.array(), uOtherIn.array(), uLimOutArray, n);
                                    FWBAP_L2_Biway_Polynomial2D(uThisIn.array(), uOtherIn.array(), uLimOutArray, n);
                            }
                            else
                                FWBAP_L2_Biway(uThisIn.array(), uOtherIn.array(), uLimOutArray, n);

                            if (uLimOutArray.hasNaN())
                            {
                                std::cout << uThisIn.array().transpose() << std::endl;
                                std::cout << uOtherIn.array().transpose() << std::endl;
                                std::cout << uLimOutArray.transpose() << std::endl;
                                assert(false);
                            }

                            // to phys space
                            auto MI = FMI(uL, uR, unitNorm);
                            uLimOutArray = (MI * uLimOutArray.matrix().transpose()).transpose().array();

                            uFaces[ic2f](Eigen::seq(LimStart, LimEnd), Eigen::all) = uLimOutArray.matrix();
                            uOthers.push_back(uLimOutArray);
                        }
                        else
                        {
                        }
                    }
                    Eigen::ArrayXXd uLimOutArray;

                    if (setting.normWBAP)
                    {
                        if (setting.orthogonalizeBase)
                            FWBAP_L2_Multiway_PolynomialOrth(uOthers, uOthers.size(), uLimOutArray);
                        else
                            // FMEMM_Multiway_Polynomial2D(uC, uOthers, uOthers.size(), uLimOutArray);
                            FWBAP_L2_Multiway_Polynomial2D(uOthers, uOthers.size(), uLimOutArray);
                    }
                    else
                        FWBAP_L2_Multiway(uOthers, uOthers.size(), uLimOutArray);

                    if (uLimOutArray.hasNaN())
                    {
                        // std::cout << uRec[iCell].m().array().transpose() << std::endl;
                        for (auto e : uOthers)
                            std::cout << "E: \n"
                                      << e.transpose() << std::endl;
                        std::cout << uLimOutArray.transpose() << std::endl;
                        assert(false);
                    }
                    uRecNewBuf1[iCell].m()(
                        Eigen::seq(
                            LimStart,
                            LimEnd),
                        Eigen::all) = uLimOutArray.matrix();
                }
            }
            uRecNewBuf1.StartPersistentPullClean();
            uRecNewBuf1.WaitPersistentPullClean();

            for (index iCell = 0; iCell < uRec.size(); iCell++)
            {
                uRecNewBuf[iCell].m() =
                    uRecNewBuf1[iCell].m();
            }
        }

        template <uint32_t vsize>
        void RecMatMulVec(ArrayLocal<SemiVarMatrix<vsize>> &x,
                          ArrayLocal<SemiVarMatrix<vsize>> &b)
        {
            for (index iCell = 0; iCell < x.dist->size(); iCell++)
            {
                auto &c2f = mesh->cell2faceLocal[iCell];

                auto matrixBatchElem = (*matrixBatch)[iCell];
                auto vectorBatchElem = (*vectorBatch)[iCell];

                b[iCell].m() = matrixAii->operator[](iCell) * x[iCell].m();

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

                    if (iCellOther != FACE_2_VOL_EMPTY)
                    {
                        // b[iCell].m() += matrixBatchElem.m(ic2f + 1) * u[iCellOther].m(); //!wrong, this matrixBatchElem is Aii^-1Bij
                    }
                    else
                    {
                        // if (faceAttribute.iPhy == BoundaryType::Wall)
                        // {
                        //     Eigen::Vector<real, vsize> uBV;
                        //     uBV.setZero();
                        //     if (!setting.SOR_Instead)
                        //         uRecNewBuf[iCell].m() +=
                        //             relax * (((uBV - u[iCell].p()) * vectorBatchElem.m(0).row(ic2f)).transpose());
                        //     else
                        //         uRec[iCell].m() +=
                        //             relax * (((uBV - u[iCell].p()) * vectorBatchElem.m(0).row(ic2f)).transpose());

                        //     // eFace.Integration(
                        //     //     BCCorrection,
                        //     //     [&](Eigen::MatrixXd &corInc, int ig, Elem::tPoint &p, Elem::tDiFj &iDiNj)
                        //     //     {
                        //     //         auto &diffI = faceDiBjGaussCache[iFace][ig * 2 + 0]; // must be left

                        //     //         Eigen::MatrixXd rowUD = (uBV - uRec[iCell].p()).transpose();
                        //     //         Eigen::MatrixXd rowDiffI = diffI.row(0);
                        //     //         FFaceFunctional(iFace,rowDiffI, rowUD, faceWeights[iFace]({0}), corInc);
                        //     //         corInc *= faceNorms[iFace][ig].norm();
                        //     //     });
                        // }
                        // else if (faceAttribute.iPhy == BoundaryType::Farfield)
                        // {
                        //     if (!setting.SOR_Instead)
                        //         uRecNewBuf[iCell].m() +=
                        //             relax * ((matrixBatchElem.m(ic2f + 1) * uRec[iCell].m()));
                        //     else
                        //         uRec[iCell].m() +=
                        //             relax * ((matrixBatchElem.m(ic2f + 1) * uRec[iCell].m()));
                        // }
                        // else
                        // {
                        //     assert(false);
                        // }
                        // ! unfinished mat-mult
                    }
                }
            }
        }

        void Initialization();
        void Initialization_RenewBase();
    };
}