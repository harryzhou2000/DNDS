#pragma once

#include "DNDS_Mesh.hpp"
#include "DNDS_HardEigen.h"
#include "DNDS_FV_Limiters.hpp"
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

            scaleL0 = nSizes(0) * 0.5;
            scaleL1 = nSizes(1) * 0.5;
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
            // std::cout << DiBj(0, Eigen::all) << std::endl;
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
        void initIntScheme(); //  2-d specific

        void initMoment();

        void initBaseDiffCache();

        void initReconstructionMatVec();

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
        template <class TU_DOF>
        // static const int vsize = 1; // intellisense helper: give example...
        void ReconstructionJacobiStep(TU_DOF &u,
                                      ArrayRecV &uRec,
                                      ArrayRecV &uRecNewBuf)
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

                real relax = cellRecAtrLocal[iCell][0].relax;
                auto &c2f = mesh->cell2faceLocal[iCell];
                if (!setting.SOR_Instead)
                    uRecNewBuf[iCell] = (1 - relax) * uRec[iCell];
                else
                    uRec[iCell] = (1 - relax) * uRec[iCell];

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
                            uRecNewBuf[iCell] +=
                                relax * ((matrixBatchElem.m(ic2f + 1) * uRec[iCellOther]) +
                                         ((u[iCellOther] - u[iCell]) * vectorBatchElem.m(0).row(ic2f)).transpose());
                        else
                            uRec[iCell] +=
                                relax * ((matrixBatchElem.m(ic2f + 1) * uRec[iCellOther]) +
                                         ((u[iCellOther] - u[iCell]) * vectorBatchElem.m(0).row(ic2f)).transpose());
                    }
                    else
                    {
                        if (faceAttribute.iPhy == BoundaryType::Wall)
                        {
                            Eigen::Vector<real, -1> uBV(u[iCell].size());
                            uBV.setZero();
                            if (!setting.SOR_Instead)
                                uRecNewBuf[iCell] +=
                                    relax * (((uBV - u[iCell]) * vectorBatchElem.m(0).row(ic2f)).transpose());
                            else
                                uRec[iCell] +=
                                    relax * (((uBV - u[iCell]) * vectorBatchElem.m(0).row(ic2f)).transpose());

                            // eFace.Integration(
                            //     BCCorrection,
                            //     [&](Eigen::MatrixXd &corInc, int ig, Elem::tPoint &p, Elem::tDiFj &iDiNj)
                            //     {
                            //         auto &diffI = faceDiBjGaussCache[iFace][ig * 2 + 0]; // must be left

                            //         Eigen::MatrixXd rowUD = (uBV - uRec[iCell]).transpose();
                            //         Eigen::MatrixXd rowDiffI = diffI.row(0);
                            //         FFaceFunctional(iFace,rowDiffI, rowUD, faceWeights[iFace]({0}), corInc);
                            //         corInc *= faceNorms[iFace][ig].norm();
                            //     });
                        }
                        else if (faceAttribute.iPhy == BoundaryType::Farfield ||
                                 faceAttribute.iPhy == BoundaryType::Special_DMRFar)
                        {
                            if (!setting.SOR_Instead)
                                uRecNewBuf[iCell] +=
                                    relax * ((matrixBatchElem.m(ic2f + 1) * uRec[iCell]));
                            else
                                uRec[iCell] +=
                                    relax * ((matrixBatchElem.m(ic2f + 1) * uRec[iCell]));
                        }
                        else if (faceAttribute.iPhy == BoundaryType::Wall_Euler)
                        {
                            Eigen::MatrixXd BCCorrection;
                            BCCorrection.resizeLike(uRec[iCell]);
                            BCCorrection.setZero();
                            eFace.Integration(
                                BCCorrection,
                                [&](Eigen::MatrixXd &corInc, int ig, Elem::tPoint &p, Elem::tDiFj &iDiNj)
                                {
                                    index nvars = u[iCell].size();
                                    // auto &diffI = faceDiBjGaussCache[iFace][ig * 2 + 0]; // must be left
                                    auto diffI = faceDiBjGaussBatchElem.m(ig * 2 + 0);
                                    Eigen::Vector<real, -1> uBV(nvars);
                                    Eigen::Vector<real, -1> uBL = (diffI.row(0).rightCols(uRec[iCell].rows()) *
                                                                   uRec[iCell])
                                                                      .transpose();
                                    uBL += u[iCell].transpose();
                                    uBV.setZero();
                                    uBV = uBL;
                                    uBV(0) = uBL(0);
                                    Elem::tPoint normOut = faceNorms[iFace][ig].stableNormalized();
                                    uBV({1, 2, 3}) = uBL({1, 2, 3}) - normOut * (normOut.dot(uBV({1, 2, 3})));
                                    uBV(4) = uBL(4);

                                    Eigen::MatrixXd rowUD = (uBV - u[iCell]).transpose();
                                    Eigen::MatrixXd rowDiffI = diffI.row(0).rightCols(uRec[iCell].rows());
                                    FFaceFunctional(iFace, ig, rowDiffI, rowUD, (*faceWeights)[iFace]({0}), corInc);
                                    corInc *= faceNorms[iFace][ig].norm();
                                });
                            if (!setting.SOR_Instead)
                                uRecNewBuf[iCell] +=
                                    relax * matrixBatchElem.m(0) * BCCorrection;
                            else
                                uRec[iCell] +=
                                    relax * matrixBatchElem.m(0) * BCCorrection;
                        }
                        else if (faceAttribute.iPhy == BoundaryType::Wall_NoSlip)
                        {
                            Eigen::MatrixXd BCCorrection;
                            BCCorrection.resizeLike(uRec[iCell]);
                            BCCorrection.setZero();
                            eFace.Integration(
                                BCCorrection,
                                [&](Eigen::MatrixXd &corInc, int ig, Elem::tPoint &p, Elem::tDiFj &iDiNj)
                                {
                                    index nvars = u[iCell].size();
                                    // auto &diffI = faceDiBjGaussCache[iFace][ig * 2 + 0]; // must be left
                                    auto diffI = faceDiBjGaussBatchElem.m(ig * 2 + 0);
                                    Eigen::Vector<real, -1> uBV(nvars);
                                    Eigen::Vector<real, -1> uBL = (diffI.row(0).rightCols(uRec[iCell].rows()) *
                                                                   uRec[iCell])
                                                                      .transpose();
                                    // Eigen::Matrix<real, -1, 2> graduBL = (diffI.row({1,2}).rightCols(uRec[iCell].rows()) *
                                    //                                       uRec[iCell])
                                    //                                          .transpose();//!2D!!

                                    uBL += u[iCell].transpose();
                                    // uBV.setZero();
                                    uBV = -uBL;
                                    uBV(0) = uBL(0);
                                    // Elem::tPoint normOut = faceNorms[iFace][ig].stableNormalized();
                                    // auto uBLMomentum = uBL({1, 2, 3});
                                    // uBV({1, 2, 3}) = uBLMomentum - normOut * (normOut.dot(uBLMomentum));
                                    // uBV({1, 2, 3}).setZero();
                                    uBV({1, 2, 3}) = -uBL({1, 2, 3});
                                    uBV(4) = uBL(4);

                                    Eigen::MatrixXd rowUD = (uBV - u[iCell]).transpose();
                                    Eigen::MatrixXd rowDiffI = diffI.row(0).rightCols(uRec[iCell].rows());
                                    FFaceFunctional(iFace, ig, rowDiffI, rowUD, (*faceWeights)[iFace]({0}), corInc);
                                    corInc *= faceNorms[iFace][ig].norm();
                                });
                            if (!setting.SOR_Instead)
                                uRecNewBuf[iCell] +=
                                    relax * matrixBatchElem.m(0) * BCCorrection;
                            else
                                uRec[iCell] +=
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
                    //           << uRecNewBuf[iCell] << std::endl;
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
                nall += (uRecE - uRecNewBuf[iCell]).squaredNorm();
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
                    uRec[iCell] = uRecNewBuf[iCell];
            }
            icount++;
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
        template <class TU_DOF, typename TFM, typename TFMI>
        // static const int vsize = 1; // intellisense helper: give example...
        void ReconstructionWBAPLimitFacialV2(TU_DOF &u,
                                             ArrayRecV &uRec,
                                             ArrayRecV &uRecNewBuf,
                                             ArrayRecV &uRecNewBuf1,
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
                            uRecValL = faceDiBjGaussBatchElemVR.m(ig * 2 + 0).rightCols(uRec[iCell].rows()) * uRec[iCell](Eigen::all, {0, 4});
                            uRecValL(0, Eigen::all) += u[iCell]({0, 4}).transpose();

                            if (iCellOther != FACE_2_VOL_EMPTY)
                            {
                                uRecValR = faceDiBjGaussBatchElemVR.m(ig * 2 + 1).rightCols(uRec[iCellOther].rows()) * uRec[iCellOther](Eigen::all, {0, 4});
                                uRecValR(0, Eigen::all) += u[iCellOther]({0, 4}).transpose();
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
                uRecNewBuf[iCell] = uRec[iCell];
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
                    Eigen::ArrayXXd uC = uRecNewBuf[iCell](
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
                                 uRecNewBuf[iCellOther])(
                                    Eigen::seq(
                                        LimStart,
                                        LimEnd),
                                    Eigen::all);
                            Eigen::MatrixXd uThisIn =
                                uC.matrix();
                            // 2 char space :
                            auto uR = iCellAtFace ? u[iCell] : u[iCellOther];
                            auto uL = iCellAtFace ? u[iCellOther] : u[iCell];
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
                        // std::cout << uRec[iCell].array().transpose() << std::endl;
                        for (auto e : uOthers)
                            std::cout << "E: \n"
                                      << e.transpose() << std::endl;
                        std::cout << uLimOutArray.transpose() << std::endl;
                        assert(false);
                    }
                    uRecNewBuf1[iCell](
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
                    uRecNewBuf[iCell](
                        Eigen::seq(
                            LimStart,
                            LimEnd),
                        Eigen::all) =
                        uRecNewBuf1[iCell](
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
                                             ArrayRecV &uRec,
                                             ArrayRecV &uRecNewBuf,
                                             ArrayRecV &uRecNewBuf1,
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
                            uRecValL = faceDiBjGaussBatchElemVR.m(ig * 2 + 0).rightCols(uRec[iCell].rows()) * uRec[iCell](Eigen::all, {0, 4});
                            uRecValL(0, Eigen::all) += u[iCell]({0, 4}).transpose();

                            if (iCellOther != FACE_2_VOL_EMPTY)
                            {
                                uRecValR = faceDiBjGaussBatchElemVR.m(ig * 2 + 1).rightCols(uRec[iCellOther].rows()) * uRec[iCellOther](Eigen::all, {0, 4});
                                uRecValR(0, Eigen::all) += u[iCellOther]({0, 4}).transpose();
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
                uRecNewBuf[iCell] = uRec[iCell];
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
                        uFaces[ic2f] = uRecNewBuf[iCellOther] * 1e100;
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
                    Eigen::ArrayXXd uC = uRecNewBuf[iCell](
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
                            Eigen::MatrixXd uOtherOther = uRecNewBuf[iCellOther](Eigen::seq(0, NRecDOFLim - 1), Eigen::all);
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
                            auto uR = iCellAtFace ? u[iCell] : u[iCellOther];
                            auto uL = iCellAtFace ? u[iCellOther] : u[iCell];
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
                        // std::cout << uRec[iCell].array().transpose() << std::endl;
                        for (auto e : uOthers)
                            std::cout << "E: \n"
                                      << e.transpose() << std::endl;
                        std::cout << uLimOutArray.transpose() << std::endl;
                        assert(false);
                    }
                    uRecNewBuf1[iCell](
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
                uRecNewBuf[iCell] =
                    uRecNewBuf1[iCell];
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

                b[iCell] = matrixAii->operator[](iCell) * x[iCell];

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
                        // b[iCell] += matrixBatchElem.m(ic2f + 1) * u[iCellOther]; //!wrong, this matrixBatchElem is Aii^-1Bij
                    }
                    else
                    {
                        // if (faceAttribute.iPhy == BoundaryType::Wall)
                        // {
                        //     Eigen::Vector<real, vsize> uBV;
                        //     uBV.setZero();
                        //     if (!setting.SOR_Instead)
                        //         uRecNewBuf[iCell] +=
                        //             relax * (((uBV - u[iCell]) * vectorBatchElem.m(0).row(ic2f)).transpose());
                        //     else
                        //         uRec[iCell] +=
                        //             relax * (((uBV - u[iCell]) * vectorBatchElem.m(0).row(ic2f)).transpose());

                        //     // eFace.Integration(
                        //     //     BCCorrection,
                        //     //     [&](Eigen::MatrixXd &corInc, int ig, Elem::tPoint &p, Elem::tDiFj &iDiNj)
                        //     //     {
                        //     //         auto &diffI = faceDiBjGaussCache[iFace][ig * 2 + 0]; // must be left

                        //     //         Eigen::MatrixXd rowUD = (uBV - uRec[iCell]).transpose();
                        //     //         Eigen::MatrixXd rowDiffI = diffI.row(0);
                        //     //         FFaceFunctional(iFace,rowDiffI, rowUD, faceWeights[iFace]({0}), corInc);
                        //     //         corInc *= faceNorms[iFace][ig].norm();
                        //     //     });
                        // }
                        // else if (faceAttribute.iPhy == BoundaryType::Farfield)
                        // {
                        //     if (!setting.SOR_Instead)
                        //         uRecNewBuf[iCell] +=
                        //             relax * ((matrixBatchElem.m(ic2f + 1) * uRec[iCell]));
                        //     else
                        //         uRec[iCell] +=
                        //             relax * ((matrixBatchElem.m(ic2f + 1) * uRec[iCell]));
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