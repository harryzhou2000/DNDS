#pragma once
#include "DNDS_Gas.hpp"
#include "DNDS_Mesh.hpp"
#include "DNDS_FV_VR.hpp"
#include "DNDS_ODE.hpp"
#include "DNDS_Scripting.hpp"
#include "DNDS_Linear.hpp"
#include "DNDS_FV_EulerEvaluator.hpp"

#include <iomanip>
#include <functional>

namespace DNDS
{

    template <EulerModel model>
    class EulerSolver
    {
        int nVars;

    public:
        typedef EulerEvaluator<model> TEval;
        static const int nVars_Fixed = TEval::nVars_Fixed;

        static const int dim = TEval::dim;
        static const int gdim = TEval::gdim;
        static const int I4 = TEval::I4;

        typedef typename TEval::TU TU;
        typedef typename TEval::TDiffU TDiffU;
        typedef typename TEval::TJacobianU TJacobianU;
        typedef typename TEval::TVec TVec;
        typedef typename TEval::TMat TMat;

    private:
        MPIInfo mpi;
        std::shared_ptr<CompactFacedMeshSerialRW> mesh;
        std::shared_ptr<ImplicitFiniteVolume2D> fv;
        std::shared_ptr<VRFiniteVolume2D> vfv;

        ArrayDOFV<nVars_Fixed> u, uPoisson, uInc, uIncRHS, uTemp;
        ArrayRecV uRec, uRecNew, uRecNew1, uOld;

        int nOUTS = 9;
        // rho u v w p T M ifUseLimiter RHS
        std::shared_ptr<Array<VarVector>> outDist;
        std::shared_ptr<Array<VarVector>> outSerial;

        // std::vector<uint32_t> ifUseLimiter;
        ArrayLocal<Batch<real, 1>> ifUseLimiter;

    public:
        EulerSolver(const MPIInfo &nmpi) : nVars(getNVars(model)), mpi(nmpi)
        {
            nOUTS = nVars + 4;
        }

        struct Configuration
        {
            int recOrder = 2;
            int nInternalRecStep = 1;
            int nTimeStep = 1000;
            int nConsoleCheck = 10;
            int nConsoleCheckInternal = 1;
            int consoleOutputMode = 0; // 0 for basic, 1 for wall force out
            int nSGSIterationInternal = 0;
            int nDataOut = 10000;
            int nDataOutC = 50;
            int nDataOutInternal = 1;
            int nDataOutCInternal = 1;
            int nTimeStepInternal = 1000;
            real tDataOut = veryLargeReal;
            real tEnd = veryLargeReal;

            real CFL = 0.2;
            real dtImplicit = 1e100;
            real rhsThresholdInternal = 1e-10;

            real meshRotZ = 0;
            std::string mName = "data/mesh/NACA0012_WIDE_H3.msh";
            std::string outPltName = "data/out/debugData_";
            std::string outLogName = "data/out/debugData_";
            real err_dMax = 0.1;

            real res_base = 0;

            VRFiniteVolume2D::Setting vfvSetting;

            int nDropVisScale;
            real vDropVisScale;
            typename EulerEvaluator<model>::Setting eulerSetting;

            int curvilinearOneStep = 500;
            int curvilinearRepeatInterval = 500;
            int curvilinearRepeatNum = 10;

            int curvilinearRestartNstep = 100;
            real curvilinearRange = 0.1;

            bool useLocalDt = true;
            bool useLimiter = true;
            int nPartialLimiterStart = 2;
            int nPartialLimiterStartLocal = 500;
            int nForceLocalStartStep = -1;
            int nCFLRampStart = 1000;
            int nCFLRampLength = 10000;
            real CFLRampEnd = 10;

            int gmresCode = 0; // 0 for lusgs, 1 for gmres, 2 for lusgs started gmres
            int nGmresSpace = 10;
            int nGmresIter = 2;

            int jacobianTypeCode = 0; // 0 for original LUSGS jacobian, 1 for ad roe, 2 for ad roe ad vis

            int nFreezePassiveInner = 0;

        } config;

        void ConfigureFromJson(const std::string &jsonName)
        {
            rapidjson::Document doc;
            JSON::ReadFile(jsonName, doc);
            JSON::ParamParser root(mpi);

            root.AddInt("nInternalRecStep", &config.nInternalRecStep);
            root.AddInt("recOrder", &config.recOrder);
            root.AddInt("nTimeStep", &config.nTimeStep);
            root.AddInt("nTimeStepInternal", &config.nTimeStepInternal);
            root.AddInt("nSGSIterationInternal", &config.nSGSIterationInternal);
            root.AddInt("nConsoleCheck", &config.nConsoleCheck);
            root.AddInt("nConsoleCheckInternal", &config.nConsoleCheckInternal);
            root.AddInt("consoleOutputMode", &config.consoleOutputMode);
            root.AddInt("nDataOutC", &config.nDataOutC);
            root.AddInt("nDataOut", &config.nDataOut);
            root.AddInt("nDataOutCInternal", &config.nDataOutCInternal);
            root.AddInt("nDataOutInternal", &config.nDataOutInternal);
            root.AddDNDS_Real("tDataOut", &config.tDataOut);
            root.AddDNDS_Real("tEnd", &config.tEnd);
            root.AddDNDS_Real("CFL", &config.CFL);
            root.AddDNDS_Real("dtImplicit", &config.dtImplicit);
            root.AddDNDS_Real("rhsThresholdInternal", &config.rhsThresholdInternal);
            root.AddDNDS_Real("meshRotZ", &config.meshRotZ);
            root.Addstd_String("meshFile", &config.mName);
            root.Addstd_String("outLogName", &config.outLogName);
            root.Addstd_String("outPltName", &config.outPltName);
            root.AddDNDS_Real("err_dMax", &config.err_dMax);
            root.AddDNDS_Real("res_base", &config.res_base);
            root.AddBool("useLocalDt", &config.useLocalDt);

            root.AddBool("useLimiter", &config.useLimiter);
            root.AddInt("nPartialLimiterStart", &config.nPartialLimiterStart);
            root.AddInt("nPartialLimiterStartLocal", &config.nPartialLimiterStartLocal);

            root.AddInt("nForceLocalStartStep", &config.nForceLocalStartStep);

            root.AddInt("nCFLRampStart", &config.nCFLRampStart);
            root.AddInt("nCFLRampLength", &config.nCFLRampLength);
            root.AddDNDS_Real("CFLRampEnd", &config.CFLRampEnd);

            root.AddInt("gmresCode", &config.gmresCode);
            root.AddInt("nGmresSpace", &config.nGmresSpace);
            root.AddInt("nGmresIter", &config.nGmresIter);
            root.AddInt("jacobianTypeCode", &config.jacobianTypeCode);

            root.AddInt("nFreezePassiveInner", &config.nFreezePassiveInner);

            JSON::ParamParser vfvParser(mpi);
            root.AddObject("vfvSetting", &vfvParser);
            {
                vfvParser.AddBool("SOR_Instead", &config.vfvSetting.SOR_Instead);
                vfvParser.AddBool("SOR_InverseScanning", &config.vfvSetting.SOR_InverseScanning);
                vfvParser.AddBool("SOR_RedBlack", &config.vfvSetting.SOR_RedBlack);
                vfvParser.AddDNDS_Real("JacobiRelax", &config.vfvSetting.JacobiRelax);
                vfvParser.AddDNDS_Real("tangWeight", &config.vfvSetting.tangWeight);
                vfvParser.AddBool("anistropicLengths", &config.vfvSetting.anistropicLengths);
                vfvParser.AddDNDS_Real("scaleMLargerPortion", &config.vfvSetting.scaleMLargerPortion);
                vfvParser.AddDNDS_Real("farWeight", &config.vfvSetting.farWeight);
                vfvParser.AddDNDS_Real("wallWeight", &config.vfvSetting.wallWeight);
                vfvParser.AddInt("curvilinearOrder", &config.vfvSetting.curvilinearOrder);
                vfvParser.AddDNDS_Real("WBAP_SmoothIndicatorScale", &config.vfvSetting.WBAP_SmoothIndicatorScale);
                vfvParser.AddBool("orthogonalizeBase", &config.vfvSetting.orthogonalizeBase);
                vfvParser.AddBool("normWBAP", &config.vfvSetting.normWBAP);
            }

            root.AddInt("nDropVisScale", &config.nDropVisScale);
            root.AddDNDS_Real("vDropVisScale", &config.vDropVisScale);

            JSON::ParamParser eulerParser(mpi);
            std::string RSName;
            root.AddObject("eulerSetting", &eulerParser);
            {
                eulerParser.Addstd_String(
                    "riemannSolverType", &RSName,
                    [&]()
                    {
                        if (RSName == "Roe")
                            config.eulerSetting.rsType = EulerEvaluator<model>::Setting::RiemannSolverType::Roe;
                        else if (RSName == "HLLC")
                            config.eulerSetting.rsType = EulerEvaluator<model>::Setting::RiemannSolverType::HLLC;
                        else if (RSName == "HLLEP")
                            config.eulerSetting.rsType = EulerEvaluator<model>::Setting::RiemannSolverType::HLLEP;
                        else
                            assert(false);
                    });
                eulerParser.AddInt("nTimeFilterPass", &config.eulerSetting.nTimeFilterPass);
                eulerParser.AddDNDS_Real("visScale", &config.eulerSetting.visScale);
                eulerParser.AddDNDS_Real("visScaleIn", &config.eulerSetting.visScaleIn);
                eulerParser.AddDNDS_Real("ekCutDown", &config.eulerSetting.ekCutDown);
                eulerParser.AddDNDS_Real("isiScale", &config.eulerSetting.isiScale);
                eulerParser.AddDNDS_Real("isiScaleIn", &config.eulerSetting.isiScaleIn);
                eulerParser.AddDNDS_Real("isiCutDown", &config.eulerSetting.isiCutDown);
                eulerParser.AddDNDS_Real("visScale", &config.eulerSetting.visScale);
            }
            JSON::ParamParser eulerGasParser(mpi);
            {
                eulerParser.AddObject("idealGasProperty", &eulerGasParser);
                {
                    eulerGasParser.AddDNDS_Real("gamma", &config.eulerSetting.idealGasProperty.gamma);
                    eulerGasParser.AddDNDS_Real("Rgas", &config.eulerSetting.idealGasProperty.Rgas, [&]()
                                                { 
                                                    real gamma = config.eulerSetting.idealGasProperty.gamma;
                                                    config.eulerSetting.idealGasProperty.CpGas = config.eulerSetting.idealGasProperty.Rgas * gamma/(gamma -1);
                                                    if(mpi.rank == 0)
                                                        std::cout << "\tCpGas = " << config.eulerSetting.idealGasProperty.CpGas << std::endl; });
                    eulerGasParser.AddDNDS_Real("muGas", &config.eulerSetting.idealGasProperty.muGas);
                }
            }
            Eigen::VectorXd eulerSetting_farFieldStaticValueBuf;
            {
                eulerParser.AddEigen_RealVec(
                    "farFieldStaticValue", &eulerSetting_farFieldStaticValueBuf,
                    [&]()
                    {
                        assert(eulerSetting_farFieldStaticValueBuf.size() == nVars);
                        config.eulerSetting.farFieldStaticValue = eulerSetting_farFieldStaticValueBuf;
                    });
            }
            Eigen::VectorXd eulerSetting_boxInitializerValueBuf;
            {
                eulerParser.AddEigen_RealVec(
                    "boxInitializerValue", &eulerSetting_boxInitializerValueBuf,
                    [&]()
                    {
                        assert(eulerSetting_boxInitializerValueBuf.size() % (6 + nVars) == 0);
                        config.eulerSetting.boxInitializers.resize(eulerSetting_boxInitializerValueBuf.size() / (6 + nVars));
                        auto &boxVec = config.eulerSetting.boxInitializers;
                        for (int iInit = 0; iInit < boxVec.size(); iInit++)
                        {
                            boxVec[iInit].x0 = eulerSetting_boxInitializerValueBuf((6 + nVars) * iInit + 0);
                            boxVec[iInit].x1 = eulerSetting_boxInitializerValueBuf((6 + nVars) * iInit + 1);
                            boxVec[iInit].y0 = eulerSetting_boxInitializerValueBuf((6 + nVars) * iInit + 2);
                            boxVec[iInit].y1 = eulerSetting_boxInitializerValueBuf((6 + nVars) * iInit + 3);
                            boxVec[iInit].z0 = eulerSetting_boxInitializerValueBuf((6 + nVars) * iInit + 4);
                            boxVec[iInit].z1 = eulerSetting_boxInitializerValueBuf((6 + nVars) * iInit + 5);
                            boxVec[iInit].v = eulerSetting_boxInitializerValueBuf(
                                Eigen::seq((6 + nVars) * iInit + 6, (6 + nVars) * iInit + 6 + nVars - 1));
                        }
                    });
            }
            Eigen::VectorXd eulerSetting_planeInitializerValueBuf;
            {
                eulerParser.AddEigen_RealVec(
                    "planeInitializerValue", &eulerSetting_planeInitializerValueBuf,
                    [&]()
                    {
                        assert(eulerSetting_planeInitializerValueBuf.size() % (4 + nVars) == 0);
                        config.eulerSetting.planeInitializers.resize(eulerSetting_planeInitializerValueBuf.size() / (4 + nVars));
                        auto &planeVec = config.eulerSetting.planeInitializers;
                        for (int iInit = 0; iInit < planeVec.size(); iInit++)
                        {
                            planeVec[iInit].a = eulerSetting_planeInitializerValueBuf((4 + nVars) * iInit + 0);
                            planeVec[iInit].b = eulerSetting_planeInitializerValueBuf((4 + nVars) * iInit + 1);
                            planeVec[iInit].c = eulerSetting_planeInitializerValueBuf((4 + nVars) * iInit + 2);
                            planeVec[iInit].h = eulerSetting_planeInitializerValueBuf((4 + nVars) * iInit + 3);
                            planeVec[iInit].v = eulerSetting_planeInitializerValueBuf(
                                Eigen::seq((4 + nVars) * iInit + 4, (4 + nVars) * iInit + 4 + nVars - 1));
                        }
                    });
            }
            {
                eulerParser.AddInt(
                    "specialBuiltinInitializer", &config.eulerSetting.specialBuiltinInitializer, []() {}, JSON::ParamParser::FLAG_NULL);
            }
            Eigen::VectorXd eulerSetting_constMassForceValueBuf;
            {
                eulerParser.AddEigen_RealVec(
                    "constMassForce", &eulerSetting_constMassForceValueBuf,
                    [&]()
                    {
                        assert(eulerSetting_constMassForceValueBuf.size() == 3);
                        config.eulerSetting.constMassForce = eulerSetting_constMassForceValueBuf;
                    },
                    JSON::ParamParser::FLAG_NULL);
            }

            root.AddInt("curvilinearOneStep", &config.curvilinearOneStep);
            root.AddInt("curvilinearRestartNstep", &config.curvilinearRestartNstep);
            root.AddInt("curvilinearRepeatInterval", &config.curvilinearRepeatInterval);
            root.AddInt("curvilinearRepeatNum", &config.curvilinearRepeatNum);
            root.AddDNDS_Real("curvilinearRange", &config.curvilinearRange);

            root.Parse(doc.GetObject(), 0);

            if (mpi.rank == 0)
                log() << "JSON: Parse Done ===" << std::endl;

            if (doc["vfvSetting"].IsObject())
            {
                if (doc["vfvSetting"]["baseCenterType"].IsString())
                {
                    std::string centerOpt = doc["vfvSetting"]["baseCenterType"].GetString();
                    config.vfvSetting.baseCenterTypeName = centerOpt;
                    if (centerOpt == "Param")
                        config.vfvSetting.baseCenterType = VRFiniteVolume2D::Setting::BaseCenterType::Paramcenter;
                    else if (centerOpt == "Bary")
                        config.vfvSetting.baseCenterType = VRFiniteVolume2D::Setting::BaseCenterType::Barycenter;
                    else
                        assert(false);
                    if (mpi.rank == 0)
                        log() << "JSON: vfvSetting.baseCenterType = " << config.vfvSetting.baseCenterTypeName << std::endl;
                }

                if (doc["vfvSetting"]["weightSchemeGeom"].IsString())
                {
                    std::string centerOpt = doc["vfvSetting"]["weightSchemeGeom"].GetString();
                    config.vfvSetting.weightSchemeGeomName = centerOpt;
                    if (centerOpt == "None")
                        config.vfvSetting.weightSchemeGeom = VRFiniteVolume2D::Setting::WeightSchemeGeom::None;
                    else if (centerOpt == "D")
                        config.vfvSetting.weightSchemeGeom = VRFiniteVolume2D::Setting::WeightSchemeGeom::D;
                    else if (centerOpt == "S")
                        config.vfvSetting.weightSchemeGeom = VRFiniteVolume2D::Setting::WeightSchemeGeom::S;
                    else
                        assert(false);
                    if (mpi.rank == 0)
                        log() << "JSON: vfvSetting.weightSchemeGeom = " << config.vfvSetting.weightSchemeGeomName << std::endl;
                }
            }
        }

        void ReadMeshAndInitialize()
        {
            // Debug::MPIDebugHold(mpi);
            CompactFacedMeshSerialRWBuild(mpi, config.mName, "data/out/debugmeshSO.plt", mesh, config.meshRotZ);
            fv = std::make_shared<ImplicitFiniteVolume2D>(mesh.get());
            vfv = std::make_shared<VRFiniteVolume2D>(mesh.get(), fv.get(), config.recOrder);
            vfv->setting = config.vfvSetting; //* currently only copies, could upgrade to referencing
            vfv->Initialization();

            fv->BuildMean(u, nVars);
            fv->BuildMean(uPoisson, nVars);
            fv->BuildMean(uInc, nVars);
            fv->BuildMean(uIncRHS, nVars);
            fv->BuildMean(uTemp, nVars);
            vfv->BuildRec(uRec, nVars);
            vfv->BuildRec(uRecNew, nVars);
            vfv->BuildRec(uRecNew1, nVars);
            vfv->BuildRec(uOld, nVars);

            Eigen::VectorXd initConstVal = config.eulerSetting.farFieldStaticValue;
            u.setConstant(initConstVal);
            if (model == EulerModel::NS_SA)
            {
                for (int iCell = 0; iCell < u.dist->size(); iCell++)
                {
                    auto c2f = mesh->cell2faceLocal[iCell];
                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                    {
                        index iFace = c2f[ic2f];
                        if (mesh->faceAtrLocal[iFace][0].iPhy == BoundaryType::Wall_NoSlip)
                            u[iCell](I4 + 1) *= 0.6;
                    }
                }
            }

            uPoisson.setConstant(0.0);

            //! serial mesh specific output method
            using outDistInitContextType = typename decltype(outDist)::element_type::tContext;
            outDist = std::make_shared<typename decltype(outDist)::element_type>(
                outDistInitContextType([&](index)
                                       { return nOUTS; },
                                       mesh->cell2faceLocal.dist->size()),
                mpi);
            outSerial = std::make_shared<typename decltype(outDist)::element_type>(outDist.get());
            outSerial->BorrowGGIndexing(*mesh->cell2node);
            outSerial->createMPITypes();
            outSerial->initPersistentPull();

            // Box
            for (auto &i : config.eulerSetting.boxInitializers)
            {
                for (index iCell = 0; iCell < u.dist->size(); iCell++)
                {
                    Elem::tPoint &pos = vfv->cellBaries[iCell];
                    if (pos(0) > i.x0 && pos(0) < i.x1 &&
                        pos(1) > i.y0 && pos(1) < i.y1 &&
                        pos(2) > i.z0 && pos(2) < i.z1)
                    {
                        u[iCell] = i.v;
                    }
                }
            }

            // Plane
            for (auto &i : config.eulerSetting.planeInitializers)
            {
                for (index iCell = 0; iCell < u.dist->size(); iCell++)
                {
                    Elem::tPoint &pos = vfv->cellBaries[iCell];
                    if (pos(0) * i.a + pos(1) * i.b + pos(2) * i.c + i.h > 0)
                    {
                        // std::cout << pos << std::endl << i.a << i.b << std::endl << i.h <<std::endl;
                        // assert(false);
                        u[iCell] = i.v;
                    }
                }
            }

            switch (config.eulerSetting.specialBuiltinInitializer)
            {
            case 1: // for RT problem
                assert(model == NS || model == NS_2D);
                if constexpr (model == NS || model == NS_2D)
                    for (index iCell = 0; iCell < u.dist->size(); iCell++)
                    {
                        Elem::tPoint &pos = vfv->cellBaries[iCell];
                        real gamma = config.eulerSetting.idealGasProperty.gamma;
                        real rho = 2;
                        real p = 1 + 2 * pos(1);
                        if (pos(1) >= 0.5)
                        {
                            rho = 1;
                            p = 1.5 + pos(1);
                        }
                        real v = -0.025 * sqrt(gamma * p / rho) * std::cos(8 * pi * pos(0));
                        if constexpr (dim == 3)
                            u[iCell] = Eigen::Vector<real, 5>{rho, 0, rho * v, 0, 0.5 * rho * sqr(v) + p / (gamma - 1)};
                        else
                            u[iCell] = Eigen::Vector<real, 4>{rho, 0, rho * v, 0.5 * rho * sqr(v) + p / (gamma - 1)};
                    }
                break;
            case 0:
                break;
            default:
                log() << "Wrong specialBuiltinInitializer" << std::endl;
                assert(false);
                break;
            }

            vfv->BuildIfUseLimiter(ifUseLimiter);
        }

        void RunImplicitEuler()
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            InsertCheck(mpi, "Implicit 1 nvars " + std::to_string(nVars));

            // ODE::ImplicitSDIRK4DualTimeStep<decltype(u)> ode(
            //     u.dist->size(),
            //     [&](decltype(u) &data)
            //     {
            //         data.resize(u.dist->size(), u.dist->getMPI(), nVars);
            //         data.CreateGhostCopyComm(mesh->cell2faceLocal);
            //         data.InitPersistentPullClean();
            //     });
            ODE::ImplicitBDFDualTimeStep<decltype(u)> ode(
                u.dist->size(),
                [&](decltype(u) &data)
                {
                    data.resize(u.dist->size(), u.dist->getMPI(), nVars);
                    data.CreateGhostCopyComm(mesh->cell2faceLocal);
                    data.InitPersistentPullClean();
                },
                2);

            Linear::GMRES_LeftPreconditioned<decltype(u)> gmres(
                config.nGmresSpace,
                [&](decltype(u) &data)
                {
                    data.resize(u.dist->size(), u.dist->getMPI(), nVars);
                    data.CreateGhostCopyComm(mesh->cell2faceLocal);
                    data.InitPersistentPullClean();
                });

            EulerEvaluator<model> eval(mesh.get(), fv.get(), vfv.get());

            std::ofstream logErr(config.outLogName + ".log");
            eval.settings = config.eulerSetting;
            // std::cout << uF0.dist->commStat.hasPersistentPullReqs << std::endl;
            // exit(0);
            // uRec.InitPersistentPullClean();
            // u.InitPersistentPullClean();
            // uInc.InitPersistentPullClean();
            // uF0.InitPersistentPullClean();
            // u.StartPersistentPullClean();
            double tstart = MPI_Wtime();
            double trec{0}, tcomm{0}, trhs{0}, tLim{0};
            int stepCount = 0;
            Eigen::Vector<real, -1> resBaseC;
            Eigen::Vector<real, -1> resBaseCInternal;
            resBaseC.resize(nVars);
            resBaseCInternal.resize(nVars);
            resBaseC.setConstant(config.res_base);

            // Doing Poisson Init:

            int curvilinearNum = 0;
            int curvilinearStepper = 0;

            real tsimu = 0.0;
            real nextTout = config.tDataOut;
            int nextStepOut = config.nDataOut;
            int nextStepOutC = config.nDataOutC;
            PerformanceTimer::Instance().clearAllTimer();

            // *** Loop variables
            real CFLNow = config.CFL;
            bool ifOutT = false;
            real curDtMin;
            real curDtImplicit = config.dtImplicit;
            int step;

            InsertCheck(mpi, "Implicit 2 nvars " + std::to_string(nVars));
            /*******************************************************/
            /*                   DEFINE LAMBDAS                    */
            /*******************************************************/
            auto frhs = [&](ArrayDOFV<nVars_Fixed> &crhs, ArrayDOFV<nVars_Fixed> &cx, int iter, real ct)
            {
                eval.FixUMaxFilter(cx);
                // cx.StartPersistentPullClean();
                // cx.WaitPersistentPullClean();

                // for (index iCell = 0; iCell < uOld.size(); iCell++)
                //     uOld[iCell].m() = uRec[iCell].m();

                InsertCheck(mpi, " Lambda RHS: StartRec");
                for (int iRec = 0; iRec < config.nInternalRecStep; iRec++)
                {
                    double tstartA = MPI_Wtime();
                    vfv->ReconstructionJacobiStep<dim, nVars_Fixed>(cx, uRec, uRecNew);
                    trec += MPI_Wtime() - tstartA;

                    uRec.StartPersistentPullClean();
                    uRec.WaitPersistentPullClean();
                }
                double tstartH = MPI_Wtime();

                // for (index iCell = 0; iCell < uOld.size(); iCell++)
                //     uRec[iCell].m() -= uOld[iCell].m();

                InsertCheck(mpi, " Lambda RHS: StartLim");
                if (config.useLimiter)
                {
                    // vfv->ReconstructionWBAPLimitFacial(
                    //     cx, uRec, uRecNew, uF0, uF1, ifUseLimiter,
                    vfv->ReconstructionWBAPLimitFacialV3<dim, nVars_Fixed>(
                        cx, uRec, uRecNew, uRecNew1, ifUseLimiter,
                        iter < config.nPartialLimiterStartLocal && step < config.nPartialLimiterStart,
                        [&](const auto &UL, const auto &UR, const auto &n) -> auto{
                            PerformanceTimer::Instance().StartTimer(PerformanceTimer::LimiterA);
                            Eigen::Vector<real, I4 + 1> UC = (UL + UR)(Seq01234)*0.5;
                            auto normBase = Elem::NormBuildLocalBaseV<dim>(n(Seq012));
                            UC(Seq123) = normBase.transpose() * UC(Seq123);

                            auto M = Gas::IdealGas_EulerGasLeftEigenVector<dim>(UC, eval.settings.idealGasProperty.gamma);
                            M(Eigen::all, Seq123) *= normBase.transpose();

                            Eigen::Matrix<real, nVars_Fixed, nVars_Fixed> ret(nVars, nVars);
                            ret.setIdentity();
                            ret(Seq01234, Seq01234) = M;
                            PerformanceTimer::Instance().EndTimer(PerformanceTimer::LimiterA);
                            return ret;
                            // return Eigen::Matrix<real, 5, 5>::Identity();
                        },
                        [&](const auto &UL, const auto &UR, const auto &n) -> auto{
                            PerformanceTimer::Instance().StartTimer(PerformanceTimer::LimiterA);
                            Eigen::Vector<real, I4 + 1> UC = (UL + UR)(Seq01234)*0.5;
                            auto normBase = Elem::NormBuildLocalBaseV<dim>(n(Seq012));
                            UC(Seq123) = normBase.transpose() * UC(Seq123);

                            // real ekFixRatio = 0.001;
                            // Eigen::Vector3d velo = UC({1, 2, 3}) / UC(0);
                            // real vsqr = velo.squaredNorm();
                            // real Ek = vsqr * 0.5 * UC(0);
                            // real Efix = Ek * ekFixRatio;
                            // real e = UC(4) - Ek;
                            // if (e < 0)
                            //     e = 0.5 * Efix;
                            // else if (e < Efix)
                            //     e = (e * e + Efix * Efix) / (2 * Efix);
                            // UC(4) = Ek + e;

                            auto M = Gas::IdealGas_EulerGasRightEigenVector<dim>(UC, eval.settings.idealGasProperty.gamma);
                            M(Seq123, Eigen::all) = normBase * M(Seq123, Eigen::all);

                            Eigen::Matrix<real, nVars_Fixed, nVars_Fixed> ret(nVars, nVars);
                            ret.setIdentity();
                            ret(Seq01234, Seq01234) = M;

                            PerformanceTimer::Instance().EndTimer(PerformanceTimer::LimiterA);
                            return ret;
                            // return Eigen::Matrix<real, 5, 5>::Identity();
                        });
                    // uRecNew.StartPersistentPullClean();
                    // uRecNew.WaitPersistentPullClean();
                }
                tLim += MPI_Wtime() - tstartH;

                // uRec.StartPersistentPullClean(); //! this also need to update!
                // uRec.WaitPersistentPullClean();

                // }

                InsertCheck(mpi, " Lambda RHS: StartEval");
                double tstartE = MPI_Wtime();
                eval.setPassiveDiscardSource(iter <= 0);
                if (config.useLimiter)
                    eval.EvaluateRHS(crhs, cx, uRecNew, tsimu + ct * curDtImplicit);
                else
                    eval.EvaluateRHS(crhs, cx, uRec, tsimu + ct * curDtImplicit);
                if (getNVars(model) > (I4 + 1) && iter <= config.nFreezePassiveInner)
                {
                    for (int i = 0; i < crhs.size(); i++)
                        crhs[i](Eigen::seq(I4 + 1, Eigen::last)).setZero();
                    // if (mpi.rank == 0)
                    //     std::cout << "Freezing all passive" << std::endl;
                }
                trhs += MPI_Wtime() - tstartE;

                InsertCheck(mpi, " Lambda RHS: End");
            };

            auto fdtau = [&](std::vector<real> &dTau, real alphaDiag)
            {
                eval.FixUMaxFilter(u);
                u.StartPersistentPullClean(); //! this also need to update!
                u.WaitPersistentPullClean();
                // uRec.StartPersistentPullClean();
                // uRec.WaitPersistentPullClean();

                eval.EvaluateDt(dTau, u, CFLNow, curDtMin, 1e100, config.useLocalDt);
                for (auto &i : dTau)
                    i /= alphaDiag;
            };

            auto fsolve = [&](ArrayDOFV<nVars_Fixed> &cx, ArrayDOFV<nVars_Fixed> &crhs, std::vector<real> &dTau,
                              real dt, real alphaDiag, ArrayDOFV<nVars_Fixed> &cxInc, int iter)
            {
                cxInc.setConstant(0.0);

                if (config.jacobianTypeCode != 0)
                {
                    eval.LUSGSADMatrixInit(dTau, dt, alphaDiag, cx, config.jacobianTypeCode, tsimu);
                }
                else
                {
                    if (config.useLimiter) // uses urec value
                        eval.LUSGSMatrixInit(dTau, dt, alphaDiag,
                                             cx, uRecNew,
                                             config.jacobianTypeCode,
                                             tsimu);
                    else
                        eval.LUSGSMatrixInit(dTau, dt, alphaDiag,
                                             cx, uRec,
                                             config.jacobianTypeCode,
                                             tsimu);
                }

                if (config.gmresCode == 0 || config.gmresCode == 2)
                {
                    // //! LUSGS
                    if (config.jacobianTypeCode == 0)
                    {
                        eval.UpdateLUSGSForward(alphaDiag, crhs, cx, cxInc, cxInc);
                        // cxInc.StartPersistentPullClean();
                        // cxInc.WaitPersistentPullClean();
                        eval.UpdateLUSGSBackward(alphaDiag, crhs, cx, cxInc, cxInc);
                        cxInc.StartPersistentPullClean();
                        cxInc.WaitPersistentPullClean();
                        for (int iIter = 1; iIter <= config.nSGSIterationInternal; iIter++)
                        {
                            cxInc.StartPersistentPullClean();
                            cxInc.WaitPersistentPullClean();
                            eval.UpdateSGS(alphaDiag, crhs, cx, cxInc, cxInc, true);
                            cxInc.StartPersistentPullClean();
                            cxInc.WaitPersistentPullClean();
                            eval.UpdateSGS(alphaDiag, crhs, cx, cxInc, cxInc, false);
                        }
                    }
                    else
                    {
                        eval.UpdateLUSGSADForward(crhs, cx, cxInc, cxInc);
                        cxInc.StartPersistentPullClean();
                        cxInc.WaitPersistentPullClean();
                        eval.UpdateLUSGSADBackward(crhs, cx, cxInc, cxInc);
                        cxInc.StartPersistentPullClean();
                        cxInc.WaitPersistentPullClean();
                    }
                }

                if (config.gmresCode != 0)
                {
                    // !  GMRES
                    // !  for gmres solver: A * uinc = rhsinc, rhsinc is average value insdead of cumulated on vol
                    gmres.solve(
                        [&](decltype(u) &x, decltype(u) &Ax)
                        {
                            if (config.jacobianTypeCode == 0)
                                eval.LUSGSMatrixVec(alphaDiag, cx, x, Ax);
                            else
                                eval.LUSGSADMatrixVec(cx, x, Ax);

                            Ax.StartPersistentPullClean();
                            Ax.WaitPersistentPullClean();
                        },
                        [&](decltype(u) &x, decltype(u) &MLx)
                        {
                            // x as rhs, and MLx as uinc

                            if (config.jacobianTypeCode == 0)
                            {
                                eval.UpdateLUSGSForward(alphaDiag, x, cx, MLx, MLx);
                                MLx.StartPersistentPullClean();
                                MLx.WaitPersistentPullClean();
                                eval.UpdateLUSGSBackward(alphaDiag, x, cx, MLx, MLx);
                                MLx.StartPersistentPullClean();
                                MLx.WaitPersistentPullClean();
                            }
                            else
                            {
                                for (index iCell = 0; iCell < x.dist->size(); iCell++) // ad series now takes rhs with volume
                                {
                                    uTemp[iCell] = x[iCell] * eval.fv->volumeLocal[iCell];
                                }
                                eval.UpdateLUSGSADForward(uTemp, cx, MLx, MLx);
                                MLx.StartPersistentPullClean();
                                MLx.WaitPersistentPullClean();
                                eval.UpdateLUSGSADBackward(uTemp, cx, MLx, MLx);
                                MLx.StartPersistentPullClean();
                                MLx.WaitPersistentPullClean();
                            }
                        },
                        crhs, cxInc, config.nGmresIter,
                        [&](uint32_t i, real res, real resB) -> bool
                        {
                            if (i > 0)
                            {
                                if (mpi.rank == 0)
                                {
                                    // log() << std::scientific;
                                    // log() << "GMRES: " << i << " " << resB << " -> " << res << std::endl;
                                }
                            }
                            return false;
                        });
                    for (index iCell = 0; iCell < cxInc.size(); iCell++)
                        cxInc[iCell] = eval.CompressInc(cx[iCell], cxInc[iCell], crhs[iCell]); // manually add fixing for gmres results
                }
                // !freeze something
                if (getNVars(model) > I4 + 1 && iter <= config.nFreezePassiveInner)
                {
                    for (int i = 0; i < crhs.size(); i++)
                        cxInc[i](Eigen::seq(I4 + 1, Eigen::last)).setZero();
                    // if (mpi.rank == 0)
                    //     std::cout << "Freezing all passive" << std::endl;
                }
            };

            auto fstop = [&](int iter, ArrayDOFV<nVars_Fixed> &cxinc, int iStep) -> bool
            {
                Eigen::Vector<real, -1> res(nVars);
                eval.EvaluateResidual(res, cxinc);
                // if (iter == 1 && iStep == 1) // * using 1st rk step for reference
                if (iter == 1)
                    resBaseCInternal = res;
                else
                    resBaseCInternal = resBaseCInternal.array().max(res.array()); //! using max !
                Eigen::Vector<real, -1> resRel = (res.array() / resBaseCInternal.array()).matrix();
                bool ifStop = resRel(0) < config.rhsThresholdInternal; // ! using only rho's residual
                if (iter % config.nConsoleCheckInternal == 0 || iter > config.nTimeStepInternal || ifStop)
                {
                    double telapsed = MPI_Wtime() - tstart;
                    if (mpi.rank == 0)
                    {
                        tcomm = PerformanceTimer::Instance().getTimer(PerformanceTimer::Comm);
                        auto fmt = log().flags();
                        log() << std::setprecision(3) << std::scientific
                              << "\t Internal === Step [" << iStep << ", " << iter << "]   "
                              << "res \033[91m[" << resRel.transpose() << "]\033[39m   "
                              << "t,dTaumin,CFL \033[92m[" << tsimu << ", " << curDtMin << ", " << CFLNow << "]\033[39m   "
                              << std::setprecision(3) << std::fixed
                              << "Time [" << telapsed << "]   recTime ["
                              << trec << "]   rhsTime ["
                              << trhs << "]   commTime ["
                              << tcomm << "]  limTime ["
                              << tLim << "]  limtimeA ["
                              << PerformanceTimer::Instance().getTimer(PerformanceTimer::LimiterA) << "]  limtimeB ["
                              << PerformanceTimer::Instance().getTimer(PerformanceTimer::LimiterB) << "]  ";
                        if (config.consoleOutputMode == 1)
                        {
                            log() << std::setprecision(4) << std::setw(10) << std::scientific
                                  << "Wall Flux \033[93m[" << eval.fluxWallSum.transpose() << "]\033[39m";
                        }
                        log() << std::endl;
                        log().setf(fmt);
                        logErr << step << "\t" << iter << "\t" << std::setprecision(9) << std::scientific
                               << res.transpose() << " "
                               << tsimu << "\t" << curDtMin << "\t" << eval.fluxWallSum.transpose() << std::endl;
                    }
                    tstart = MPI_Wtime();
                    trec = tcomm = trhs = tLim = 0.;
                    PerformanceTimer::Instance().clearAllTimer();
                }

                if (iter % config.nDataOutInternal == 0)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + "_" + std::to_string(step) + "_" + std::to_string(iter) + ".plt", ode);
                    nextStepOut += config.nDataOut;
                }
                if (iter % config.nDataOutCInternal == 0)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + "_" + "C" + ".plt", ode);
                    nextStepOutC += config.nDataOutC;
                }
                if (iter >= config.nCFLRampStart && iter <= config.nCFLRampLength + config.nCFLRampStart)
                {
                    real inter = real(iter - config.nCFLRampStart) / config.nCFLRampLength;
                    real logCFL = std::log(config.CFL) + (std::log(config.CFLRampEnd / config.CFL) * inter);
                    CFLNow = std::exp(logCFL);
                }

                // return resRel.maxCoeff() < config.rhsThresholdInternal;
                return ifStop;
            };

            // fmainloop gets the time-variant residual norm,
            // handles the output / log nested loops,
            // integrates physical time tsimu
            // and finally decides if break time loop
            auto fmainloop = [&]() -> bool
            {
                tsimu += curDtImplicit;
                if (ifOutT)
                    tsimu = nextTout;
                Eigen::Vector<real, -1> res(nVars);
                eval.EvaluateResidual(res, ode.rhsbuf[0]);
                if (stepCount == 0 && resBaseC.norm() == 0)
                    resBaseC = res;

                if (step % config.nConsoleCheck == 0)
                {
                    double telapsed = MPI_Wtime() - tstart;
                    if (mpi.rank == 0)
                    {
                        tcomm = PerformanceTimer::Instance().getTimer(PerformanceTimer::Comm);
                        auto fmt = log().flags();
                        log() << std::setprecision(3) << std::scientific
                              << "=== Step [" << step << "]   "
                              << "res \033[91m[" << (res.array() / resBaseC.array()).transpose() << "]\033[39m   "
                              << "t,dt(min) \033[92m[" << tsimu << ", " << curDtMin << "]\033[39m   "
                              << std::setprecision(3) << std::fixed
                              << "Time [" << telapsed << "]   recTime [" << trec << "]   rhsTime [" << trhs << "]   commTime [" << tcomm << "]  limTime [" << tLim << "]  " << std::endl;
                        log().setf(fmt);
                        logErr << step << "\t" << std::setprecision(9) << std::scientific
                               << res.transpose() << " "
                               << tsimu << " " << curDtMin << std::endl;
                    }
                    tstart = MPI_Wtime();
                    trec = tcomm = trhs = tLim = 0.;
                    PerformanceTimer::Instance().clearAllTimer();
                }
                if (step == nextStepOut)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + "_" + std::to_string(step) + ".plt", ode);
                    nextStepOut += config.nDataOut;
                }
                if (step == nextStepOutC)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + "_" + "C" + ".plt", ode);
                    nextStepOutC += config.nDataOutC;
                }
                if (ifOutT)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + "_" + "t_" + std::to_string(nextTout) + ".plt", ode);
                    nextTout += config.tDataOut;
                    if (nextTout > config.tEnd)
                        nextTout = config.tEnd;
                }

                stepCount++;

                return tsimu >= config.tEnd;
            };

            /**********************************/
            /*           MAIN LOOP            */
            /**********************************/

            for (step = 1; step <= config.nTimeStep; step++)
            {
                InsertCheck(mpi, "Implicit Step");
                ifOutT = false;
                curDtImplicit = config.dtImplicit; //* could add CFL driven dt here
                if (tsimu + curDtImplicit > nextTout)
                {
                    ifOutT = true;
                    curDtImplicit = (nextTout - tsimu);
                }
                CFLNow = config.CFL;
                ode.Step(
                    u, uInc,
                    frhs,
                    fdtau,
                    fsolve,
                    config.nTimeStepInternal,
                    fstop,
                    curDtImplicit + verySmallReal);

                if (fmainloop())
                    break;
            }

            // u.WaitPersistentPullClean();
            logErr.close();
        }

        template <typename tODE>
        void PrintData(const std::string &fname, tODE &ode)
        {

            for (int iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
                DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
                TU recu =
                    vfv->cellDiBjCenterBatch->operator[](iCell).m(0)({0}, Eigen::all).rightCols(uRec[iCell].rows()) *
                    uRec[iCell];
                // recu += u[iCell];
                // assert(recu(0) > 0);
                // recu = EulerEvaluator::CompressRecPart(u[iCell], recu);
                recu = u[iCell] + recu * 0;
                TVec velo = (recu(Seq123).array() / recu(0)).matrix();
                real vsqr = velo.squaredNorm();
                real asqr, p, H;
                Gas::IdealGasThermal(recu(I4), recu(0), vsqr, config.eulerSetting.idealGasProperty.gamma, p, asqr, H);
                // assert(asqr > 0);
                real M = std::sqrt(vsqr / asqr);
                real T = p / recu(0) / config.eulerSetting.idealGasProperty.Rgas;

                (*outDist)[iCell][0] = recu(0);
                for (int i = 0; i < dim; i++)
                    (*outDist)[iCell][i + 1] = velo(i);
                (*outDist)[iCell][I4 + 0] = p;
                (*outDist)[iCell][I4 + 1] = T;
                (*outDist)[iCell][I4 + 2] = M;
                // (*outDist)[iCell][7] = (bool)(ifUseLimiter[iCell] & 0x0000000FU);
                (*outDist)[iCell][I4 + 3] = ifUseLimiter[iCell][0] / config.vfvSetting.WBAP_SmoothIndicatorScale;
                // std::cout << iCell << ode.rhsbuf[0][iCell] << std::endl;
                (*outDist)[iCell][I4 + 4] = ode.rhsbuf[0][iCell](0);
                // (*outDist)[iCell][8] = (*vfv->SOR_iCell2iScan)[iCell];//!using SOR rb seq instead

                for (int i = I4 + 1; i < nVars; i++)
                {
                    (*outDist)[iCell][4 + i] = recu(i) / recu(0); // 4 is additional amount offset, not Index of last flow variable (I4)
                }
            }
            outSerial->startPersistentPull();
            outSerial->waitPersistentPull();
            std::vector<std::string> names{
                "R", "U", "V", "W", "P", "T", "M", "ifUseLimiter", "RHSr"};
            for (int i = I4 + 1; i < nVars; i++)
            {
                names.push_back("V" + std::to_string(i - I4));
            }
            mesh->PrintSerialPartPltBinaryDataArray(
                fname, 0, nOUTS, //! oprank = 0
                [&](int idata)
                { return names[idata]; },
                [&](int idata, index iv)
                {
                    return (*outSerial)[iv][idata];
                },
                0);
        }
    };

}