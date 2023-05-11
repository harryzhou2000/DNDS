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

        std::shared_ptr<rapidjson::Document> config_doc;
        std::string output_stamp = "";

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
            bool uniqueStamps = true;
            real err_dMax = 0.1;

            real res_base = 0;
            bool useVolWiseResidual = false;

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
            int limiterProcedure = 0; // 0 for V2==3WBAP, 1 for V3==CWBAP

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

            bool steadyQuit = false;

            int odeCode = 0;

            int runningMode = 0;

        } config;

        void ConfigureFromJson(const std::string &jsonName)
        {
            config_doc = std::make_shared<rapidjson::Document>();
            JSON::ReadFile(jsonName, *config_doc);
            JSON::ParamParser root(mpi);

            root.AddInt(
                "runningMode", &config.runningMode, []() {}, JSON::ParamParser::FLAG_NULL);

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
            root.AddBool(
                "uniqueStamps", &config.uniqueStamps, []() {}, JSON::ParamParser::FLAG_NULL);
            root.AddDNDS_Real("err_dMax", &config.err_dMax);
            root.AddDNDS_Real("res_base", &config.res_base);
            root.AddBool(
                "useVolWiseResidual", &config.useVolWiseResidual, []() {}, JSON::ParamParser::FLAG_NULL);

            root.AddBool("useLocalDt", &config.useLocalDt);

            root.AddBool("useLimiter", &config.useLimiter);
            root.AddInt(
                "limiterProcedure", &config.limiterProcedure, []() {}, JSON::ParamParser::FLAG_NULL);

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
                vfvParser.AddDNDS_Real(
                    "tangWeightModMin", &config.vfvSetting.tangWeightModMin, []() {}, JSON::ParamParser::FLAG_NULL);
                vfvParser.AddInt(
                    "tangWeightModLoc", &config.vfvSetting.tangWeightModLoc, []() {}, JSON::ParamParser::FLAG_NULL);
                vfvParser.AddDNDS_Real(
                    "tangWeightModPower", &config.vfvSetting.tangWeightModPower, []() {}, JSON::ParamParser::FLAG_NULL);
                vfvParser.AddBool(
                    "useLocalCoord", &config.vfvSetting.useLocalCoord, []() {}, JSON::ParamParser::FLAG_NULL);
                vfvParser.AddDNDS_Real(
                    "weightLenScale", &config.vfvSetting.weightLenScale, []() {}, JSON::ParamParser::FLAG_NULL);
                vfvParser.AddBool("anisotropicLengths", &config.vfvSetting.anisotropicLengths);
                vfvParser.AddDNDS_Real("scaleMLargerPortion", &config.vfvSetting.scaleMLargerPortion);
                vfvParser.AddDNDS_Real("farWeight", &config.vfvSetting.farWeight);
                vfvParser.AddDNDS_Real("wallWeight", &config.vfvSetting.wallWeight);
                vfvParser.AddInt("curvilinearOrder", &config.vfvSetting.curvilinearOrder);
                vfvParser.AddDNDS_Real("WBAP_SmoothIndicatorScale", &config.vfvSetting.WBAP_SmoothIndicatorScale);
                vfvParser.AddBool("orthogonalizeBase", &config.vfvSetting.orthogonalizeBase);
                vfvParser.AddBool("normWBAP", &config.vfvSetting.normWBAP);
            }
            std::string centerOpt, weightOpt, tangWeightDirectionOpt;
            {
                vfvParser.Addstd_String(
                    "baseCenterType", &centerOpt,
                    [&]()
                    {
                        config.vfvSetting.baseCenterTypeName = centerOpt;
                        if (centerOpt == "Param")
                            config.vfvSetting.baseCenterType = VRFiniteVolume2D::Setting::BaseCenterType::Paramcenter;
                        else if (centerOpt == "Bary")
                            config.vfvSetting.baseCenterType = VRFiniteVolume2D::Setting::BaseCenterType::Barycenter;
                        else
                            DNDS_assert(false);
                        if (mpi.rank == 0)
                            log() << "JSON: vfvSetting.baseCenterType = " << config.vfvSetting.baseCenterTypeName << std::endl;
                    });
                vfvParser.Addstd_String(
                    "weightSchemeGeom", &weightOpt,
                    [&]()
                    {
                        config.vfvSetting.weightSchemeGeomName = weightOpt;
                        if (weightOpt == "None")
                            config.vfvSetting.weightSchemeGeom = VRFiniteVolume2D::Setting::WeightSchemeGeom::NoneGeom;
                        else if (weightOpt == "D")
                            config.vfvSetting.weightSchemeGeom = VRFiniteVolume2D::Setting::WeightSchemeGeom::D;
                        else if (weightOpt == "S")
                            config.vfvSetting.weightSchemeGeom = VRFiniteVolume2D::Setting::WeightSchemeGeom::S;
                        else if (weightOpt == "SDHQM")
                            config.vfvSetting.weightSchemeGeom = VRFiniteVolume2D::Setting::WeightSchemeGeom::SDHQM;
                        else
                            DNDS_assert(false);
                        if (mpi.rank == 0)
                            log() << "JSON: vfvSetting.weightSchemeGeom = " << config.vfvSetting.weightSchemeGeomName << std::endl;
                    });
                vfvParser.Addstd_String(
                    "weightSchemeDir", &weightOpt,
                    [&]()
                    {
                        config.vfvSetting.weightSchemeDirName = weightOpt;
                        if (weightOpt == "None")
                            config.vfvSetting.weightSchemeDir = VRFiniteVolume2D::Setting::WeightSchemeDir::NoneDir;
                        else if (weightOpt == "optHQM")
                            config.vfvSetting.weightSchemeDir = VRFiniteVolume2D::Setting::WeightSchemeDir::OPTHQM;
                        else
                            DNDS_assert(false);
                        if (mpi.rank == 0)
                            log() << "JSON: vfvSetting.weightSchemeDir = " << config.vfvSetting.weightSchemeDirName << std::endl;
                    },
                    JSON::ParamParser::FLAG_NULL);
                vfvParser.Addstd_String(
                    "tangWeightDirection", &tangWeightDirectionOpt,
                    [&]()
                    {
                        config.vfvSetting.tangWeightDirectionName = tangWeightDirectionOpt;
                        if (tangWeightDirectionOpt == "Bary")
                            config.vfvSetting.tangWeightDirection = VRFiniteVolume2D::Setting::TangWeightDirection::TWD_Bary;
                        else if (tangWeightDirectionOpt == "Norm")
                            config.vfvSetting.tangWeightDirection = VRFiniteVolume2D::Setting::TangWeightDirection::TWD_Norm;
                        else
                            DNDS_assert(false);
                        if (mpi.rank == 0)
                            log() << "JSON: vfvSetting.tangWeightDirection = " << config.vfvSetting.tangWeightDirectionName << std::endl;
                    },
                    JSON::ParamParser::FLAG_NULL);
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
                        else if (RSName == "Roe_M1")
                            config.eulerSetting.rsType = EulerEvaluator<model>::Setting::RiemannSolverType::Roe_M1;
                        else if (RSName == "Roe_M2")
                            config.eulerSetting.rsType = EulerEvaluator<model>::Setting::RiemannSolverType::Roe_M2;
                        else if (RSName == "Roe_M3")
                            config.eulerSetting.rsType = EulerEvaluator<model>::Setting::RiemannSolverType::Roe_M3;
                        else if (RSName == "Roe_M4")
                            config.eulerSetting.rsType = EulerEvaluator<model>::Setting::RiemannSolverType::Roe_M4;
                        else if (RSName == "Roe_M5")
                            config.eulerSetting.rsType = EulerEvaluator<model>::Setting::RiemannSolverType::Roe_M5;
                        else
                            DNDS_assert(false);
                    });
                eulerParser.AddInt("nTimeFilterPass", &config.eulerSetting.nTimeFilterPass);
                eulerParser.AddDNDS_Real("visScale", &config.eulerSetting.visScale);
                eulerParser.AddDNDS_Real("visScaleIn", &config.eulerSetting.visScaleIn);
                eulerParser.AddDNDS_Real("ekCutDown", &config.eulerSetting.ekCutDown);
                eulerParser.AddDNDS_Real("isiScale", &config.eulerSetting.isiScale);
                eulerParser.AddDNDS_Real("isiScaleIn", &config.eulerSetting.isiScaleIn);
                eulerParser.AddDNDS_Real("isiCutDown", &config.eulerSetting.isiCutDown);
                eulerParser.AddDNDS_Real("visScale", &config.eulerSetting.visScale);

                eulerParser.AddBool(
                    "useScalarJacobian", &config.eulerSetting.useScalarJacobian, []() {}, JSON::ParamParser::FLAG_NULL);
                eulerParser.AddBool(
                    "ignoreSourceTerm", &config.eulerSetting.ignoreSourceTerm, []() {}, JSON::ParamParser::FLAG_NULL);
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
                        DNDS_assert(eulerSetting_farFieldStaticValueBuf.size() == nVars);
                        config.eulerSetting.farFieldStaticValue = eulerSetting_farFieldStaticValueBuf;
                    });
            }
            Eigen::VectorXd eulerSetting_boxInitializerValueBuf;
            {
                eulerParser.AddEigen_RealVec(
                    "boxInitializerValue", &eulerSetting_boxInitializerValueBuf,
                    [&]()
                    {
                        DNDS_assert(eulerSetting_boxInitializerValueBuf.size() % (6 + nVars) == 0);
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
                        DNDS_assert(eulerSetting_planeInitializerValueBuf.size() % (4 + nVars) == 0);
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
                        DNDS_assert(eulerSetting_constMassForceValueBuf.size() == 3);
                        config.eulerSetting.constMassForce = eulerSetting_constMassForceValueBuf;
                    },
                    JSON::ParamParser::FLAG_NULL);
            }

            root.AddInt("curvilinearOneStep", &config.curvilinearOneStep);
            root.AddInt("curvilinearRestartNstep", &config.curvilinearRestartNstep);
            root.AddInt("curvilinearRepeatInterval", &config.curvilinearRepeatInterval);
            root.AddInt("curvilinearRepeatNum", &config.curvilinearRepeatNum);
            root.AddDNDS_Real("curvilinearRange", &config.curvilinearRange);

            root.AddBool(
                "ifSteadyQuit", &config.steadyQuit, []() {}, JSON::ParamParser::FLAG_NULL);

            root.AddInt(
                "odeCode", &config.odeCode, []() {}, JSON::ParamParser::FLAG_NULL);

            root.Parse(config_doc->GetObject(), 0);

            if (mpi.rank == 0)
                log() << "JSON: Parse Done ===" << std::endl;
        }

        void ReadMeshAndInitialize()
        {
            output_stamp = getTimeStamp(mpi);
            if (!config.uniqueStamps)
                output_stamp = "";
            if (mpi.rank == 0)
                log() << "=== Got Time Stamp: [" << output_stamp << "] ===" << std::endl;
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
                            u[iCell](I4 + 1) *= 1.0; // ! not fixing first layer!
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
                        // DNDS_assert(false);
                        u[iCell] = i.v;
                    }
                }
            }

            switch (config.eulerSetting.specialBuiltinInitializer)
            {
            case 1: // for RT problem
                DNDS_assert(model == NS || model == NS_2D);
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
            case 2: // for IV10 problem
                DNDS_assert(model == NS || model == NS_2D);
                for (index iCell = 0; iCell < u.dist->size(); iCell++)
                {
                    Elem::tPoint &pos = vfv->cellBaries[iCell];
                    real chi = 5;
                    real gamma = config.eulerSetting.idealGasProperty.gamma;
                    auto &cellAttribute = mesh->cellAtrLocal[iCell][0];
                    auto &cellRecAttribute = vfv->cellRecAtrLocal[iCell][0];
                    auto c2n = mesh->cell2nodeLocal[iCell];
                    Elem::ElementManager eCell(cellAttribute.type, cellRecAttribute.intScheme);
                    TU um;
                    um.setZero();
                    Eigen::MatrixXd coords;
                    mesh->LoadCoords(c2n, coords);
                    eCell.Integration(
                        um,
                        [&](TU &inc, int ig, Elem::tPoint &pparam, Elem::tDiFj &DiNj)
                        {
                            // std::cout << coords<< std::endl << std::endl;
                            // std::cout << DiNj << std::endl;
                            Elem::tPoint pPhysics = coords * DiNj(0, Eigen::all).transpose();
                            real rm = 8;
                            real r = std::sqrt(sqr(pPhysics(0) - 5) + sqr(pPhysics(1) - 5));
                            real dT = -(gamma - 1) / (8 * gamma * sqr(pi)) * sqr(chi) * std::exp(1 - sqr(r)) * (1 - 1. / std::exp(std::max(sqr(rm) - sqr(r), 0.0)));
                            real dux = chi / 2 / pi * std::exp((1 - sqr(r)) / 2) * -(pPhysics(1) - 5) * (1 - 1. / std::exp(std::max(sqr(rm) - sqr(r), 0.0)));
                            real duy = chi / 2 / pi * std::exp((1 - sqr(r)) / 2) * +(pPhysics(0) - 5) * (1 - 1. / std::exp(std::max(sqr(rm) - sqr(r), 0.0)));
                            real T = dT + 1;
                            real ux = dux + 1;
                            real uy = duy + 1;
                            real S = 1;
                            real rho = std::pow(T / S, 1 / (gamma - 1));
                            real p = T * rho;

                            real E = 0.5 * (sqr(ux) + sqr(uy)) * rho + p / (gamma - 1);

                            // std::cout << T << " " << rho << std::endl;
                            inc.setZero();
                            inc(0) = rho;
                            inc(1) = rho * ux;
                            inc(2) = rho * uy;
                            inc(dim + 1) = E;

                            inc *= vfv->cellGaussJacobiDets[iCell][ig]; // don't forget this
                        });
                    u[iCell] = um / fv->volumeLocal[iCell]; // mean value
                }
            case 0:
                break;
            default:
                log() << "Wrong specialBuiltinInitializer" << std::endl;
                DNDS_assert(false);
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
            std::shared_ptr<ODE::ImplicitDualTimeStep<decltype(u)>> ode;

            if (config.steadyQuit)
            {
                if (mpi.rank == 0)
                    log() << "Using steady!" << std::endl;
                config.odeCode = 1; // To bdf;
                config.nTimeStep = 1;
            }
            switch (config.odeCode)
            {
            case 0: // sdirk4
                if (mpi.rank == 0)
                    log() << "=== ODE: SDIRK4 " << std::endl;
                ode = std::make_shared<ODE::ImplicitSDIRK4DualTimeStep<decltype(u)>>(
                    u.dist->size(),
                    [&](decltype(u) &data)
                    {
                        data.resize(u.dist->size(), u.dist->getMPI(), nVars);
                        data.CreateGhostCopyComm(mesh->cell2faceLocal);
                        data.InitPersistentPullClean();
                    });
                break;
            case 1: // BDF2
                if (mpi.rank == 0)
                    log() << "=== ODE: BDF2 " << std::endl;
                ode = std::make_shared<ODE::ImplicitBDFDualTimeStep<decltype(u)>>(
                    u.dist->size(),
                    [&](decltype(u) &data)
                    {
                        data.resize(u.dist->size(), u.dist->getMPI(), nVars);
                        data.CreateGhostCopyComm(mesh->cell2faceLocal);
                        data.InitPersistentPullClean();
                    },
                    2);
                break;
            }

            Linear::GMRES_LeftPreconditioned<decltype(u)> gmres(
                config.nGmresSpace,
                [&](decltype(u) &data)
                {
                    data.resize(u.dist->size(), u.dist->getMPI(), nVars);
                    data.CreateGhostCopyComm(mesh->cell2faceLocal);
                    data.InitPersistentPullClean();
                });

            EulerEvaluator<model> eval(mesh.get(), fv.get(), vfv.get());

            /************* Files **************/
            if (mpi.rank == 0)
            {
                std::ofstream logConfig(config.outLogName + "_" + output_stamp + ".config.json");
                rapidjson::OStreamWrapper logConfigWrapper(logConfig);
                rapidjson::Writer<rapidjson::OStreamWrapper> writer(logConfigWrapper);
                rapidjson::Value ctd_info, partnum;
                ctd_info.SetString(DNDS_Defines_state.c_str(), DNDS_Defines_state.length());
                partnum.SetInt(mpi.size);
                config_doc->GetObject().AddMember("___Compile_Time_Defines", ctd_info, config_doc->GetAllocator());
                config_doc->GetObject().AddMember("___Runtime_PartitionNumber", partnum, config_doc->GetAllocator());
                config_doc->Accept(writer);
                logConfig.close();
            }

            std::ofstream logErr(config.outLogName + "_" + output_stamp + ".log");
            /************* Files **************/

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

            real tSimu = 0.0;
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
                    vfv->ReconstructionJacobiStep<dim, nVars_Fixed>(
                        cx, uRec, uRecNew,
                        [&](TU &UL, const TVec &normOut, const Elem::tPoint &pPhy, const BoundaryType bType) -> TU
                        {
                            auto normBase = Elem::NormBuildLocalBaseV(normOut);
                            return eval.generateBoundaryValue(UL, normOut, normBase, pPhy(Seq012), tSimu + ct * curDtImplicit, bType, true);
                        });
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

                    auto fML = [&](const auto &UL, const auto &UR, const auto &n) -> auto
                    {
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
                        // return real(1);
                    };
                    auto fMR = [&](const auto &UL, const auto &UR, const auto &n) -> auto
                    {
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
                        // return real(1);
                    };
                    if (config.limiterProcedure == 1)
                        vfv->ReconstructionWBAPLimitFacialV3<dim, nVars_Fixed>(
                            cx, uRec, uRecNew, uRecNew1, ifUseLimiter,
                            iter < config.nPartialLimiterStartLocal && step < config.nPartialLimiterStart,
                            fML, fMR);
                    else if (config.limiterProcedure == 0)
                        vfv->ReconstructionWBAPLimitFacialV2<dim, nVars_Fixed>(
                            cx, uRec, uRecNew, uRecNew1, ifUseLimiter,
                            iter < config.nPartialLimiterStartLocal && step < config.nPartialLimiterStart,
                            fML, fMR);
                    else
                    {
                        DNDS_assert(false);
                    }
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
                    eval.EvaluateRHS(crhs, cx, uRecNew, tSimu + ct * curDtImplicit);
                else
                    eval.EvaluateRHS(crhs, cx, uRec, tSimu + ct * curDtImplicit);
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
                    eval.LUSGSADMatrixInit(dTau, dt, alphaDiag, cx, config.jacobianTypeCode, tSimu);
                }
                else
                {
                    if (config.useLimiter) // uses urec value
                        eval.LUSGSMatrixInit(dTau, dt, alphaDiag,
                                             cx, uRecNew,
                                             config.jacobianTypeCode,
                                             tSimu);
                    else
                        eval.LUSGSMatrixInit(dTau, dt, alphaDiag,
                                             cx, uRec,
                                             config.jacobianTypeCode,
                                             tSimu);
                }
                for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
                {
                    crhs[iCell] = eval.CompressInc(cx[iCell], crhs[iCell] * dTau[iCell], crhs[iCell]) / dTau[iCell];
                }

                if (config.gmresCode == 0 || config.gmresCode == 2)
                {
                    // //! LUSGS
                    if (config.jacobianTypeCode == 0)
                    {
                        eval.UpdateLUSGSForward(alphaDiag, crhs, cx, cxInc, cxInc);
                        cxInc.StartPersistentPullClean();
                        cxInc.WaitPersistentPullClean();
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
                        DNDS_assert(false);
                        eval.UpdateLUSGSADForward(crhs, cx, cxInc, cxInc);
                        cxInc.StartPersistentPullClean();
                        cxInc.WaitPersistentPullClean();
                        eval.UpdateLUSGSADBackward(crhs, cx, cxInc, cxInc);
                        cxInc.StartPersistentPullClean();
                        cxInc.WaitPersistentPullClean();
                    }
                    for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
                    {
                        cxInc[iCell] = eval.CompressInc(cx[iCell], cxInc[iCell], crhs[iCell]);
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
                eval.EvaluateResidual(res, cxinc, 1, config.useVolWiseResidual);
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
                              << "t,dTaumin,CFL,nFix \033[92m["
                              << tSimu << ", " << curDtMin << ", " << CFLNow << ", " << eval.nFaceReducedOrder << "]\033[39m   "
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
                        std::string delimC = " ";
                        logErr
                            << std::left
                            << step << delimC
                            << std::left
                            << iter << delimC
                            << std::left
                            << std::setprecision(9) << std::scientific
                            << res.transpose() << delimC
                            << tSimu << delimC
                            << curDtMin << delimC
                            << real(eval.nFaceReducedOrder) << delimC
                            << eval.fluxWallSum.transpose() << std::endl;
                    }
                    tstart = MPI_Wtime();
                    trec = tcomm = trhs = tLim = 0.;
                    PerformanceTimer::Instance().clearAllTimer();
                }

                if (iter % config.nDataOutInternal == 0)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + "_" + output_stamp + "_" + std::to_string(step) + "_" + std::to_string(iter) + ".plt", ode);
                    nextStepOut += config.nDataOut;
                }
                if (iter % config.nDataOutCInternal == 0)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + "_" + output_stamp + "_" + "C" + ".plt", ode);
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
                tSimu += curDtImplicit;
                if (ifOutT)
                    tSimu = nextTout;
                Eigen::Vector<real, -1> res(nVars);
                eval.EvaluateResidual(res, ode->getLatestRHS(), 1, config.useVolWiseResidual);
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
                              << "t,dt(min) \033[92m[" << tSimu << ", " << curDtMin << "]\033[39m   "
                              << std::setprecision(3) << std::fixed
                              << "Time [" << telapsed << "]   recTime [" << trec << "]   rhsTime [" << trhs << "]   commTime [" << tcomm << "]  limTime [" << tLim << "]  " << std::endl;
                        log().setf(fmt);
                        std::string delimC = " ";
                        logErr
                            << std::left
                            << step << delimC
                            << std::left
                            << -1 << delimC
                            << std::left
                            << std::setprecision(9) << std::scientific
                            << res.transpose() << delimC
                            << tSimu << delimC
                            << curDtMin << delimC
                            << eval.fluxWallSum.transpose() << std::endl;
                    }
                    tstart = MPI_Wtime();
                    trec = tcomm = trhs = tLim = 0.;
                    PerformanceTimer::Instance().clearAllTimer();
                }
                if (step == nextStepOut)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + "_" + output_stamp + "_" + std::to_string(step) + ".plt", ode);
                    nextStepOut += config.nDataOut;
                }
                if (step == nextStepOutC)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + "_" + output_stamp + "_" + "C" + ".plt", ode);
                    nextStepOutC += config.nDataOutC;
                }
                if (ifOutT)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + "_" + output_stamp + "_" + "t_" + std::to_string(nextTout) + ".plt", ode);
                    nextTout += config.tDataOut;
                    if (nextTout > config.tEnd)
                        nextTout = config.tEnd;
                }
                if (config.eulerSetting.specialBuiltinInitializer == 2 && (step % config.nConsoleCheck == 0)) // IV problem special: reduction on solution
                {
                    real xymin = 5 + tSimu - 2;
                    real xymax = 5 + tSimu + 2;
                    real xyc = 5 + tSimu;
                    real sumErrRho = 0.0;
                    real sumErrRhoSum = 0.0 / 0.0;
                    real sumVol = 0.0;
                    real sumVolSum = 0.0 / 0.0;
                    for (index iCell = 0; iCell < u.dist->size(); iCell++)
                    {
                        Elem::tPoint &pos = vfv->cellBaries[iCell];
                        real chi = 5;
                        real gamma = config.eulerSetting.idealGasProperty.gamma;
                        auto &cellAttribute = mesh->cellAtrLocal[iCell][0];
                        auto &cellRecAttribute = vfv->cellRecAtrLocal[iCell][0];
                        auto c2n = mesh->cell2nodeLocal[iCell];
                        Elem::ElementManager eCell(cellAttribute.type, cellRecAttribute.intScheme);
                        TU um;
                        um.setZero();
                        Eigen::MatrixXd coords;
                        mesh->LoadCoords(c2n, coords);
                        eCell.Integration(
                            um,
                            [&](TU &inc, int ig, Elem::tPoint &pparam, Elem::tDiFj &DiNj)
                            {
                                // std::cout << coords<< std::endl << std::endl;
                                // std::cout << DiNj << std::endl;
                                Elem::tPoint pPhysics = coords * DiNj(0, Eigen::all).transpose();
                                real r = std::sqrt(sqr(pPhysics(0) - xyc) + sqr(pPhysics(1) - xyc));
                                real dT = -(gamma - 1) / (8 * gamma * sqr(pi)) * sqr(chi) * std::exp(1 - sqr(r));
                                real dux = chi / 2 / pi * std::exp((1 - sqr(r)) / 2) * -(pPhysics(1) - xyc);
                                real duy = chi / 2 / pi * std::exp((1 - sqr(r)) / 2) * +(pPhysics(0) - xyc);
                                real T = dT + 1;
                                real ux = dux + 1;
                                real uy = duy + 1;
                                real S = 1;
                                real rho = std::pow(T / S, 1 / (gamma - 1));
                                real p = T * rho;

                                real E = 0.5 * (sqr(ux) + sqr(uy)) * rho + p / (gamma - 1);

                                // std::cout << T << " " << rho << std::endl;
                                inc.setZero();
                                inc(0) = rho;
                                inc(1) = rho * ux;
                                inc(2) = rho * uy;
                                inc(dim + 1) = E;

                                inc *= vfv->cellGaussJacobiDets[iCell][ig]; // don't forget this
                            });
                        if (vfv->cellBaries[iCell](0) > xymin && vfv->cellBaries[iCell](0) < xymax && vfv->cellBaries[iCell](1) > xymin && vfv->cellBaries[iCell](1) < xymax)
                        {
                            um /= fv->volumeLocal[iCell]; // mean value
                            real errRhoMean = u[iCell](0) - um(0);
                            sumErrRho += std::abs(errRhoMean) * fv->volumeLocal[iCell];
                            sumVol += fv->volumeLocal[iCell];
                        }
                    }
                    MPI_Allreduce(&sumErrRho, &sumErrRhoSum, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
                    MPI_Allreduce(&sumVol, &sumVolSum, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
                    if (mpi.rank == 0)
                    {
                        log() << "=== Mean Error IV: [" << std::scientific << std::setprecision(5) << sumErrRhoSum << ", " << sumErrRhoSum / sumVolSum << "]" << std::endl;
                    }
                }

                stepCount++;

                return tSimu >= config.tEnd;
            };

            /**********************************/
            /*           MAIN LOOP            */
            /**********************************/

            for (step = 1; step <= config.nTimeStep; step++)
            {
                InsertCheck(mpi, "Implicit Step");
                ifOutT = false;
                curDtImplicit = config.dtImplicit; //* could add CFL driven dt here
                if (tSimu + curDtImplicit > nextTout)
                {
                    ifOutT = true;
                    curDtImplicit = (nextTout - tSimu);
                }
                CFLNow = config.CFL;
                ode->Step(
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

        void RunStaticReconstruction()
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            InsertCheck(mpi, "Implicit 1 nvars " + std::to_string(nVars));

            EulerEvaluator<model> eval(mesh.get(), fv.get(), vfv.get());

            /************* Files **************/
            if (mpi.rank == 0)
            {
                std::ofstream logConfig(config.outLogName + "_" + output_stamp + ".config.json");
                rapidjson::OStreamWrapper logConfigWrapper(logConfig);
                rapidjson::Writer<rapidjson::OStreamWrapper> writer(logConfigWrapper);
                rapidjson::Value ctd_info, partnum;
                ctd_info.SetString(DNDS_Defines_state.c_str(), DNDS_Defines_state.length());
                partnum.SetInt(mpi.size);
                config_doc->GetObject().AddMember("___Compile_Time_Defines", ctd_info, config_doc->GetAllocator());
                config_doc->GetObject().AddMember("___Runtime_PartitionNumber", partnum, config_doc->GetAllocator());
                config_doc->Accept(writer);
                logConfig.close();
            }
            /************* Files **************/

            eval.settings = config.eulerSetting;

            double tstart = MPI_Wtime();
            double trec{0}, tcomm{0}, trhs{0}, tLim{0};
            int stepCount = 0;
            Eigen::Vector<real, -1> resBaseC;
            Eigen::Vector<real, -1> resBaseCInternal;
            resBaseC.resize(nVars);
            resBaseCInternal.resize(nVars);
            resBaseC.setConstant(config.res_base);

            auto fTest = [&](Elem::tPoint p) -> Eigen::Vector<real, 3>
            {
                Eigen::Vector<real, 3> fs;
                real a = 0.0027;
                real b = -0.1409;
                real utau = std::sqrt(a * 1 * sqr(1) * std::pow(p(0) + 1e-10, b) / 2 / 1);
                real utau_dx = std::sqrt(a * 1 * sqr(1) * std::pow(p(0) + 1e-10, b) / 2. / 1.) / (p(0) + 1e-10) * (b / 2);
                real kap = 0.42;
                real nu = eval.settings.idealGasProperty.muGas;
                real c1 = 1;
                real yPlus = p(1) * utau / nu;
                real yPlus_dx = p(1) * utau_dx / nu;
                real yPlus_dy = utau / nu;
                real uPlus =
                    1. / kap * std::log(1 + kap * yPlus) +
                    c1 * (1 - std::exp(-yPlus / 11) - yPlus / 11 * exp(-0.33 * yPlus));
                real uPlus_dyPlus =
                    1. / kap * 1. / (1 + kap * yPlus) * kap +
                    c1 * (std::exp(-yPlus / 11) / 11 -
                          exp(-0.33 * yPlus) / 11 +
                          yPlus / 11 * exp(-0.33 * yPlus) * 0.33);
                real uPlus_dx = uPlus_dyPlus * yPlus_dx;
                real uPlus_dy = uPlus_dyPlus * yPlus_dy;

                real uu = uPlus * utau;
                real uu_dx = uPlus_dx * utau + utau_dx * uPlus;
                real uu_dy = uPlus_dy * utau;
                fs(0) = uu;
                fs(1) = uu_dx;
                fs(2) = uu_dy;

                // std::cout << " p " << p.transpose() << std::endl;
                // std::cout << fs.transpose() << std::endl;
                // std::cout << yPlus << std::endl;
                if (!fs.allFinite() || fs.hasNaN())
                {
                    DNDS_assert(false);
                }
                return fs;
            };

            for (index iCell = 0; iCell < u.dist->size(); iCell++)
            {
                auto &cellRecAtr = vfv->cellRecAtrLocal[iCell][0];
                auto &cellAtr = mesh->cellAtrLocal[iCell][0];
                auto &c2n = mesh->cell2nodeLocal[iCell];
                Elem::ElementManager eCell(cellAtr.type, cellRecAtr.intScheme);

                auto cellDiBjGaussBatchElemVR = (*vfv->cellDiBjGaussBatch)[iCell];
                Eigen::MatrixXd coords;
                mesh->LoadCoords(c2n, coords);

                real valueInt = 0;

                eCell.Integration(
                    valueInt,
                    [&](decltype(valueInt) &finc, int ig, Elem::tPoint &p, Elem::tDiFj &iDiNj)
                    {
                        Elem::tPoint pPhysical = coords * iDiNj(0, Eigen::all).transpose();
                        finc = fTest(pPhysical)(0);
                        finc *= vfv->cellGaussJacobiDets[iCell][ig];
                    });
                // std::cout << coords << std::endl;
                valueInt /= fv->volumeLocal[iCell];
                u[iCell](0) = 1;
                u[iCell](4) = valueInt;
            }
            u.StartPersistentPullClean();
            u.WaitPersistentPullClean();

            for (int iter = 1; iter <= config.nInternalRecStep; iter++)
            {
                vfv->ReconstructionJacobiStep<dim, nVars_Fixed>(
                    u, uRec, uRecNew,
                    [&](TU &UL, const TVec &normOut, const Elem::tPoint &pPhy, const BoundaryType bType) -> TU
                    {
                        TU Ub = UL;
                        UL(0) = fTest(pPhy)(0);
                        return Ub;
                    });
                uRec.StartPersistentPullClean();
                uRec.WaitPersistentPullClean();
                if (iter % config.nConsoleCheckInternal == 0)
                    log() << "Rec Step " << iter << std::endl;
            }
            real errP = 2.;

            Eigen::Vector<real, 3> errSum, errSumSum;
            errSum.setZero(), errSumSum.setZero();

            for (index iCell = 0; iCell < u.dist->size(); iCell++)
            {
                auto &cellRecAtr = vfv->cellRecAtrLocal[iCell][0];
                auto &cellAtr = mesh->cellAtrLocal[iCell][0];
                auto &c2n = mesh->cell2nodeLocal[iCell];
                Elem::ElementManager eCell(cellAtr.type, cellRecAtr.intScheme);

                auto cellDiBjGaussBatchElemVR = (*vfv->cellDiBjGaussBatch)[iCell];
                Eigen::MatrixXd coords;
                mesh->LoadCoords(c2n, coords);

                Eigen::Vector<real, 3> valueInt;
                valueInt.setZero();

                eCell.Integration(
                    valueInt,
                    [&](decltype(valueInt) &finc, int ig, Elem::tPoint &p, Elem::tDiFj &iDiNj)
                    {
                        Elem::tPoint pPhysical = coords * iDiNj(0, Eigen::all).transpose();
                        TDiffU GradU;
                        GradU({0, 1}, Eigen::all) =
                            cellDiBjGaussBatchElemVR.m(ig)({1, 2}, Eigen::seq(Eigen::fix<1>, Eigen::last)) *
                            uRec[iCell];
                        TU ULxy = (cellDiBjGaussBatchElemVR.m(ig).row(0).rightCols(uRec[iCell].rows()) *
                                   uRec[iCell])
                                      .transpose() +
                                  u[iCell];

                        auto fAcc = fTest(pPhysical);
                        // std::cout << ULxy.transpose() << fAcc.transpose() << std::endl;
                        finc(0) = std::pow(std::abs(ULxy(4) - fAcc(0)), errP);
                        finc(1) = std::pow(std::abs(GradU(0, 4) - fAcc(1)), errP);
                        finc(2) = std::pow(std::abs(GradU(1, 4) - fAcc(2)), errP);
                        finc *= vfv->cellGaussJacobiDets[iCell][ig];
                    });

                valueInt /= fv->volumeLocal[iCell];
                u[iCell]({1, 2, 3}) = valueInt.array().pow(1. / errP);

                if (vfv->cellBaries[iCell](0) > 0.5) //! neglecting !!!
                    errSum += valueInt * fv->volumeLocal[iCell];
            }

            MPI_Allreduce(errSum.data(), errSumSum.data(), 3, DNDS_MPI_REAL, MPI_SUM, mpi.comm);

            errSumSum = errSumSum.array().pow(1. / errP);

            if (mpi.rank == 0)
            {
                std::cout << "Errors: " << std::scientific
                          << "[" << errSumSum(0) << "]  "
                          << "[" << errSumSum(1) << "]  "
                          << "[" << errSumSum(2) << "]  "
                          << std::endl;
            }

            for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
                for (int i = 0; i < nVars; i++)
                    (*outDist)[iCell][i] = u[iCell][i];
                // std::cout << u[iCell].transpose() << std::endl;
            }
            outSerial->startPersistentPull();
            outSerial->waitPersistentPull();
            std::vector<std::string> names;
            for (int i = 0; i < nVars; i++)
            {
                names.push_back("V" + std::to_string(i));
            }
            mesh->PrintSerialPartPltBinaryDataArray(
                config.outPltName + "_static.plt", 0, nVars, //! oprank = 0
                [&](int idata)
                { return names[idata]; },
                [&](int idata, index iv)
                {
                    return (*outSerial)[iv][idata];
                },
                0);
        }

        template <typename tODE>
        void PrintData(const std::string &fname, tODE &ode)
        {

            for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
            {
                DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
                TU recu =
                    vfv->cellDiBjCenterBatch->operator[](iCell).m(0)({0}, Eigen::all).rightCols(uRec[iCell].rows()) *
                    uRec[iCell];
                // recu += u[iCell];
                // DNDS_assert(recu(0) > 0);
                // recu = EulerEvaluator::CompressRecPart(u[iCell], recu);
                recu = u[iCell] + recu * 0;
                TVec velo = (recu(Seq123).array() / recu(0)).matrix();
                real vsqr = velo.squaredNorm();
                real asqr, p, H;
                Gas::IdealGasThermal(recu(I4), recu(0), vsqr, config.eulerSetting.idealGasProperty.gamma, p, asqr, H);
                // DNDS_assert(asqr > 0);
                real M = std::sqrt(vsqr / asqr);
                real T = p / recu(0) / config.eulerSetting.idealGasProperty.Rgas;

                (*outDist)[iCell][0] = recu(0);
                for (int i = 0; i < dim; i++)
                    (*outDist)[iCell][i + 1] = velo(i);
                (*outDist)[iCell][I4 + 0] = p;
                (*outDist)[iCell][I4 + 1] = T;
                (*outDist)[iCell][I4 + 2] = M;
                // (*outDist)[iCell][7] = (bool)(ifUseLimiter[iCell] & 0x0000000FU);
                (*outDist)[iCell][I4 + 3] = ifUseLimiter[iCell][0] / (config.vfvSetting.WBAP_SmoothIndicatorScale + verySmallReal);
                // std::cout << iCell << ode.rhsbuf[0][iCell] << std::endl;
                (*outDist)[iCell][I4 + 4] = ode->getLatestRHS()[iCell][0];
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