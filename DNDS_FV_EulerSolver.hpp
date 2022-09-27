#pragma once
#include "DNDS_Gas.hpp"
#include "DNDS_Mesh.hpp"
#include "DNDS_FV_VR.hpp"
#include "DNDS_ODE.hpp"
#include "DNDS_Scripting.hpp"
#include "DNDS_Linear.hpp"
#include <iomanip>

namespace DNDS
{
    class EulerEvaluator
    {
    public:
        CompactFacedMeshSerialRW *mesh = nullptr;
        ImplicitFiniteVolume2D *fv = nullptr;
        VRFiniteVolume2D *vfv = nullptr;
        int kAv = 0;

        std::vector<real> lambdaCell;
        std::vector<real> lambdaFace;

        struct Setting
        {
            struct IdealGasProperty
            {
                real gamma = 1.4;
                real Rgas = 289;
                real muGas = 1;
                real prGas = 0.7;
                real CpGas = Rgas * gamma / (gamma - 1);
            } idealGasProperty;
            int nTimeFilterPass = 0;

            real visScale = 1;
            real visScaleIn = 1;
            real isiScale = 1;
            real isiScaleIn = 1;
            real isiCutDown = 0.5;
            real ekCutDown = 0.5;

            Eigen::Vector<real, 5> farFieldStaticValue = Eigen::Vector<real, 5>{1, 0, 0, 0, 2.5};

            struct BoxInitializer
            {
                real x0, x1, y0, y1, z0, z1;
                Eigen::Vector<real, 5> v;
            };
            std::vector<BoxInitializer> boxInitializers;

        } settings;

        EulerEvaluator(CompactFacedMeshSerialRW *Nmesh, ImplicitFiniteVolume2D *Nfv, VRFiniteVolume2D *Nvfv)
            : mesh(Nmesh), fv(Nfv), vfv(Nvfv), kAv(Nvfv->P_ORDER + 1)
        {
            lambdaCell.resize(mesh->cell2nodeLocal.size()); // but only dist part are used, ghost part to not judge for it in facial iter
            lambdaFace.resize(mesh->face2nodeLocal.size());
        }

        static Eigen::Vector<real, 5> CompressRecPart(
            const Eigen::Vector<real, 5> &umean,
            const Eigen::Vector<real, 5> &uRecInc);

        void EvaluateDt(std::vector<real> &dt,
                        ArrayLocal<VecStaticBatch<5>> &u,
                        // ArrayLocal<SemiVarMatrix<5>> &uRec,
                        real CFL, real &dtMinall, real MaxDt = 1,
                        bool UseLocaldt = false);
        /**
         * @brief
         * \param rhs overwritten;
         *
         */
        void EvaluateRHS(ArrayDOF<5u> &rhs, ArrayDOF<5u> &u,
                         ArrayLocal<SemiVarMatrix<5u>> &uRec);

        void LUSGSMatrixVec(std::vector<real> &dTau, real dt, real alphaDiag,
                            ArrayDOF<5u> &u, ArrayDOF<5u> &uInc, ArrayDOF<5u> &AuInc);

        /**
         * @brief to use LUSGS, use LUSGSForward(..., uInc, uInc); uInc.pull; LUSGSBackward(..., uInc, uInc);
         * the underlying logic is that for index, ghost > dist, so the forward uses no ghost,
         * and ghost should be pulled before using backward;
         * to use Jacobian instead of LUSGS, use LUSGSForward(..., uInc, uIncNew); LUSGSBackward(..., uInc, uIncNew); uIncNew.pull; uInc = uIncNew;
         * \param uIncNew overwritten;
         *
         */
        void UpdateLUSGSForward(std::vector<real> &dTau, real dt, real alphaDiag,
                                ArrayDOF<5u> &rhs, ArrayDOF<5u> &u, ArrayDOF<5u> &uInc, ArrayDOF<5u> &uIncNew);

        /**
         * @brief
         * \param uIncNew overwritten;
         *
         */
        void UpdateLUSGSBackward(std::vector<real> &dTau, real dt, real alphaDiag,
                                 ArrayDOF<5u> &rhs, ArrayDOF<5u> &u, ArrayDOF<5u> &uInc, ArrayDOF<5u> &uIncNew);

        void UpdateSGS(std::vector<real> &dTau, real dt, real alphaDiag,
                       ArrayDOF<5u> &rhs, ArrayDOF<5u> &u, ArrayDOF<5u> &uInc, ArrayDOF<5u> &uIncNew, bool ifForward);

        void FixUMaxFilter(ArrayDOF<5u> &u);

        void EvaluateResidual(Eigen::Vector<real, 5> &res, ArrayDOF<5u> &rhs, index P = 1);
    };

    class EulerSolver
    {
        MPIInfo mpi;
        std::shared_ptr<CompactFacedMeshSerialRW> mesh;
        std::shared_ptr<ImplicitFiniteVolume2D> fv;
        std::shared_ptr<VRFiniteVolume2D> vfv;

        ArrayDOF<5u> u, uPoisson, uInc, uIncRHS, uTemp;
        ArrayLocal<SemiVarMatrix<5u>> uRec, uRecNew, uRecNew1;

        static const int nOUTS = 9;
        // rho u v w p T M ifUseLimiter RHS
        std::shared_ptr<Array<VecStaticBatch<nOUTS>>> outDist;
        std::shared_ptr<Array<VecStaticBatch<nOUTS>>> outSerial;

        ArrayLocal<SemiVarMatrix<5u>> uF0, uF1; // ! to be dumped
        // std::vector<uint32_t> ifUseLimiter;
        ArrayLocal<Batch<real, 1>> ifUseLimiter;

    public:
        EulerSolver(const MPIInfo &nmpi) : mpi(nmpi)
        {
        }

        struct Configuration
        {
            int recOrder = 2;
            int nInternalRecStep = 1;
            int nTimeStep = 1000;
            int nConsoleCheck = 10;
            int nConsoleCheckInternal = 1;
            int nSGSIterationInternal = 0;
            int nDataOut = 10000;
            int nDataOutC = 50;
            int nDataOutInternal = 1;
            int nDataOutCInternal = 1;
            int nTimeStepInternal = 1000;
            real tDataOut = veryLargeReal;
            real tEnd = veryLargeReal;

            real CFL = 0.5;
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
            EulerEvaluator::Setting eulerSetting;

            int curvilinearOneStep = 500;
            int curvilinearRepeatInterval = 500;
            int curvilinearRepeatNum = 10;

            int curvilinearRestartNstep = 100;
            real curvilinearRange = 0.1;

            bool useLocalDt = true;
            bool useLimiter = true;
            int nForceLocalStartStep = -1;
            int nCFLRampStart = 1000;
            int nCFLRampLength = 10000;
            real CFLRampEnd = 10;

            int gmresCode = 0; // 0 for lusgs, 1 for gmres, 2 for lusgs started gmres
            int nGmresSpace = 10;
            int nGmresIter = 2;
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
            root.AddInt("nForceLocalStartStep", &config.nForceLocalStartStep);

            root.AddInt("nCFLRampStart", &config.nCFLRampStart);
            root.AddInt("nCFLRampLength", &config.nCFLRampLength);
            root.AddDNDS_Real("CFLRampEnd", &config.CFLRampEnd);

            root.AddInt("gmresCode", &config.gmresCode);
            root.AddInt("nGmresSpace", &config.nGmresSpace);
            root.AddInt("nGmresIter", &config.nGmresIter);

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
            }

            root.AddInt("nDropVisScale", &config.nDropVisScale);
            root.AddDNDS_Real("vDropVisScale", &config.vDropVisScale);

            JSON::ParamParser eulerParser(mpi);
            root.AddObject("eulerSetting", &eulerParser);
            {
                eulerParser.AddDNDS_Real("visScale", &config.eulerSetting.visScale);
                eulerParser.AddDNDS_Real("visScaleIn", &config.eulerSetting.visScaleIn);
                eulerParser.AddDNDS_Real("ekCutDown", &config.eulerSetting.ekCutDown);
                eulerParser.AddDNDS_Real("isiScale", &config.eulerSetting.isiScale);
                eulerParser.AddDNDS_Real("isiScaleIn", &config.eulerSetting.isiScaleIn);
                eulerParser.AddDNDS_Real("isiCutDown", &config.eulerSetting.isiCutDown);
                eulerParser.AddDNDS_Real("visScale", &config.eulerSetting.visScale);
                eulerParser.AddInt("nTimeFilterPass", &config.eulerSetting.nTimeFilterPass);
            }
            JSON::ParamParser eulerGasParser(mpi);
            {
                eulerParser.AddObject("idealGasProperty", &eulerGasParser);
                {
                    eulerGasParser.AddDNDS_Real("gamma", &config.eulerSetting.idealGasProperty.gamma);
                    eulerGasParser.AddDNDS_Real("Rgas", &config.eulerSetting.idealGasProperty.Rgas);
                    eulerGasParser.AddDNDS_Real("muGas", &config.eulerSetting.idealGasProperty.muGas);
                }
            }
            Eigen::VectorXd eulerSetting_farFieldStaticValueBuf;
            {
                eulerParser.AddEigen_RealVec("farFieldStaticValue", &eulerSetting_farFieldStaticValueBuf);
            }
            Eigen::VectorXd eulerSetting_boxInitializerValueBuf;
            {
                eulerParser.AddEigen_RealVec("boxInitializerValue", &eulerSetting_boxInitializerValueBuf);
            }
            root.AddInt("curvilinearOneStep", &config.curvilinearOneStep);
            root.AddInt("curvilinearRestartNstep", &config.curvilinearRestartNstep);
            root.AddInt("curvilinearRepeatInterval", &config.curvilinearRepeatInterval);
            root.AddInt("curvilinearRepeatNum", &config.curvilinearRepeatNum);
            root.AddDNDS_Real("curvilinearRange", &config.curvilinearRange);

            root.Parse(doc.GetObject(), 0);
            assert(eulerSetting_farFieldStaticValueBuf.size() == 5);
            config.eulerSetting.farFieldStaticValue = eulerSetting_farFieldStaticValueBuf;

            assert(eulerSetting_boxInitializerValueBuf.size() % (6 + 5) == 0);
            config.eulerSetting.boxInitializers.resize(eulerSetting_boxInitializerValueBuf.size() / (6 + 5));
            auto &boxVec = config.eulerSetting.boxInitializers;
            for (int iInit = 0; iInit < boxVec.size(); iInit++)
            {
                boxVec[iInit].x0 = eulerSetting_boxInitializerValueBuf((6 + 5) * iInit + 0);
                boxVec[iInit].x1 = eulerSetting_boxInitializerValueBuf((6 + 5) * iInit + 1);
                boxVec[iInit].y0 = eulerSetting_boxInitializerValueBuf((6 + 5) * iInit + 2);
                boxVec[iInit].y1 = eulerSetting_boxInitializerValueBuf((6 + 5) * iInit + 3);
                boxVec[iInit].z0 = eulerSetting_boxInitializerValueBuf((6 + 5) * iInit + 4);
                boxVec[iInit].z1 = eulerSetting_boxInitializerValueBuf((6 + 5) * iInit + 5);
                boxVec[iInit].v = eulerSetting_boxInitializerValueBuf(
                    Eigen::seq((6 + 5) * iInit + 6, (6 + 5) * iInit + 6 + 5 - 1));
            }
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

            fv->BuildMean(u);
            fv->BuildMean(uPoisson);
            fv->BuildMean(uInc);
            fv->BuildMean(uIncRHS);
            fv->BuildMean(uTemp);
            vfv->BuildRec(uRec);
            vfv->BuildRec(uRecNew);
            vfv->BuildRec(uRecNew1);
            vfv->BuildRecFacial(uF0);

            uF1.Copy(uF0);
            uF1.InitPersistentPullClean();

            // vfv->BuildRecFacial(uF1);//! why copy is bad ???
            // vfv->BuildRec(uRecNew);

            u.setConstant(config.eulerSetting.farFieldStaticValue);
            uPoisson.setConstant(0.0);

            outDist = std::make_shared<decltype(outDist)::element_type>(
                decltype(outDist)::element_type::tContext(mesh->cell2faceLocal.dist->size()), mpi);
            outSerial = std::make_shared<decltype(outDist)::element_type>(outDist.get());
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

            vfv->BuildIfUseLimiter(ifUseLimiter);
        }

        void RunExplicitSSPRK4()
        {

            ODE::ExplicitSSPRK3LocalDt<decltype(u)> ode(
                u.dist->size(),
                [&](decltype(u) &data)
                {
                    data.resize(u.dist->size(), u.dist->getMPI());
                    data.CreateGhostCopyComm(mesh->cell2faceLocal);
                });
            EulerEvaluator eval(mesh.get(), fv.get(), vfv.get());
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
            Eigen::Vector<real, 5> resBaseC;
            resBaseC.setConstant(config.res_base);

            // Doing Poisson Init:

            int curvilinearNum = 0;
            int curvilinearStepper = 0;

            real tsimu = 0.0;
            real nextTout = 0.0;
            int nextStepOut = config.nDataOut;
            int nextStepOutC = config.nDataOutC;
            PerformanceTimer::Instance().clearAllTimer();
            real CFLNow = config.CFL;
            for (int step = 1; step <= config.nTimeStep; step++)
            {

                if (step == config.nForceLocalStartStep)
                    config.useLocalDt = true;
                if (step == config.nDropVisScale)
                    eval.settings.visScale *= config.vDropVisScale;
                bool ifOutT = false;
                real curDtMin;
                curvilinearStepper++;
                ode.Step(
                    u,
                    [&](ArrayDOF<5u> &crhs, ArrayDOF<5u> &cx)
                    {
                        eval.FixUMaxFilter(u);
                        u.StartPersistentPullClean();
                        u.WaitPersistentPullClean();

                        for (int iRec = 0; iRec < config.nInternalRecStep; iRec++)
                        {
                            double tstartA = MPI_Wtime();
                            vfv->ReconstructionJacobiStep(cx, uRec, uRecNew);
                            trec += MPI_Wtime() - tstartA;

                            uRec.StartPersistentPullClean();
                            uRec.WaitPersistentPullClean();
                        }
                        double tstartH = MPI_Wtime();

                        vfv->ReconstructionWBAPLimitFacial(
                            cx, uRec, uRec, uF0, uF1, ifUseLimiter,
                            [&](const auto &UL, const auto &UR, const auto &n) -> auto{
                                Eigen::Vector<real, 5> UC = (UL + UR) * 0.5;
                                auto normBase = Elem::NormBuildLocalBaseV(n);
                                UC({1, 2, 3}) = normBase.transpose() * UC({1, 2, 3});

                                real ekFixRatio = 0.001;
                                Eigen::Vector3d velo = UC({1, 2, 3}) / UC(0);
                                real vsqr = velo.squaredNorm();
                                real Ek = vsqr * 0.5 * UC(0);
                                real Efix = Ek * ekFixRatio;
                                real e = UC(4) - Ek;
                                if (e < 0)
                                    e = 0.5 * Efix;
                                else if (e < Efix)
                                    e = (e * e + Efix * Efix) / (2 * Efix);
                                UC(4) = Ek + e;

                                // return Gas::IdealGas_EulerGasLeftEigenVector(UC, eval.settings.idealGasProperty.gamma);
                                return Eigen::Matrix<real, 5, 5>::Identity();
                            },
                            [&](const auto &UL, const auto &UR, const auto &n) -> auto{
                                Eigen::Vector<real, 5> UC = (UL + UR) * 0.5;
                                auto normBase = Elem::NormBuildLocalBaseV(n);
                                UC({1, 2, 3}) = normBase.transpose() * UC({1, 2, 3});

                                real ekFixRatio = 0.001;
                                Eigen::Vector3d velo = UC({1, 2, 3}) / UC(0);
                                real vsqr = velo.squaredNorm();
                                real Ek = vsqr * 0.5 * UC(0);
                                real Efix = Ek * ekFixRatio;
                                real e = UC(4) - Ek;
                                if (e < 0)
                                    e = 0.5 * Efix;
                                else if (e < Efix)
                                    e = (e * e + Efix * Efix) / (2 * Efix);
                                UC(4) = Ek + e;

                                // return Gas::IdealGas_EulerGasRightEigenVector(UC, eval.settings.idealGasProperty.gamma);
                                return Eigen::Matrix<real, 5, 5>::Identity();
                            });
                        tLim += MPI_Wtime() - tstartH;

                        uRec.StartPersistentPullClean(); //! this also need to update!
                        uRec.WaitPersistentPullClean();

                        double tstartE = MPI_Wtime();
                        eval.EvaluateRHS(crhs, cx, uRec);
                        trhs += MPI_Wtime() - tstartE;
                    },
                    [&](std::vector<real> &dt)
                    {
                        eval.FixUMaxFilter(u);
                        u.StartPersistentPullClean(); //! this also need to update!
                        u.WaitPersistentPullClean();
                        uRec.StartPersistentPullClean();
                        uRec.WaitPersistentPullClean();

                        eval.EvaluateDt(dt, u, CFLNow, curDtMin, 1e100, config.useLocalDt);
                        if (curDtMin + tsimu > nextTout)
                            curDtMin = nextTout - tsimu, ifOutT = true;
                        if (!config.useLocalDt)
                            for (auto &dti : dt)
                                dti = curDtMin;
                    });
                // std::cout << "A\n"
                //           << std::setprecision(15)
                //           << u[12279].transpose() << "\n"
                //           << u[12280].transpose() << std::endl;
                tsimu += curDtMin;
                if (ifOutT)
                    tsimu = nextTout;
                Eigen::Vector<real, 5> res;
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
                        log() << std::setprecision(15) << std::scientific
                              << "=== Step [" << step << "]   "
                              << "res \033[91m[" << (res.array() / resBaseC.array()).transpose() << "]\033[39m   "
                              << "t,dt(min) \033[92m[" << tsimu << ", " << curDtMin << "]\033[39m   "
                              << std::setprecision(3) << std::fixed
                              << "Time [" << telapsed << "]   recTime [" << trec << "]   rhsTime [" << trhs << "]   commTime [" << tcomm << "]  limTime [" << tLim << "]  " << std::endl;
                        log().setf(fmt);
                        logErr << step << "\t" << std::setprecision(9) << std::scientific
                               << (res.array() / resBaseC.array()).transpose() << " "
                               << tsimu << " " << curDtMin << std::endl;
                    }
                    tstart = MPI_Wtime();
                    trec = tcomm = trhs = tLim = 0.;
                    PerformanceTimer::Instance().clearAllTimer();
                }
                if (step == nextStepOut)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + std::to_string(step) + ".plt", ode);
                    nextStepOut += config.nDataOut;
                }
                if (step == nextStepOutC)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + "C" + ".plt", ode);
                    nextStepOutC += config.nDataOutC;
                }
                if (ifOutT)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + "t_" + std::to_string(nextTout) + ".plt", ode);
                    nextTout += config.tDataOut;
                    if (nextTout > config.tEnd)
                        nextTout = config.tEnd;
                }
#ifdef USE_LOCAL_COORD_CURVILINEAR
                if ((curvilinearStepper == config.curvilinearOneStep && curvilinearNum == 0) ||
                    (curvilinearStepper == config.curvilinearRepeatInterval && (curvilinearNum > 0 && curvilinearNum < config.curvilinearRepeatNum)))
                {
                    assert(!vfv->setting.anistropicLengths);
                    curvilinearStepper = 0;
                    curvilinearNum++;

                    forEachInArray(
                        *vfv->uCurve.dist,
                        [&](decltype(vfv->uCurve.dist)::element_type::tComponent &e, index iCell)
                        {
                            if (u[iCell](0) > config.curvilinearRange)
                                return;
                            auto em = e.m();

                            em.setZero();
                            em(0, 0) = em(1, 1) = 1.0;
                            int nZetaDof = em.rows();

                            auto &cellAtr = mesh->cellAtrLocal[iCell][0];
                            auto &cellAtrRec = vfv->cellRecAtrLocal[iCell][0];
                            auto eCell = Elem::ElementManager(cellAtr.type, cellAtrRec.intScheme);
                            Eigen::MatrixXd coords;
                            mesh->LoadCoords(mesh->cell2nodeLocal[iCell], coords);
                            Elem::tPoint sScale = vfv->CoordMinMaxScale(coords);
                            Elem::tPoint center = vfv->getCellCenter(iCell);

                            Eigen::MatrixXd A(nZetaDof, nZetaDof);
                            A.setZero();
                            eCell.Integration(
                                A,
                                [&](Eigen::MatrixXd &inc, int ig, Elem::tPoint pparam, Elem::tDiFj &DiNj)
                                {
                                    Eigen::MatrixXd incFull;
                                    Eigen::MatrixXd DiBj(6, nZetaDof + 1); //*remember add 1 Dof for constvalue-base
                                    vfv->FDiffBaseValue(
                                        iCell, eCell, coords, DiNj,
                                        pparam, center, sScale,
                                        Eigen::VectorXd::Zero(nZetaDof + 1),
                                        DiBj);
                                    Eigen::MatrixXd DiBjSlice = DiBj({1, 2}, Eigen::all);
                                    Eigen::MatrixXd DiBjSlice2 = DiBj({3, 4, 5}, Eigen::all);
                                    Eigen::VectorXd Weights(6);
                                    real L = sScale(0);
                                    Weights << 0, L, L, L * L, 2 * L * L, L * L;

                                    // incFull = DiBjSlice.transpose() * DiBjSlice + DiBjSlice2.transpose();
                                    incFull = DiBj.transpose() * Weights.asDiagonal() * DiBj;

                                    inc = incFull.bottomRightCorner(incFull.rows() - 1, incFull.cols() - 1);
                                    inc *= vfv->cellGaussJacobiDets[iCell][ig];
                                });

                            // std::cout << "Amat good \n"
                            //           << std::endl;

                            Eigen::MatrixXd b(nZetaDof, 2);
                            b.setZero();
                            eCell.Integration(
                                b,
                                [&](Eigen::MatrixXd &inc, int ig, Elem::tPoint pparam, Elem::tDiFj &DiNj)
                                {
                                    Eigen::MatrixXd incFull;
                                    Eigen::MatrixXd DiBj(6, nZetaDof + 1);
                                    Elem::tPoint pPhysics = coords * DiNj(0, Eigen::all).transpose();
                                    vfv->FDiffBaseValue(
                                        iCell, eCell, coords, DiNj,
                                        pparam, center, sScale,
                                        Eigen::VectorXd::Zero(nZetaDof + 1),
                                        DiBj);
                                    Eigen::MatrixXd DiBjSlice = DiBj({1, 2}, Eigen::all); //? why can't use auto to recieve
                                    // Eigen::MatrixXd DiBjSlice0 = DiBj({0}, Eigen::all);
                                    // real recVal = (vfv->cellDiBjGaussBatch->operator[](iCell).m(ig)({0}, Eigen::all).rightCols(uRec[iCell].m().rows()) * uRec[iCell].m())(0);
                                    Eigen::Vector2d recGrad = vfv->cellDiBjGaussBatch->operator[](iCell).m(ig)({1, 2}, Eigen::all).rightCols(uRec[iCell].m().rows()) * uRec[iCell].m();
                                    Eigen::Matrix2d recGrad01;
                                    recGrad01.col(0) = recGrad;
                                    recGrad01.col(1)(0) = -recGrad(1), recGrad01.col(1)(1) = recGrad(0);
                                    incFull = DiBjSlice.transpose() * recGrad01;

                                    Eigen::VectorXd Weights(6);
                                    real L = sScale(0);
                                    Weights << 0, L, L, L * L, 2 * L * L, L * L;
                                    Eigen::MatrixXd recAll = vfv->cellDiBjGaussBatch->operator[](iCell).m(ig)({0, 1, 2, 3, 4, 5}, Eigen::all).rightCols(uRec[iCell].m().rows()) *
                                                             uRec[iCell].m();
                                    Eigen::MatrixXd recAll2(6, 2);
                                    recAll2.col(0) = recAll;
                                    recAll2.col(1).setZero();
                                    recAll2.col(1)(1) = -recAll(2), recAll2.col(1)(2) = recAll(1);
                                    incFull = DiBj.transpose() * Weights.asDiagonal() * recAll2;

                                    inc = incFull.bottomRows(incFull.rows() - 1);
                                    inc *= vfv->cellGaussJacobiDets[iCell][ig];
                                });
                            Eigen::MatrixXd Ainv;
                            HardEigen::EigenLeastSquareInverse(A, Ainv);
                            em = Ainv * b;
                            Eigen::MatrixXd lengths = em({0, 1}, Eigen::all).colwise().norm();
                            real length0 = lengths.norm() / std::sqrt(2);
                            length0 = sScale(0);
                            em /= length0;

                            // std::cout << "REC " << uRec[iCell].m().transpose();
                            // std::cout << " EM \n"
                            //           << std::scientific << std::setprecision(6) << em.transpose() << std::endl;
                            // exit(123);
                        });

                    vfv->uCurve.StartPersistentPullClean();
                    vfv->uCurve.WaitPersistentPullClean();
                    // InsertCheck(mpi, "CHECK VFVRENEW B");
                    vfv->Initialization_RenewBase();
                    // InsertCheck(mpi, "CHECK VFVRENEW");
                    cfv = std::make_shared<CRFiniteVolume2D>(*vfv);
                    // InsertCheck(mpi, "CHECK CFVDONE");
                    cfv->Initialization();
                    // std::cout << cfv->baseMoments.size() << "cfv- "<< cfv->faceNormCenter[0].size()
                    eval.cfv = cfv.get();
                    forEachInArray(
                        *uRec.dist,
                        [&](decltype(uRec.dist)::element_type::tComponent &e, index iCell)
                        {
                            e.m().setZero();
                        });
                    for (int i = 0; i < config.curvilinearRestartNstep; i++)
                    {
                        uRec.StartPersistentPullClean();
                        uRec.WaitPersistentPullClean();
                        vfv->ReconstructionJacobiStep(u, uRec, uRecNew);
                        if (mpi.rank == 0)
                            log() << "--- Restart Reconstruction " << i << std::endl;
                    }
                }
#endif

                stepCount++;

                if (tsimu == config.tEnd)
                    break;
            }

            // u.WaitPersistentPullClean();
            logErr.close();
        }

        void RunImplicitEuler()
        {

            ODE::ImplicitSDIRK4DualTimeStep<decltype(u)> ode(
                u.dist->size(),
                [&](decltype(u) &data)
                {
                    data.resize(u.dist->size(), u.dist->getMPI());
                    data.CreateGhostCopyComm(mesh->cell2faceLocal);
                    data.InitPersistentPullClean();
                });
            Linear::GMRES_LeftPreconditioned<decltype(u)> gmres(
                config.nGmresSpace,
                [&](decltype(u) &data)
                {
                    data.resize(u.dist->size(), u.dist->getMPI());
                    data.CreateGhostCopyComm(mesh->cell2faceLocal);
                    data.InitPersistentPullClean();
                });

            EulerEvaluator eval(mesh.get(), fv.get(), vfv.get());

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
            Eigen::Vector<real, 5> resBaseC;
            Eigen::Vector<real, 5> resBaseCInternal;
            resBaseC.setConstant(config.res_base);

            // Doing Poisson Init:

            int curvilinearNum = 0;
            int curvilinearStepper = 0;

            real tsimu = 0.0;
            real nextTout = config.tDataOut;
            int nextStepOut = config.nDataOut;
            int nextStepOutC = config.nDataOutC;
            PerformanceTimer::Instance().clearAllTimer();
            real CFLNow = config.CFL;
            for (int step = 1; step <= config.nTimeStep; step++)
            {

                bool ifOutT = false;
                real curDtMin;
                real curDtImplicit = config.dtImplicit;
                if (tsimu + curDtImplicit > nextTout)
                {
                    ifOutT = true;
                    curDtImplicit = (nextTout - tsimu);
                }
                CFLNow = config.CFL;
                ode.Step(
                    u, uInc,
                    [&](ArrayDOF<5u> &crhs, ArrayDOF<5u> &cx)
                    {
                        eval.FixUMaxFilter(cx);
                        cx.StartPersistentPullClean();
                        cx.WaitPersistentPullClean();

                        for (int iRec = 0; iRec < config.nInternalRecStep; iRec++)
                        {
                            double tstartA = MPI_Wtime();
                            vfv->ReconstructionJacobiStep(cx, uRec, uRecNew);
                            trec += MPI_Wtime() - tstartA;

                            uRec.StartPersistentPullClean();
                            uRec.WaitPersistentPullClean();
                        }
                        double tstartH = MPI_Wtime();

                        if (config.useLimiter)
                        {
                            // vfv->ReconstructionWBAPLimitFacial(
                            //     cx, uRec, uRecNew, uF0, uF1, ifUseLimiter,
                            vfv->ReconstructionWBAPLimitFacialV2(
                                cx, uRec, uRecNew, uRecNew1, ifUseLimiter,
                                [&](const auto &UL, const auto &UR, const auto &n) -> auto{
                                    Eigen::Vector<real, 5> UC = (UL + UR) * 0.5;
                                    auto normBase = Elem::NormBuildLocalBaseV(n);
                                    UC({1, 2, 3}) = normBase.transpose() * UC({1, 2, 3});

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

                                    return Gas::IdealGas_EulerGasLeftEigenVector(UC, eval.settings.idealGasProperty.gamma);
                                    // return Eigen::Matrix<real, 5, 5>::Identity();
                                },
                                [&](const auto &UL, const auto &UR, const auto &n) -> auto{
                                    Eigen::Vector<real, 5> UC = (UL + UR) * 0.5;
                                    auto normBase = Elem::NormBuildLocalBaseV(n);
                                    UC({1, 2, 3}) = normBase.transpose() * UC({1, 2, 3});

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

                                    return Gas::IdealGas_EulerGasRightEigenVector(UC, eval.settings.idealGasProperty.gamma);
                                    // return Eigen::Matrix<real, 5, 5>::Identity();
                                });
                            uRecNew.StartPersistentPullClean();
                            uRecNew.WaitPersistentPullClean();
                        }
                        tLim += MPI_Wtime() - tstartH;

                        uRec.StartPersistentPullClean(); //! this also need to update!
                        uRec.WaitPersistentPullClean();

                        // }

                        double tstartE = MPI_Wtime();
                        if (config.useLimiter)
                            eval.EvaluateRHS(crhs, cx, uRecNew);
                        else
                            eval.EvaluateRHS(crhs, cx, uRec);
                        trhs += MPI_Wtime() - tstartE;
                    },
                    [&](std::vector<real> &dTau)
                    {
                        eval.FixUMaxFilter(u);
                        u.StartPersistentPullClean(); //! this also need to update!
                        u.WaitPersistentPullClean();
                        // uRec.StartPersistentPullClean();
                        // uRec.WaitPersistentPullClean();

                        eval.EvaluateDt(dTau, u, CFLNow, curDtMin, 1e100, true);
                    },
                    [&](ArrayDOF<5u> &cx, ArrayDOF<5u> &crhs, std::vector<real> &dTau,
                        real dt, real alphaDiag, ArrayDOF<5u> &cxInc)
                    {
                        cxInc.setConstant(0.0);

                        for (int iPass = 1; iPass <= 0; iPass++)
                        {
                            crhs.StartPersistentPullClean();
                            crhs.WaitPersistentPullClean();
                            for (index iCell = 0; iCell < mesh->cell2nodeLocal.dist->size(); iCell++)
                            {
                                auto &c2f = mesh->cell2faceLocal[iCell];
                                // Eigen::Vector<real, 5> duC = crhs[iCell] * 1.0;
                                // real duC_N = 1.0;

                                Eigen::Vector<real, 5> duC;
                                duC.setZero();
                                real duC_N = 0.0;

                                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                                {
                                    index iFace = c2f[ic2f];
                                    auto &f2c = (*mesh->face2cellPair)[iFace];
                                    index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                                    index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                                    if (iCellOther != FACE_2_VOL_EMPTY)
                                    {
                                        real dist = (eval.vfv->cellBaries[iCellOther] - eval.vfv->cellBaries[iCell]).norm();
                                        real divisor = eval.fv->faceArea[iFace] / dist;
                                        duC += crhs[iCellOther] * divisor;
                                        duC_N += divisor;
                                    }
                                }
                                uPoisson[iCell] = 0.5 * (duC / duC_N + crhs[iCell]);
                            }
                            uPoisson.StartPersistentPullClean();
                            uPoisson.WaitPersistentPullClean();
                            crhs = uPoisson;
                        }

                        if (config.gmresCode == 0 || config.gmresCode == 2)
                        {
                            //! LUSGS
                            eval.UpdateLUSGSForward(dTau, dt, alphaDiag, crhs, cx, cxInc, cxInc);
                            cxInc.StartPersistentPullClean();
                            cxInc.WaitPersistentPullClean();
                            eval.UpdateLUSGSBackward(dTau, dt, alphaDiag, crhs, cx, cxInc, cxInc);
                            cxInc.StartPersistentPullClean();
                            cxInc.WaitPersistentPullClean();
                            for (int iIter = 1; iIter <= config.nSGSIterationInternal; iIter++)
                            {
                                cxInc.StartPersistentPullClean();
                                cxInc.WaitPersistentPullClean();
                                eval.UpdateSGS(dTau, dt, alphaDiag, crhs, cx, cxInc, cxInc, true);
                                cxInc.StartPersistentPullClean();
                                cxInc.WaitPersistentPullClean();
                                eval.UpdateSGS(dTau, dt, alphaDiag, crhs, cx, cxInc, cxInc, false);
                            }
                            // return;
                        }

                        if (config.gmresCode != 0)
                        {
                            // !  GMRES
                            for (index iCell = 0; iCell < crhs.dist->size(); iCell++)
                            {
                                uIncRHS[iCell] = crhs[iCell] * eval.fv->volumeLocal[iCell];
                            }

                            gmres.solve(
                                [&](decltype(u) &x, decltype(u) &Ax)
                                {
                                    eval.LUSGSMatrixVec(dTau, dt, alphaDiag, cx, x, Ax);
                                },
                                [&](decltype(u) &x, decltype(u) &MLx)
                                {
                                    // x as rhs, and MLx as uinc
                                    for (index iCell = 0; iCell < x.dist->size(); iCell++)
                                    {
                                        uTemp[iCell] = x[iCell] / eval.fv->volumeLocal[iCell]; //! UpdateLUSGS needs not these volumes multiplied
                                    }
                                    eval.UpdateLUSGSForward(dTau, dt, alphaDiag, uTemp, cx, MLx, MLx);
                                    MLx.StartPersistentPullClean();
                                    MLx.WaitPersistentPullClean();
                                    eval.UpdateLUSGSBackward(dTau, dt, alphaDiag, uTemp, cx, MLx, MLx);
                                    MLx.StartPersistentPullClean();
                                    MLx.WaitPersistentPullClean();
                                },
                                uIncRHS, cxInc, config.nGmresIter,
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
                        }
                    },
                    config.nTimeStepInternal,
                    [&](int iter, ArrayDOF<5u> &cxinc, int iStep) -> bool
                    {
                        Eigen::Vector<real, 5> res;
                        eval.EvaluateResidual(res, cxinc);
                        // if (iter == 1 && iStep == 1) // * using 1st rk step for reference
                        if (iter == 1)
                            resBaseCInternal = res;
                        Eigen::Vector<real, 5> resRel = (res.array() / resBaseCInternal.array()).matrix();
                        if (iter % config.nConsoleCheckInternal == 0)
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
                                      << "Time [" << telapsed << "]   recTime [" << trec << "]   rhsTime [" << trhs << "]   commTime [" << tcomm << "]  limTime [" << tLim << "]  " << std::endl;
                                log().setf(fmt);
                                logErr << step << "\t" << iter << "\t" << std::setprecision(9) << std::scientific
                                       << resRel.transpose() << " "
                                       << tsimu << " " << curDtMin << std::endl;
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
                        return resRel(0) < config.rhsThresholdInternal; // ! using only rho's residual
                    },
                    curDtImplicit + verySmallReal);

                tsimu += curDtImplicit;
                if (ifOutT)
                    tsimu = nextTout;
                Eigen::Vector<real, 5> res;
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
                               << (res.array() / resBaseC.array()).transpose() << " "
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

                if (tsimu >= config.tEnd)
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
                Eigen::Vector<real, 5> recu =
                    vfv->cellDiBjCenterBatch->operator[](iCell).m(0)({0}, Eigen::all).rightCols(uRec[iCell].m().rows()) *
                    uRec[iCell].m();
                // recu += u[iCell];
                // assert(recu(0) > 0);
                recu = EulerEvaluator::CompressRecPart(u[iCell], recu);
                Gas::tVec velo = (recu({1, 2, 3}).array() / recu(0)).matrix();
                real vsqr = velo.squaredNorm();
                real asqr, p, H;
                Gas::IdealGasThermal(recu(4), recu(0), vsqr, config.eulerSetting.idealGasProperty.gamma, p, asqr, H);
                // assert(asqr > 0);
                real M = std::sqrt(vsqr / asqr);
                real T = p / recu(0) / config.eulerSetting.idealGasProperty.Rgas;

                (*outDist)[iCell][0] = recu(0);
                (*outDist)[iCell][1] = velo(0);
                (*outDist)[iCell][2] = velo(1);
                (*outDist)[iCell][3] = velo(2);
                (*outDist)[iCell][4] = p;
                (*outDist)[iCell][5] = T;
                (*outDist)[iCell][6] = M;
                // (*outDist)[iCell][7] = (bool)(ifUseLimiter[iCell] & 0x0000000FU);
                (*outDist)[iCell][7] = ifUseLimiter[iCell][0] / config.vfvSetting.WBAP_SmoothIndicatorScale;
                // std::cout << iCell << ode.rhsbuf[0][iCell] << std::endl;
                (*outDist)[iCell][8] = ode.rhsbuf[0][iCell](0);
            }
            outSerial->startPersistentPull();
            outSerial->waitPersistentPull();
            const static std::vector<std::string> names{
                "R", "U", "V", "W", "P", "T", "M", "ifUseLimiter", "RHSr"};
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