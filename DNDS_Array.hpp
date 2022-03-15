#pragma once

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "DNDS_Defines.h"
#include "DNDS_MPI.hpp"
#include "DNDS_IndexMapping.hpp"
#include "DNDS_BasicTypes.hpp"

namespace DNDS
{
    struct ArrayCommStat
    {
        bool hasGlobalMapping = false;
        bool hasGhostMapping = false;
        bool hasGhostIndexer = false;
        bool hasCommTypes = false;
        bool hasPersistentPullReqs = false;
        bool PersistentPullFinished = true;
        bool hasPersistentPushReqs = false;
        bool PersistentPushFinished = true;
    };

    template <class T>
    class Array
    {

    private:
        std::vector<uint8_t> data;
        std::vector<uint8_t> dataGhost;

        MPIInfo mpi;

    public:
        /******************************************************************************************************************************/
        // basic aux info
        typedef typename T::Indexer tIndexer;
        typedef typename T::Context tContext;
        tContext context;
        tIndexer indexer;
        
        /******************************************************************************************************************************/

        /******************************************************************************************************************************/
        // *** NOTE: the comm info is only store once, for currently i only need one ghosting comm scheme per Array<T> ***
        // ** comm aux info: ghost mapping **
        typedef std::shared_ptr<GlobalOffsetsMapping> tpLGlobalMapping;
        typedef std::shared_ptr<OffsetAscendIndexMapping> tpLGhostMapping;
        tpLGlobalMapping pLGlobalMapping;
        tpLGhostMapping pLGhostMapping;

        // ** comm aux info: sparse byte structures **
        tIndexer ghostIndexer;
        // pull means from other main to this ghost
        // records received datatype
        tpMPITypePairHolder pPushTypeVec;
        tpMPITypePairHolder pPullTypeVec;

        // ** comm aux info: comm running structures **
        MPIReqHolder PushReqVec;
        MPIReqHolder PullReqVec;
        tMPI_statVec PushStatVec;
        tMPI_statVec PullStatVec;

        // ** comm aux info: safe guard status **
        ArrayCommStat commStat;
        /******************************************************************************************************************************/

        /******************************************************************************************************************************/

        static std::string PrintTypes()
        {
            return "Types: " + T::Indexer::Tname + ", " + T::Context::Tname;
        }

        /******************************************************************************************************************************/
        /**
         * \brief construct array and its indexer via a new context
         *
         */
        template <class TtContext>
        Array(TtContext &&newContext, const MPIInfo &nmpi)
            : context(std::forward<TtContext>(newContext)),
              indexer(context), ghostIndexer(), mpi(nmpi)
        {
            data.resize(indexer.LengthByte());
            // std::cout << "IndexerLengthByte " << indexer.LengthByte() << std::endl;
            // ghostIndexer is initialized as empty
        }
        /******************************************************************************************************************************/

        /******************************************************************************************************************************/
        /**
         *  \brief  copy constructor: similar with automatic, data is also copied, not copying any ReqVec or StatVec
         *  \warning To do comm-s, must InitPersistent... -> start/wait
         */
        Array(const Array<T> &Rarray)
            : data(Rarray.data),
              dataGhost(Rarray.dataGhost),
              mpi(Rarray.mpi),
              context(Rarray.context),
              indexer(Rarray.indexer),
              pLGlobalMapping(Rarray.pLGlobalMapping),
              pLGhostMapping(Rarray.pLGhostMapping),
              ghostIndexer(Rarray.ghostIndexer),
              pPushTypeVec(Rarray.pPushTypeVec),
              pPullTypeVec(Rarray.pPullTypeVec)
        {
        }

        /**
         *  \brief  copyer, comm topology copied, not copying ghostIndexer, not copying any data, comm-type, ReqVec or StatVec
         *  \warning To do comm-s, must createMPITypes -> InitPersistent... -> start/wait
         */
        template <class TtContext, class TR>
        Array(TtContext &&newContext, const Array<TR> &Rarray)
            : data(),
              dataGhost(),
              mpi(Rarray.mpi),
              context(std::forward<TtContext>(newContext)),
              indexer(context),
              pLGlobalMapping(Rarray.pLGlobalMapping),
              pLGhostMapping(Rarray.pLGhostMapping),
              ghostIndexer(),
              pPushTypeVec(),
              pPullTypeVec()
        {
            data.resize(indexer.LengthByte());
        }
        /******************************************************************************************************************************/

        // template <class TtContext>
        // Array(const Array<T> &R, TtContext &&newContext, const MPIInfo &nmpi)
        //     : context(std::forward<TtContext>(newContext)),
        //       indexer(context),
        // {
        // }

        ~Array()
        {
        }

        T operator[](index i)
        {
            auto indexInfo = indexer[i];
            assert(std::get<0>(indexInfo) + std::get<1>(indexInfo) <= data.size());
            assert(std::get<0>(indexInfo) >= 0);
            return T(data.data() + std::get<0>(indexInfo), std::get<1>(indexInfo), context, i);
        }

        T indexGhostData(index i)
        {
            auto indexInfo = ghostIndexer[i];
            assert(std::get<0>(indexInfo) + std::get<1>(indexInfo) <= dataGhost.size());
            assert(std::get<0>(indexInfo) >= 0);
            return T(dataGhost.data() + std::get<0>(indexInfo), std::get<1>(indexInfo), context, i);
        }

        class Iterator
        {
            T instace;
            uint8_t *ptr;
            tIndexer *pIndexer;
            tContext *pContext;
            index i;

        public:
            Iterator(uint8_t *nptr, index ni, tIndexer *npIndexer, tContext *npContext)
                : ptr(nptr), i(ni), pIndexer(npIndexer),
                  pContext(npContext)
            {
                if (i < pIndexer->Length)
                {
                    auto indexInfo = (*pIndexer)[i];
                    assert(std::get<0>(indexInfo) + std::get<1>(indexInfo) < data.size());
                    assert(std::get<0>(indexInfo) >= 0);
                    instace = T(ptr + std::get<0>(indexInfo), std::get<1>(indexInfo), *pContext, i);
                }
            }

            inline bool operator!=(const Iterator &r)
            {
                assert(ptr == r.ptr);
                return i != r.i;
            }

            inline void operator++()
            {
                ++i;
                auto indexInfo = (*pIndexer)[i];
                instace.ConstructOn(ptr + std::get<0>(indexInfo), std::get<1>(indexInfo), *pContext, i);
            }

            T &operator*()
            {
                assert(i < pIndexer->Length);
                return instace;
            }
        };

        index size() { return indexer.Length; }                     // in unit of T instances
        index sizeGhost() { return ghostIndexer.Length; };          // in unit of T instances
        index sizeByte() { return indexer.LengthByte(); }           // in unit of bytes
        index sizeByteGhost() { return ghostIndexer.LengthByte(); } // in unit of bytes

        /******************************************************************************************************************************/
        // recommend: TPullSet == tIndexVec (& &&),
        // // has to be able to be copied into or moved operator= int tIndexVec
        // has to be able to be range - iterated
        /**
         * \brief from a set of global pulling indexes, create pLGlobalMapping pLGhostMapping
         * \post pLGlobalMapping pLGhostMapping established
         */
        template <class TPullSet>
        void createGhostMapping(TPullSet &&pullingIndexGlobal)
        {
            // phase1.1: create localGlobal mapping (broadcast)
            pLGlobalMapping = std::make_shared<GlobalOffsetsMapping>();
            pLGlobalMapping->setMPIAlignBcast(mpi, size());
            commStat.hasGlobalMapping = true;

            // phase1.2: count how many to pull and allocate the localGhost mapping, fill the mapping
            // counting could overflow
            // tMPI_intVec ghostSizes(mpi.size, 0); // == pulling sizes
            pLGhostMapping = std::make_shared<OffsetAscendIndexMapping>((*pLGlobalMapping)(mpi.rank, 0), size(),
                                                                        std::forward<TPullSet>(pullingIndexGlobal),
                                                                        *pLGlobalMapping,
                                                                        mpi);
            commStat.hasGhostMapping = true;
        }
        /******************************************************************************************************************************/

        /******************************************************************************************************************************/
        /**
         * \brief get real element byte info into account, with ghost indexer and comm types built, need two mappings
         * \pre has pLGlobalMapping pLGhostMapping
         * \post ghostIndexer pPullTypeVec pPushTypeVec established
         */
        void createMPITypes()
        {
            assert(commStat.hasGlobalMapping && commStat.hasGhostMapping);
            assert(pLGhostMapping.use_count() > 0 && pLGlobalMapping.use_count() > 0);
            /*********************************************/ // starts to deal with actual byte sizes
            // phase2.1: build push sizes and push disps
            tMPI_intVec pushingSizes(pLGhostMapping->pushingIndexGlobal.size());  // pushing sizes in bytes
            tMPI_AintVec pushingDisps(pLGhostMapping->pushingIndexGlobal.size()); // pushing disps in bytes

            for (index i = 0; i < pLGhostMapping->pushingIndexGlobal.size(); i++)
            {
                MPI_int rank;
                index loc;
                bool found = pLGhostMapping->search(pLGhostMapping->pushingIndexGlobal[i], rank, loc);
                assert(found && rank == -1); // must be at local main
                auto indexerRet = indexer[loc];
                pushingDisps[i] = std::get<0>(indexerRet); // pushing disps are not contiguous
                pushingSizes[i] = std::get<1>(indexerRet);
            }

            // phase2.2: be informed of pulled sub-indexer
            // equals to: building pullingSizes and pullingDisps, bytes size and disps of ghost
            ghostIndexer.buildAsGhostAlltoall(indexer, pushingSizes, *pLGhostMapping, mpi);
            dataGhost.resize(ghostIndexer.LengthByte(), 1); // data section must correspond to indexer
                                                            // std::cout << "Resize Ghost" << dataGhost.size() << std::endl;
            commStat.hasGhostIndexer = true;

            // InsertCheck(mpi);
            // std::cout << mpi.rank << " VEC ";
            // PrintVec(pushingSizes, std::cout);
            // std::cout << std::endl;
            // InsertCheck(mpi);

            // phase3: create and register MPI types of pushing and pulling
            pPushTypeVec = std::make_shared<MPITypePairHolder>(0);
            pPullTypeVec = std::make_shared<MPITypePairHolder>(0);
            for (MPI_int r = 0; r < mpi.size; r++)
            {
                // push
                MPI_int pushNumber = pLGhostMapping->pushIndexSizes[r];
                if (pushNumber > 0)
                {
                    MPI_Aint *pPushDisps = pushingDisps.data() + pLGhostMapping->pushIndexStarts[r];
                    MPI_int *pPushSizes = pushingSizes.data() + pLGhostMapping->pushIndexStarts[r];
                    // std::cout <<mpi.rank<< " pushSlice " << pPushDisps[0] << outputDelim << pPushSizes[0] << std::endl;
                    MPI_Datatype dtype;
                    MPI_Type_create_hindexed(pushNumber, pPushSizes, pPushDisps, MPI_BYTE, &dtype);
                    MPI_Type_commit(&dtype);
                    pPushTypeVec->push_back(std::make_pair(r, dtype));
                    // OPT: could use MPI_Type_create_hindexed_block to save some space
                }
                // pull
                MPI_Aint pullDisp[1];
                MPI_int pullBytes[1];
                auto gRbyte = ghostIndexer(index(pLGhostMapping->ghostStart[r + 1]));
                auto gLbyte = ghostIndexer(index(pLGhostMapping->ghostStart[r]));

                pullBytes[0] = gRbyte - gLbyte; // warning: overflow here
                pullDisp[0] = gLbyte;
                if (pullBytes[0] > 0)
                {
                    MPI_Datatype dtype;
                    MPI_Type_create_hindexed(1, pullBytes, pullDisp, MPI_BYTE, &dtype);
                    // std::cout << mpi.rank << " pullSlice " << pullDisp[0] << outputDelim << pullBytes[0] << std::endl;
                    MPI_Type_commit(&dtype);
                    pPullTypeVec->push_back(std::make_pair(r, dtype));
                }
            }
            pPullTypeVec->shrink_to_fit();
            pPushTypeVec->shrink_to_fit(); // shrink as was dynamically modified
            commStat.hasCommTypes = true;
        }
        /******************************************************************************************************************************/

        /******************************************************************************************************************************/
        /**
         * \brief when established push and pull types, init persistent-nonblocked-nonbuffered MPI reqs
         * \pre has pPullTypeVec pPushTypeVec
         * \post PushReqVec established
         */
        void initPersistentPush()
        {
            assert(commStat.hasCommTypes);
            assert(pPullTypeVec.use_count() > 0 && pPushTypeVec.use_count() > 0);
            auto nReqs = pPullTypeVec->size() + pPushTypeVec->size();
            // assert(nReqs > 0);
            PushReqVec.resize(nReqs, (MPI_REQUEST_NULL)), PushStatVec.resize(nReqs);
            for (auto ip = 0; ip < pPullTypeVec->size(); ip++)
            {
                auto dtypeInfo = (*pPullTypeVec)[ip];
                MPI_int rankOther = dtypeInfo.first;
                MPI_int tag = rankOther + mpi.rank;
                MPI_Send_init(dataGhost.data(), 1, dtypeInfo.second, rankOther, tag, mpi.comm, PushReqVec.data() + ip);
            }
            for (auto ip = 0; ip < pPushTypeVec->size(); ip++)
            {
                auto dtypeInfo = (*pPushTypeVec)[ip];
                MPI_int rankOther = dtypeInfo.first;
                MPI_int tag = rankOther + mpi.rank;
                MPI_Recv_init(data.data(), 1, dtypeInfo.second, rankOther, tag, mpi.comm, PushReqVec.data() + pPullTypeVec->size() + ip);
            }
            commStat.hasPersistentPushReqs = true;
        }
        /******************************************************************************************************************************/

        /******************************************************************************************************************************/
        /**
         * \brief when established push and pull types, init persistent-nonblocked-nonbuffered MPI reqs
         * \pre has pPullTypeVec pPushTypeVec
         * \post PullReqVec established
         */
        void initPersistentPull()
        {
            assert(commStat.hasCommTypes);
            assert(pPullTypeVec.use_count() > 0 && pPushTypeVec.use_count() > 0);
            auto nReqs = pPullTypeVec->size() + pPushTypeVec->size();
            // assert(nReqs > 0);
            PullReqVec.resize(nReqs, (MPI_REQUEST_NULL)), PullStatVec.resize(nReqs);
            for (auto ip = 0; ip < pPullTypeVec->size(); ip++)
            {
                auto dtypeInfo = (*pPullTypeVec)[ip];
                MPI_int rankOther = dtypeInfo.first;
                MPI_int tag = rankOther + mpi.rank;
                // std::cout << mpi.rank << " Recv " << rankOther << std::endl;
                MPI_Recv_init(dataGhost.data(), 1, dtypeInfo.second, rankOther, tag, mpi.comm, PullReqVec.data() + ip);
                // std::cout << *(real *)(dataGhost.data() + 8 * 0) << std::endl;
            }
            for (auto ip = 0; ip < pPushTypeVec->size(); ip++)
            {
                auto dtypeInfo = (*pPushTypeVec)[ip];
                MPI_int rankOther = dtypeInfo.first;
                MPI_int tag = rankOther + mpi.rank;
                // std::cout << mpi.rank << " Send " << rankOther << std::endl;
                MPI_Send_init(data.data(), 1, dtypeInfo.second, rankOther, tag, mpi.comm, PullReqVec.data() + pPullTypeVec->size() + ip);
                // std::cout << *(real *)(data.data() + 8 * 1) << std::endl;
            }
            commStat.hasPersistentPullReqs = true;
        }
        /******************************************************************************************************************************/

        void startPersistentPush()
        {
            assert(commStat.hasPersistentPushReqs && commStat.PersistentPushFinished);
            MPI_Startall(PushReqVec.size(), PushReqVec.data());
            commStat.PersistentPushFinished = false;
        }
        void startPersistentPull()
        {
            assert(commStat.hasPersistentPullReqs && commStat.PersistentPullFinished);
            MPI_Startall(PullReqVec.size(), PullReqVec.data());
            commStat.PersistentPullFinished = false;
        }

        void waitPersistentPush()
        {
            assert(commStat.hasPersistentPushReqs && !commStat.PersistentPushFinished);
            MPI_Waitall(PushReqVec.size(), PushReqVec.data(), PushStatVec.data());
            commStat.PersistentPushFinished = true;
        }
        void waitPersistentPull()
        {
            assert(commStat.hasPersistentPullReqs && !commStat.PersistentPullFinished);
            MPI_Waitall(PullReqVec.size(), PullReqVec.data(), PullStatVec.data());
            commStat.PersistentPullFinished = true;
        }
    };
}