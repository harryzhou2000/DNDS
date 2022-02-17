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
    template <class T>
    class Array
    {

    private:
        std::vector<uint8_t> data;
        std::vector<uint8_t> dataGhost;

        MPIInfo mpi;

    public:
        typedef typename T::Indexer tIndexer;
        typedef typename T::Context tContext;
        tContext context;
        tIndexer indexer;
        tIndexer ghostIndexer;

        typedef std::shared_ptr<GlobalOffsetsMapping> tpLGlobalMapping;
        typedef std::shared_ptr<OffsetAscendIndexMapping> tpLGhostMapping;
        tpLGlobalMapping pLGlobalMapping;
        tpLGhostMapping pLGhostMapping;

        typedef std::vector<std::pair<MPI_int, MPI_Datatype>> tMPI_typePairVec;
        typedef std::shared_ptr<tMPI_typePairVec> tpMPI_typePairVec;
        // pull means from other main to this ghost
        // records received datatype
        tpMPI_typePairVec pPushTypeVec;
        tpMPI_typePairVec pPullTypeVec;

        tMPI_reqVec PushReqVec;
        tMPI_reqVec PullReqVec;
        tMPI_statVec PushStatVec;
        tMPI_statVec PullStatVec;

        static std::string PrintTypes()
        {
            return "Types: " + T::Indexer::Tname + ", " + T::Context::Tname;
        }

        template <class TtContext>
        Array(TtContext &&newContext, const MPIInfo &nmpi) : context(std::forward<TtContext>(newContext)),
                                                             indexer(context), ghostIndexer(indexer), mpi(nmpi)
        {
            data.resize(indexer.LengthByte());
        }

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

        // recommend: TPullSet == tIndexVec (& &&),
        // has to be able to be copied into or moved operator= int tIndexVec
        template <class TPullSet>
        void createGhost(TPullSet &&pullingIndexGlobal)
        {
            // phase1.1: create localGlobal mapping (broadcast)
            pLGlobalMapping = std::make_shared<GlobalOffsetsMapping>();

            pLGlobalMapping->setMPIAlignBcast(mpi, size());
            pLGhostMapping = std::make_shared<OffsetAscendIndexMapping>((*pLGlobalMapping)(mpi.rank, 0), size());

            // phase1.2: count how many to pull and allocate the localGhost mapping, fill the mapping
            // counting could overflow
            tMPI_intVec ghostSizes(mpi.size, 0LL); // == pulling sizes
            for (auto i : pullingIndexGlobal)
            {
                MPI_int rank;
                index loc;
                pLGlobalMapping->search(i, rank, loc);
                ghostSizes[rank]++;
            }

            pLGhostMapping->allocateGhostIndex(ghostSizes);
            pLGhostMapping->ghost() = std::forward<TPullSet>(pullingIndexGlobal);
            pLGhostMapping->sort();

            // phase1.3: inform each rank which to pull while being informed of which to push
            tMPI_intVec pushIndexSizes(mpi.size);
            MPI_Alltoall(ghostSizes.data(), 1, MPI_INT, pushIndexSizes.data(), 1, MPI_INT, mpi.comm);
            tMPI_intVec pushGlobalStart;
            AccumulateRowSize(pushIndexSizes, pushGlobalStart);
            tIndexVec pushingIndexGlobal(pushGlobalStart[pushGlobalStart.size() - 1]);

            MPI_Alltoallv(pLGhostMapping->ghost().data(), ghostSizes.data(), pLGhostMapping->gStarts().data(), DNDS_MPI_INDEX,
                          pushingIndexGlobal.data(), pushIndexSizes.data(), pushGlobalStart.data(), DNDS_MPI_INDEX,
                          mpi.comm);

            // phase2.1: build push sizes and push disps
            tMPI_intVec pushingSizes(pushingIndexGlobal.size());  // pushing sizes in bytes
            tMPI_AintVec pushingDisps(pushingIndexGlobal.size()); // pushing disps in bytes

            for (index i = 0; i < pushingIndexGlobal.size(); i++)
            {
                MPI_int rank;
                index loc;
                bool found = pLGhostMapping->search(pushingIndexGlobal[i], rank, loc);
                assert(found && rank == -1);
                auto indexerRet = indexer[loc];
                pushingDisps[i] = std::get<0>(indexerRet);
                pushingSizes[i] = std::get<1>(indexerRet);
            }

            // phase2.2: be informed of pulled sub-indexer
            ghostIndexer.buildAsGhostAlltoall(indexer, pushingSizes,
                                              pushIndexSizes, pushGlobalStart,
                                              ghostSizes, *pLGhostMapping, mpi);

            // InsertCheck(mpi);
            // std::cout << mpi.rank << " VEC ";
            // PrintVec(pushingSizes, std::cout);
            // std::cout << std::endl;
            // InsertCheck(mpi);

            // phase3: create and register MPI types of pushing and pulling
            pPushTypeVec = std::make_shared<tMPI_typePairVec>(0);
            pPullTypeVec = std::make_shared<tMPI_typePairVec>(0);
            for (MPI_int r = 0; r < mpi.size; r++)
            {
                // push
                MPI_int pushNumber = pushIndexSizes[r];
                if (pushNumber > 0)
                {
                    MPI_Aint *pPushDisps = pushingDisps.data() + pushGlobalStart[r];
                    MPI_int *pPushSizes = pushingSizes.data() + pushGlobalStart[r];
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
                auto gRbyte = ghostIndexer(index(pLGhostMapping->gStarts()[r + 1]));
                auto gLbyte = ghostIndexer(index(pLGhostMapping->gStarts()[r]));

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
            auto nReqs = pPullTypeVec->size() + pPushTypeVec->size();
            PullReqVec.resize(nReqs, (MPI_REQUEST_NULL)), PullStatVec.resize(nReqs);
            PushReqVec.resize(nReqs, (MPI_REQUEST_NULL)), PushStatVec.resize(nReqs);

            dataGhost.resize(ghostIndexer.LengthByte(), 1);
            // std::cout << "Resize Ghost" << dataGhost.size() << std::endl;
        }

        void initPersistentPush()
        {
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
        }

        void initPersistentPull()
        {
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
        }

        void startPersistentPush() { MPI_Startall(PushReqVec.size(), PushReqVec.data()); }
        void startPersistentPull() { MPI_Startall(PullReqVec.size(), PullReqVec.data()); }

        void waitPersistentPush() { MPI_Waitall(PushReqVec.size(), PushReqVec.data(), PushStatVec.data()); }
        void waitPersistentPull() { MPI_Waitall(PullReqVec.size(), PullReqVec.data(), PullStatVec.data()); }
    };
}