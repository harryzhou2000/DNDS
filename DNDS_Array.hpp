#pragma once

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "DNDS_Defines.h"
#include "DNDS_MPI.hpp"
#include "DNDS_IndexMapping.hpp"
#include "DNDS_BasicTypes.hpp"
#include "DNDS_DerivedTypes.hpp"

namespace DNDS
{

    namespace ArrayInternal
    {
        template <typename T>
        class ArrayDataContainer
        {
            T *_data = nullptr;
            index _size = 0;
            bool _own = true;

        public:
            ArrayDataContainer() {}

            ArrayDataContainer(const ArrayDataContainer &R)
            {
                resize(R.size());
                for (index i = 0; i < R.size(); i++)
                    _data[i] = R._data[i];
            }
            ArrayDataContainer(ArrayDataContainer &&R)
            {
                if (_data && _own)
                    delete[] _data;
                _data = R._data;
                R._data = nullptr;
                R._size = 0;
            }
            ~ArrayDataContainer()
            {
                if (_data && _own)
                    delete[] _data;
                _data = nullptr;
                _size = 0;
            }
            void operator=(const ArrayDataContainer &R)
            {
                resize(R.size());
                for (index i = 0; i < R.size(); i++)
                    _data[i] = R._data[i];
            }
            void operator=(ArrayDataContainer &&R)
            {
                if (_data && _own)
                    delete[] _data;
                _data = R._data;
                R._data = nullptr;
            }
            void connectWith(ArrayDataContainer &R)
            {
                index sumSize = size() + R.size();
                uint8_t* _data_old = _data;
                uint8_t* _Rdata_old = R._data;
                if (sumSize)
                    _data = new T[sumSize];
                R._data = _data + size();
                R._own = false;
                for(index i = 0; i<size(); i++)
                    _data[i] = _data_old[i];
                for(index i = 0; i<R.size(); i++)
                    R._data[i] = _Rdata_old[i];

                delete[] _data_old;
                delete[] _Rdata_old;
            }

            void resize(index nsize, T fill = 0)
            {
                _own = true;
                if (_data && _own)
                    delete[] _data;
                if (nsize)
                    _data = new T[nsize];
                _size = nsize;
                for (index i = 0; i < _size; i++)
                    _data[i] = fill;
            }
            index size() const
            {
                return _size;
            }
            T *data() const
            {
                return _data;
            }
        };
    }
}

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
        friend std::ostream &operator<<(std::ostream &out, const ArrayCommStat &a)
        {
            out << a.hasGlobalMapping << " " << a.hasGhostMapping << " " << a.hasGhostIndexer << " " << a.hasCommTypes << " "
                << a.hasPersistentPullReqs << " " << a.PersistentPullFinished << " " << a.hasPersistentPushReqs << " " << a.PersistentPushFinished;
            return out;
        }
    };

    template <class T>
    class ArraySingle
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
        ArraySingle(TtContext &&newContext, const MPIInfo &nmpi)
            : context(std::forward<TtContext>(newContext)),
              indexer(context), ghostIndexer(), mpi(nmpi)
        {
            data.resize(indexer.LengthByte(), 0);
            // std::cout << "IndexerLengthByte " << indexer.LengthByte() << std::endl;
            // ghostIndexer is initialized as empty
        }
        /******************************************************************************************************************************/

        /******************************************************************************************************************************/
        /**
         *  \brief  copy constructor: similar with automatic, data is also copied, not copying any ReqVec or StatVec
         *  \warning To do comm-s, must InitPersistent... -> start/wait
         */
        ArraySingle(const ArraySingle<T> &Rarray)
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
        ArraySingle(TtContext &&newContext, const ArraySingle<TR> &Rarray)
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
            data.resize(indexer.LengthByte(), 0);
        }
        /******************************************************************************************************************************/

        // template <class TtContext>
        // Array(const Array<T> &R, TtContext &&newContext, const MPIInfo &nmpi)
        //     : context(std::forward<TtContext>(newContext)),
        //       indexer(context),
        // {
        // }

        ~ArraySingle()
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
                // std::cout << found << " " << rank << std::endl;
                assert(found && rank == -1); // must be at local main

                auto indexerRet = indexer[loc];
                pushingDisps[i] = std::get<0>(indexerRet); // pushing disps are not contiguous
                pushingSizes[i] = std::get<1>(indexerRet);
            }

            // phase2.2: be informed of pulled sub-indexer
            // equals to: building pullingSizes and pullingDisps, bytes size and disps of ghost
            ghostIndexer.buildAsGhostAlltoall(indexer, pushingSizes, *pLGhostMapping, mpi);
            dataGhost.resize(ghostIndexer.LengthByte(), 0); // data section must correspond to indexer
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
        void clearPersistentPush() { PushReqVec.clear(); }
        void clearPersistentPull() { PullReqVec.clear(); }
    };

    template <class T>
    class ArrayCascade
    {

    public:
        // std::vector<uint8_t> data;
        ArrayInternal::ArrayDataContainer<uint8_t> data;
        // std::vector<uint8_t> dataGhost;
        MPIInfo mpi;

        /******************************************************************************************************************************/
        // ** the father array to pull from or push to **
        // ** the user is partly responsible for maintaining the validity of the naked pointer **
        // can only have one father, but multiple sons; son points to current son which is used as ghost part data
        ArrayCascade<T> *father = nullptr;
        ArrayCascade<T> *son = nullptr;
        /******************************************************************************************************************************/

    public:
        /******************************************************************************************************************************/
        // basic aux info
        typedef typename T::Indexer tIndexer;
        typedef typename T::Context tContext;
        typedef typename T::tElement tElement;
        typedef T tComponent;
        tContext context;
        tIndexer indexer;

        /******************************************************************************************************************************/

        /******************************************************************************************************************************/
        // ** comm aux info: ghost mapping **
        typedef std::shared_ptr<GlobalOffsetsMapping> tpLGlobalMapping;
        typedef std::shared_ptr<OffsetAscendIndexMapping> tpLGhostMapping;
        tpLGlobalMapping pLGlobalMapping;
        tpLGhostMapping pLGhostMapping;

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

        const MPIInfo &getMPI() { return mpi; }

        /******************************************************************************************************************************/
        /**
         * \brief construct array and its indexer via a new context
         *
         */
        template <class TtContext>
        ArrayCascade(TtContext &&newContext, const MPIInfo &nmpi)
            : context(std::forward<TtContext>(newContext)),
              indexer(context), mpi(nmpi)
        {
            data.resize(indexer.LengthByte(), 0);
            initializeData();
        }
        /******************************************************************************************************************************/

        /******************************************************************************************************************************/
        /**
         * \brief constuct as a son while setting father's current son as this
         * \warning to validate any comm, do comm setting sequence
         */
        ArrayCascade(ArrayCascade<T> *nfather) : mpi(nfather->mpi)
        {
            assert(nfather);
            // std::cout << "Call pointer init " << nfather << std::endl;
            father = nfather;
            father->son = this;
        }
        /******************************************************************************************************************************/

        /******************************************************************************************************************************/
        /**
         *  \brief  copy constructor: similar with automatic, data is also copied, not copying any ReqVec or StatVec
         *  \warning To do comm-s, must InitPersistent... -> start/wait
         */
        ArrayCascade(const ArrayCascade<T> &Rarray, ArrayCascade<T> *nfather = nullptr)
            : data(Rarray.data),
              //   dataGhost(Rarray.dataGhost),
              mpi(Rarray.mpi),
              context(Rarray.context),
              indexer(Rarray.indexer),
              pLGlobalMapping(Rarray.pLGlobalMapping),
              pLGhostMapping(Rarray.pLGhostMapping),
              //   ghostIndexer(Rarray.ghostIndexer),
              pPushTypeVec(Rarray.pPushTypeVec),
              pPullTypeVec(Rarray.pPullTypeVec)
        {
            if (nfather)
            {
                father = nfather;
                father->son = this;
            }
            commStat = Rarray.commStat;
            commStat.hasPersistentPushReqs = false;
            commStat.PersistentPushFinished = true;
            commStat.hasPersistentPullReqs = false;
            commStat.PersistentPullFinished = true;
        }

    private:
        /**
         *  \brief //!deprecated
         *   copyer, comm topology copied, not copying ghostIndexer, not copying any data, comm-type, ReqVec or StatVec
         *  \warning To do comm-s, must createMPITypes -> InitPersistent... -> start/wait
         */
        template <class TtContext, class TR>
        ArrayCascade(TtContext &&newContext, const ArrayCascade<TR> &Rarray, ArrayCascade<T> *nfather = nullptr)
            : data(),
              //   dataGhost(),
              mpi(Rarray.mpi),
              context(std::forward<TtContext>(newContext)),
              indexer(context),
              pLGlobalMapping(Rarray.pLGlobalMapping),
              pLGhostMapping(Rarray.pLGhostMapping),
              //   ghostIndexer(),
              pPushTypeVec(),
              pPullTypeVec()
        {
            data.resize(indexer.LengthByte(), 0);
            if (nfather)
            {
                father = nfather;
                father->son = this;
            }
            commStat = Rarray.commStat;
            commStat.hasCommTypes = false;
            commStat.hasGhostIndexer = false;
            commStat.hasPersistentPushReqs = false;
            commStat.PersistentPushFinished = true;
            commStat.hasPersistentPullReqs = false;
            commStat.PersistentPullFinished = true;
            initializeData();
        }

    public:
        /******************************************************************************************************************************/

        // template <class TtContext>
        // Array(const Array<T> &R, TtContext &&newContext, const MPIInfo &nmpi)
        //     : context(std::forward<TtContext>(newContext)),
        //       indexer(context),
        // {
        // }

        ~ArrayCascade()
        {
            if (son)
            {
                son->father = nullptr;
                son = nullptr;
            }
            if (father)
            {
                if (father->son == this)
                    father->son = nullptr;
                father = nullptr;
            }
        }

        void initializeData()
        {
            if (!context.needInitialize)
                return;
            for (index i = 0; i < size(); i++)
            {
                auto indexInfo = indexer[i];
                assert(std::get<0>(indexInfo) + std::get<1>(indexInfo) <= data.size());
                assert(std::get<0>(indexInfo) >= 0);
                context.fInit((tElement *)(data.data() + std::get<0>(indexInfo)),
                              std::get<1>(indexInfo),
                              i);
            }
        }

        void connectWith(ArrayCascade<T> &other)
        {
            data.connectWith(other.data);
        }

        // use local index to get T data
        T
        operator[](index i)
        {
            auto indexInfo = indexer[i];
            assert(std::get<0>(indexInfo) + std::get<1>(indexInfo) <= data.size());
            assert(std::get<0>(indexInfo) >= 0);
            return T(data.data() + std::get<0>(indexInfo), std::get<1>(indexInfo), context, i);
        }

        // use as father; use local index at ghost to get T data
        // not recommended
        T indexGhostData(index i)
        {
            assert(son); // must have son
            auto &dataGhost = son->data;
            auto &ghostIndexer = son->indexer;
            auto indexInfo = ghostIndexer[i];
            assert(std::get<0>(indexInfo) + std::get<1>(indexInfo) <= dataGhost.size());
            assert(std::get<0>(indexInfo) >= 0);
            return T(dataGhost.data() + std::get<0>(indexInfo), std::get<1>(indexInfo), context, i);
        }

        index size() const { return indexer.Length; } // in unit of T instances
        index sizeGhost() const                       // in unit of T instances
        {
            assert(son); // must have son
            return son->indexer.Length;
        };
        index sizeByte() const { return indexer.LengthByte(); } // in unit of bytes
        index sizeByteGhost() const                             // in unit of bytes
        {
            assert(son); // must have son
            return son->indexer.LengthByte();
        }

        void SwitchSon(ArrayCascade<T> *nson)
        {
            assert(nson->father == this);
            son = nson;
        }

        void ForgetFather()
        {
            father = nullptr;
            clearPersistentPull();
            clearPersistentPush();
            clearMPITypes();
            clearGhostMapping();
            clearGlobalMapping();
        }

        /**
         * @brief think clear before using, using same comm topology
         * \post a counted share of global and ghost mappings
         */
        template <class TR>
        void BorrowGGIndexing(const ArrayCascade<TR> &Rarray)
        {
            // assert(father && Rarray.father); // Rarray's father is not visible...
            // assert(father->obtainTotalSize() == Rarray.father->obtainTotalSize());
            assert(Rarray.pLGhostMapping && Rarray.pLGlobalMapping);
            assert(Rarray.commStat.hasGhostMapping && Rarray.commStat.hasGlobalMapping);
            pLGhostMapping = Rarray.pLGhostMapping;
            pLGlobalMapping = Rarray.pLGlobalMapping;
            commStat.hasGhostMapping = commStat.hasGlobalMapping = true;
        }

        // *** warning: cascade arrays' comm set sequence are for sons/ghosts side to execute***
        /******************************************************************************************************************************/
        // recommend: TPullSet == tIndexVec (& &&),
        // // has to be able to be copied into or moved operator= int tIndexVec
        // has to be able to be range - iterated
        /**
         * \brief from a set of global pulling indexes, create pLGlobalMapping pLGhostMapping
         * \post pLGlobalMapping pLGhostMapping established
         */

        void createGlobalMapping() // collective;
        {
            assert(bool(father)); // has to be a son
            // phase1.1: create localGlobal mapping (broadcast)
            pLGlobalMapping = std::make_shared<GlobalOffsetsMapping>();
            pLGlobalMapping->setMPIAlignBcast(mpi, father->size()); // cascade from father
            commStat.hasGlobalMapping = true;
        }

        template <class TPullSet>
        void createGhostMapping(TPullSet &&pullingIndexGlobal) // collective;
        {
            assert(bool(father)); // has to be a son
            assert(commStat.hasGlobalMapping);
            // phase1.2: count how many to pull and allocate the localGhost mapping, fill the mapping
            // counting could overflow
            // tMPI_intVec ghostSizes(mpi.size, 0); // == pulling sizes
            pLGhostMapping = std::make_shared<OffsetAscendIndexMapping>(
                (*pLGlobalMapping)(mpi.rank, 0), father->size(),
                std::forward<TPullSet>(pullingIndexGlobal),
                *pLGlobalMapping,
                mpi);
            commStat.hasGhostMapping = true;
        }

        template <class TPushSet, class TPushStart>
        void createGhostMapping(TPushSet &&pushingIndexLocal, TPushStart &&pushStarts) // collective;
        {
            assert(bool(father)); // has to be a son
            assert(commStat.hasGlobalMapping);
            // phase1.2: count how many to pull and allocate the localGhost mapping, fill the mapping
            // counting could overflow
            pLGhostMapping = std::make_shared<OffsetAscendIndexMapping>(
                (*pLGlobalMapping)(mpi.rank, 0), father->size(),
                std::forward<TPushSet>(pushingIndexLocal),
                std::forward<TPushStart>(pushStarts),
                *pLGlobalMapping,
                mpi);
            commStat.hasGhostMapping = true;
        }
        /******************************************************************************************************************************/

        /******************************************************************************************************************************/
        /**
         * \brief get real element byte info into account, with ghost indexer and comm types built, need two mappings
         * \pre has pLGlobalMapping pLGhostMapping
         * \post my indexer pPullTypeVec pPushTypeVec established
         */
        void createMPITypes() // collective;
        {
            if (!father)
            {
                std::cout << "\n\n\nRank " << mpi.rank << " \n"
                          << father << std::endl;
            }
            assert(bool(father));

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
                // std::cout << found << " " << rank << std::endl;
                assert(found && rank == -1); // must be at local main

                auto indexerRet = father->indexer[loc];    // cascade from father
                pushingDisps[i] = std::get<0>(indexerRet); // pushing disps are not contiguous
                pushingSizes[i] = std::get<1>(indexerRet);
            }
            // PrintVec(pushingSizes, std::cout);
            // std::cout << std::endl;

            // phase2.2: be informed of pulled sub-indexer
            // equals to: building pullingSizes and pullingDisps, bytes size and disps of ghost
            indexer.buildAsGhostAlltoall(indexer, pushingSizes, *pLGhostMapping, mpi); // cascade from father
            data.resize(indexer.LengthByte(), 0);                                      // data section must correspond to indexer  // cascade from father
                                                                                       // std::cout << "Resize Ghost" << dataGhost.size() << std::endl;
            commStat.hasGhostIndexer = true;                                           // note that "hasGhostIndexer" means I as son have indexer

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
                // std::cout << "PN" << pushNumber << std::endl;
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
                auto gRbyte = indexer(index(pLGhostMapping->ghostStart[r + 1])); // cascade from father
                auto gLbyte = indexer(index(pLGhostMapping->ghostStart[r]));     // cascade from father

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
            // std::cout << "NPull" << pPullTypeVec->size() << std::endl;
        }
        /******************************************************************************************************************************/

        /******************************************************************************************************************************/
        /**
         * \brief when established push and pull types, init persistent-nonblocked-nonbuffered MPI reqs
         * \pre has pPullTypeVec pPushTypeVec
         * \post PushReqVec established
         * \warning after init, raw buffers of data for both father and son/ghost should remain static
         */
        void initPersistentPush() // collective;
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
                MPI_Send_init(data.data(), 1, dtypeInfo.second, rankOther, tag, mpi.comm, PushReqVec.data() + ip);
                // cascade from father
            }
            for (auto ip = 0; ip < pPushTypeVec->size(); ip++)
            {
                auto dtypeInfo = (*pPushTypeVec)[ip];
                MPI_int rankOther = dtypeInfo.first;
                MPI_int tag = rankOther + mpi.rank;
                MPI_Recv_init(father->data.data(), 1, dtypeInfo.second, rankOther, tag, mpi.comm, PushReqVec.data() + pPullTypeVec->size() + ip);
                // cascade from father
            }
            commStat.hasPersistentPushReqs = true;
        }
        /******************************************************************************************************************************/

        /******************************************************************************************************************************/
        /**
         * \brief when established push and pull types, init persistent-nonblocked-nonbuffered MPI reqs
         * \pre has pPullTypeVec pPushTypeVec
         * \post PullReqVec established
         * \warning after init, raw buffers of data for both father and son/ghost should remain static
         */
        void initPersistentPull() // collective;
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
                MPI_int tag = rankOther + mpi.rank; //! receives a lot of messages, this distinguishes them
                // std::cout << mpi.rank << " Recv " << rankOther << std::endl;
                MPI_Recv_init(data.data(), 1, dtypeInfo.second, rankOther, tag, mpi.comm, PullReqVec.data() + ip);
                // std::cout << *(real *)(dataGhost.data() + 8 * 0) << std::endl;
                // cascade from father
            }
            for (auto ip = 0; ip < pPushTypeVec->size(); ip++)
            {
                auto dtypeInfo = (*pPushTypeVec)[ip];
                MPI_int rankOther = dtypeInfo.first;
                MPI_int tag = rankOther + mpi.rank;
                // std::cout << mpi.rank << " Send " << rankOther << std::endl;
                MPI_Send_init(father->data.data(), 1, dtypeInfo.second, rankOther, tag, mpi.comm, PullReqVec.data() + pPullTypeVec->size() + ip);
                // std::cout << *(real *)(data.data() + 8 * 1) << std::endl;
                // cascade from father
            }
            commStat.hasPersistentPullReqs = true;
        }
        /******************************************************************************************************************************/

        void startPersistentPush() // collective;
        {
            assert(commStat.hasPersistentPushReqs && commStat.PersistentPushFinished);
            if (PushReqVec.size())
                MPI_Startall(PushReqVec.size(), PushReqVec.data());
            commStat.PersistentPushFinished = false;
        }
        void startPersistentPull() // collective;
        {
            assert(commStat.hasPersistentPullReqs && commStat.PersistentPullFinished);
            if (PullReqVec.size())
                MPI_Startall(PullReqVec.size(), PullReqVec.data());
            commStat.PersistentPullFinished = false;
        }

        void waitPersistentPush() // collective;
        {
            assert(commStat.hasPersistentPushReqs && !commStat.PersistentPushFinished);
            if (PushReqVec.size())
                MPI_Waitall(PushReqVec.size(), PushReqVec.data(), PushStatVec.data());
            commStat.PersistentPushFinished = true;
        }
        void waitPersistentPull() // collective;
        {
            assert(commStat.hasPersistentPullReqs && !commStat.PersistentPullFinished);
            if (PullReqVec.size())
                MPI_Waitall(PullReqVec.size(), PullReqVec.data(), PullStatVec.data());
            commStat.PersistentPullFinished = true;
        }

        void clearPersistentPush() // collective;
        {
            assert(commStat.PersistentPushFinished);
            PushReqVec.clear(); // stat vec is left untouched here
            commStat.hasPersistentPushReqs = false;
        }
        void clearPersistentPull() // collective;
        {
            assert(commStat.PersistentPullFinished);
            PullReqVec.clear();
            commStat.hasPersistentPullReqs = false;
        }

        void clearMPITypes() // collective;
        {
            assert(!commStat.hasPersistentPullReqs && !commStat.hasPersistentPushReqs);
            pPullTypeVec.reset();
            pPushTypeVec.reset();
            commStat.hasCommTypes = false;
        }

        void clearGlobalMapping() // collective;
        {
            assert(!commStat.hasGhostMapping);
            pLGlobalMapping.reset();
            commStat.hasGlobalMapping = false;
        }

        void clearGhostMapping() // collective;
        {
            assert(!commStat.hasCommTypes);
            pLGhostMapping.reset();
            commStat.hasGhostMapping = false;
        }

        void pullOnce() // collective;
        {
            initPersistentPull();
            startPersistentPull();
            waitPersistentPull();
            clearPersistentPull();
        }

        void pushOnce() // collective;
        {
            initPersistentPush();
            startPersistentPush();
            waitPersistentPush();
            clearPersistentPush();
        }

        /**
         * @brief collective; should be logically barriered before hand
         *
         */
        void LogStatus(bool printData0 = false)
        {
            if (mpi.rank == 0)
            {
                std::ios::fmtflags originalFlags(log().flags());
                log() << "Comm [" << mpi.comm << "]; Rank [" << mpi.rank << "]; Size [" << size() << "]; Byte [" << sizeByte() << "];\n";
                log() << "\t"
                      << "This [" << this << "]; Father [" << father << "]; Son [" << son << "] \n";
                log() << "\t"
                      << "CommStat [" << commStat << "];\n";
                // if (printData0)
                //     PrintVec<uint8_t, index>(data, std::cout);
                for (int r = 1; r < mpi.size; r++)
                {
                    index thatSizes[2];
                    index ptrs[3];
                    ArrayCommStat thatComm;
                    MPI_Status stat;
                    MPI_Recv(thatSizes, 2, DNDS_MPI_INDEX, r, r + 0 * mpi.size, mpi.comm, &stat);
                    MPI_Recv(ptrs, 3, DNDS_MPI_INDEX, r, r + 1 * mpi.size, mpi.comm, &stat);
                    MPI_Recv(&thatComm, sizeof(ArrayCommStat), MPI_BYTE, r, r + 3 * mpi.size, mpi.comm, &stat);

                    log() << "Comm [" << mpi.comm << "]; Rank [" << r << "]; Size [" << thatSizes[0] << "]; Byte [" << thatSizes[1] << "];\n";
                    log() << "\t"
                          << "This [" << decltype(this)(ptrs[0]) << "]; Father [" << decltype(this)(ptrs[1]) << "]; Son [" << decltype(this)(ptrs[2]) << "] \n";
                    log() << "\t"
                          << "CommStat [" << thatComm << "];\n";
                }
                log() << std::endl;
            }
            else
            {
                index sizes[2];
                index ptrs[3];
                sizes[0] = size(), sizes[1] = sizeByte();
                ptrs[0] = index(this), ptrs[1] = index(father), ptrs[2] = index(son);

                MPI_Send(sizes, 2, DNDS_MPI_INDEX, 0, mpi.rank + 0 * mpi.size, mpi.comm);
                MPI_Send(ptrs, 3, DNDS_MPI_INDEX, 0, mpi.rank + 1 * mpi.size, mpi.comm);
                MPI_Send(&commStat, sizeof(ArrayCommStat), MPI_BYTE, 0, mpi.rank + 3 * mpi.size, mpi.comm);
            }
        }

        /**
         * @brief collective; get the total size & sizebyte
         * should be logically barriered before hand
         */
        index obtainTotalSize() const
        {
            index siz = size();
            index sizeSum;
            MPI_Allreduce(&siz, &sizeSum, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            return sizeSum;
        }
        /**
         * @brief collective; get the total size & sizebyte
         * should be logically barriered before hand
         */
        index obtainTotalSizeByte()
        {
            index siz = sizeByte();
            index sizeSum;
            MPI_Allreduce(&siz, &sizeSum, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            return sizeSum;
        }
    };
}
/*















*/
namespace DNDS
{
    typedef ArrayCascade<IndexOne> IndexArray;

    /**
     * @brief f(T &element, index i)
     *
     */
    template <class T, class Tf>
    void forEachInArray(ArrayCascade<T> &arr, Tf &&f)
    {
        // if (arr.getMPI().rank == 3)
        //     std::cout << "SDFS" << arr.size() << std::endl;
        for (index i = 0; i < arr.size(); i++)
        {
            auto e = arr[i];
            // std::cout << e[0] << std::endl;
            f(e, i);
            // std::cout << "<-" << e[1] << std::endl;
        }
    }

    /**
     * @brief f(T &element, index i, decltype(T[0]) &basic, index j)
     *
     */
    template <class T, class Tf>
    void forEachBasicInArray(ArrayCascade<T> &arr, Tf &&f)
    {
        for (index i = 0; i < arr.size(); i++)
        {
            auto e = arr[i];
            for (index j = 0; j < e.size(); j++)
                f(e, i, e[j], j);
        }
    }

}