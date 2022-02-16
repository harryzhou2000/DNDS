#pragma once

#include <memory>
#include <string>
#include <vector>

#include "DNDS_Defines.h"
#include "DNDS_MPI.hpp"
#include "DNDS_IndexMapping.hpp"

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

        typedef std::shared_ptr<GlobalOffsetsMapping> tpLGlobalMapping;
        typedef std::shared_ptr<OffsetAscendIndexMapping> tpLGhostMapping;

        tpLGlobalMapping pLGlobalMapping;
        tpLGhostMapping pLGhostMapping;

        std::vector<tIndexer>
            ghostIndexerVec;

        static std::string PrintTypes()
        {
            return "Types: " + T::Indexer::Tname + ", " + T::Context::Tname;
        }

        template <class TtContext>
        Array(TtContext &&newContext, const MPIInfo &nmpi) : context(std::forward<TtContext>(newContext)),
                                                             indexer(context), mpi(nmpi)
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

        index size() { return indexer.Length; }

        index sizeByte() { return indexer.LengthByte(); }

        void createGhost(const std::vector<tIndexVec> &pullingIndexGlobal)
        {
            // phase1.1: create localGlobal mapping (broadcast)

            // phase1.2: inform each rank which to pull while being informed of which to push 
        
            // phase2.1: count how many to pull and allocate the localGhost mapping, fill the mapping

            // phase2.2: be informed of pulled sub-indexers, (use mpi's comm requests)
        
            // phase3: create and register MPI types of pushing and pulling
        }

        void initPersistentPush()
        {

        }

        void initPersistentPull()
        {

        }

        void startPush()
        {

        }

        void startPull()
        {

        }

        void waitPush()
        {

        }

        void waitPull()
        {
            
        }

    };
}