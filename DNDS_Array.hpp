#pragma once

#include "DNDS_Defines.h"

#include <tuple>
#include <memory>
#include <string>
#include <vector>

#include "DNDS_MPI.hpp"

namespace DNDS
{
    template <class T>
    class Array
    {

    public:
        std::vector<uint8_t> data;

        MPIInfo mpi;

    public:
        typedef typename T::Indexer tIndexer;
        typedef typename T::Context tContext;
        typedef std::shared_ptr<tIndexer> tpIndexer;
        typedef std::shared_ptr<tContext> tpContext;

        tpIndexer pIndexer;
        tpContext pContext;

        static std::string PrintTypes()
        {
            return "Types: " + T::Indexer::Tname + ", " + T::Context::Tname;
        }

        Array(tpContext &&newContext, const MPIInfo &nmpi) : mpi(nmpi)
        {
            pContext = std::forward<tpContext>(newContext);
            pIndexer = tpIndexer(new tIndexer(*pContext));
            data.resize(pIndexer->LengthByte());
        }

        ~Array()
        {
        }

        T operator[](index i)
        {
            auto indexInfo = (*pIndexer)[i];
            assert(std::get<0>(indexInfo) + std::get<1>(indexInfo) <= data.size());
            assert(std::get<0>(indexInfo) >= 0);
            return T(data.data() + std::get<0>(indexInfo), std::get<1>(indexInfo), *pContext, i);
        }




















        class Iterator
        {
            T instace;
            uint8_t *ptr;
            tpIndexer pIndexer;
            tpContext pContext;
            index i;

        public:
            Iterator(uint8_t *nptr, index ni, tpIndexer &&npIndexer, tpContext &&npContext)
                : ptr(nptr), i(ni), pIndexer(std::forward(npIndexer)),
                  pContext(std::forward(npContext))
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

            T& operator*()
            {
                assert(i < pIndexer->Length);
                return instace;
            }
        };
    };

}