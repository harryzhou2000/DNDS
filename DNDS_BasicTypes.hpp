#pragma once

#include "DNDS_Defines.h"

#include <string>
#include <iostream>
#include <memory>
#include <functional>

namespace DNDS
{
    // typedef std::function<indexerPair(indexerPair)> indexerPairModFunc;
    // const indexerPairModFunc indexerPairModFunc_Uniform =
    //     [](indexerPair in) -> indexerPair
    // { return in; };
}

// A constant size of batch of Ts, DNDS component type
namespace DNDS
{
    template <class T, uint32_t Bsize>
    class Batch
    {
        T *data;

    public:
        static std::string Tname;

        struct Context
        {
            static std::string Tname;

            index Length;

            Context(int nLength) : Length(nLength) {}
        };

        struct Indexer
        {
            static const uint32_t BsizeByte = Bsize * sizeof(T);
            static std::string Tname;

            index Length;

            Indexer(const Context &context) : Length(context.Length) {}

            constexpr inline std::tuple<index, index> operator[](index i)
            {
                assert(i < Length);
                assert(i >= 0);
                return std::make_tuple<index, index>(index(BsizeByte * i), index(BsizeByte));
            }

            constexpr inline int LengthByte() { return Length * BsizeByte; }
        };

        Batch(uint8_t *dataPos, index size, const Context &context, index i)
        {
            ConstructOn(dataPos, size, context, i);
        }

        inline void ConstructOn(uint8_t *dataPos, index size, const Context &context, index i)
        {
            data = (T *)(dataPos);
        }

        T &operator[](uint32_t i)
        {
            assert(i >= 0 && i < Bsize);
            return data[i];
        }

        index size() { return Bsize; }

        friend std::ostream &operator<<(std::ostream &out, const Batch &rb)
        {
            for (uint32_t i = 0; i < Bsize; i++)
                out << rb.data[i] << outputDelim;
            return out;
        }
    };

    template <class T, uint32_t Bsize>
    std::string Batch<T, Bsize>::Tname = std::string(typeid(T).name()) + " Batch<" + std::to_string(Bsize) + ">";

    template <class T, uint32_t Bsize>
    std::string Batch<T, Bsize>::Indexer::Tname = Batch<T, Bsize>::Tname + " Indexer";

    template <class T, uint32_t Bsize>
    std::string Batch<T, Bsize>::Context::Tname = Batch<T, Bsize>::Tname + " Context";
}

/*









*/

namespace DNDS
{
    void AccumulateRowSize(const tRowsizeVec &rowsizes, tIndexVec &rowstarts)
    {
        rowstarts.resize(rowsizes.size() + 1);
        rowstarts[0] = 0;
        for (index i = 1; i < rowstarts.size(); i++)
            rowstarts[i] = rowstarts[i - 1] + rowsizes[i - 1];
    }
}

/*









*/

// A varied size of batch of Ts, DNDS component type
namespace DNDS
{
    typedef std::function<rowsize(index)> tRowsizFunc;

    struct IndexPairUniformFunc
    {
        indexerPair operator()(indexerPair in) const { return in; }
    };

    template <class T, class IndexModder = IndexPairUniformFunc>
    class VarBatch
    {
        T *data;
        index _size; // in unit of # T instances
        static const IndexModder indexModder;

    public:
        static std::string Tname;

        typedef tpIndexVec tpRowstart;

        struct Context
        {
            static std::string Tname;

            index Length;
            tpRowstart pRowstart; // unit in bytes

            // initializing using rowstart table unit in bytes
            // dummy int to not confuse with the moving constructor
            template <class TtpRowstart>
            Context(int dummy, TtpRowstart &&npRowstart) : Length(npRowstart->size() - 1), pRowstart(std::forward<TtpRowstart>(npRowstart)) {}


            // rowSizes unit in bytes
            Context(const tRowsizFunc &rowSizes, index newLength) : Length(newLength)
            {
                pRowstart.reset(); // abandon any hooked row info
                pRowstart = std::make_shared<tIndexVec>(new tIndexVec(Length + 1));
                (*pRowstart)[0] = 0;
                for (index i = 0; i < Length; i++)
                    (*pRowstart)[i + 1] = rowSizes(i) + (*pRowstart)[i];
            }
        };

        struct Indexer
        {
            static std::string Tname;

            index Length;
            tpRowstart pRowstart; // unit in bytes

            Indexer(const Context &context) : Length(context.Length), pRowstart(context.pRowstart) {}

            constexpr inline std::tuple<index, index> operator[](index i)
            {
                assert(i < Length);
                assert(i >= 0);
                return indexModder(
                    std::make_tuple<index, index>(
                        index((*pRowstart)[i]) * sizeof(T),
                        index((*pRowstart)[i + 1] - (*pRowstart)[i]) * sizeof(T)));
            }

            constexpr inline int LengthByte() { return (*pRowstart)[Length] * sizeof(T); }
        };

        VarBatch(uint8_t *dataPos, index nsize, const Context &context, index i)
        {
            ConstructOn(dataPos, nsize, context, i);
        }

        // nsize is in unit of bytes
        inline void ConstructOn(uint8_t *dataPos, index nsize, const Context &context, index i)
        {
            data = (T *)(dataPos);
            _size = nsize / sizeof(T);
        }

        T &operator[](uint32_t i)
        {
            assert(i >= 0 && i < _size);
            return data[i];
        }

        index size() { return _size; }

        friend std::ostream &operator<<(std::ostream &out, const VarBatch &rb)
        {
            for (uint32_t i = 0; i < rb._size; i++)
                out << rb.data[i] << outputDelim;
            return out;
        }
    };

    template <class T, class IndexModder>
    std::string VarBatch<T, IndexModder>::Tname = std::string(typeid(T).name()) + " VarBatch";

    template <class T, class IndexModder>
    std::string VarBatch<T, IndexModder>::Indexer::Tname = VarBatch<T>::Tname + " Indexer";

    template <class T, class IndexModder>
    std::string VarBatch<T, IndexModder>::Context::Tname = VarBatch<T>::Tname + " Context";

    template <class T, class IndexModder>
    const IndexModder VarBatch<T, IndexModder>::indexModder = IndexModder();
}