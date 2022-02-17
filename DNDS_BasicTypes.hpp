#pragma once

#include "DNDS_Defines.h"
#include "DNDS_MPI.hpp"
#include "DNDS_IndexMapping.hpp"

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

            constexpr inline index operator()(index i)
            {
                assert(i <= Length);
                assert(i >= 0);
                return index(BsizeByte * i);
            }

            constexpr inline int LengthByte() { return Length * BsizeByte; }

            // LGhostMapping is actually const here
            void buildAsGhostAlltoall(Indexer indexer, tMPI_intVec pushingSizes,
                                      tMPI_intVec pushIndexSizes, tMPI_intVec pushGlobalStart,         // pushing side structure
                                      tMPI_intVec ghostSizes, OffsetAscendIndexMapping &LGhostMapping, // pulling side structure
                                      MPIInfo mpi)
            {
                assert(mpi.size == LGhostMapping.gStarts().size() - 1);
                Length = LGhostMapping.gStarts()[LGhostMapping.gStarts().size() - 1];
                // already know that all pushing sizes of all ranks are Bsize
            }
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
    // Note that TtIndexVec being accumulated could overflow
    template <class TtRowsizeVec, class TtIndexVec>
    void AccumulateRowSize(const TtRowsizeVec &rowsizes, TtIndexVec &rowstarts)
    {
        rowstarts.resize(rowsizes.size() + 1);
        rowstarts[0] = 0;
        for (index i = 1; i < rowstarts.size(); i++)
            rowstarts[i] = rowstarts[i - 1] + rowsizes[i - 1];
    }

    template <class T>
    bool checkUniformVector(const std::vector<T> &dat, T &value)
    {
        if (dat.size() == 0)
            return false;
        value = dat[0];
        for (auto i = 1; i < dat.size(); i++)
            if (dat[i] != value)
                return false;
        return true;
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
        index operator()(index size) const { return size; }                  // from stored to actual
        indexerPair operator()(indexerPair in, index i) const { return in; } // from stored to actual
        index operator[](index size) const { return size; }                  // from actual to stored
        // size conversion must meed inverse constraints
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
                pRowstart = std::make_shared<tIndexVec>(tIndexVec(Length + 1));
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
                        index((*pRowstart)[i + 1] - (*pRowstart)[i]) * sizeof(T)),
                    i);
            }

            constexpr inline index operator()(index i)
            {
                assert(i <= Length);
                assert(i >= 0);
                return index((*pRowstart)[i]);
            }

            constexpr inline int LengthByte()
            {
                return std::get<0>(
                    indexModder(
                        std::make_tuple<index, index>(
                            (*pRowstart)[Length] * sizeof(T), 0),
                        Length));
            }

            // LGhostMapping is actually const here
            void buildAsGhostAlltoall(Indexer indexer, tMPI_intVec pushingSizes,                       // pushingSizes in in bytes
                                      tMPI_intVec pushIndexSizes, tMPI_intVec pushGlobalStart,         // pushing side structure
                                      tMPI_intVec ghostSizes, OffsetAscendIndexMapping &LGhostMapping, // pulling side structure
                                      MPIInfo mpi)
            {
                assert(mpi.size == LGhostMapping.gStarts().size() - 1);
                Length = LGhostMapping.gStarts()[LGhostMapping.gStarts().size() - 1];
                pRowstart.reset();
                pRowstart = std::make_shared<tIndexVec>(Length + 1);

                tMPI_intVec pullingSizes(Length);
                MPI_Alltoallv(pushingSizes.data(), pushIndexSizes.data(), pushGlobalStart.data(), MPI_INT,
                              pullingSizes.data(), ghostSizes.data(), LGhostMapping.gStarts().data(), MPI_INT,
                              mpi.comm);

                (*pRowstart)[0] = 0;
                for (index i = 0; i < Length; i++)
                    (*pRowstart)[i + 1] = (*pRowstart)[i] + indexModder[pullingSizes[i]];
                // note that Rowstart and pullingSizes are in bytes
                // pullingSizes is actual but Rowstart is before indexModder(), use indexModder[] to invert
            }
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