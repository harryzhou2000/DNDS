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
    protected:
        T *data;

    public:
        static std::string Tname;

        struct Context
        {
            static std::string Tname;

            index Length;

            Context() : Length(indexMin){};
            Context(int nLength) : Length(nLength) {}
        };

        struct Indexer
        {
            static const uint32_t BsizeByte = Bsize * sizeof(T);
            static std::string Tname;

            index Length;
            Indexer() : Length(indexMin) {} // null constructor
            Indexer(const Context &context) : Length(context.Length) {}

            /// \brief returns start index and length in byte
            constexpr inline std::tuple<index, index> operator[](index i) const
            {
                assert(i < Length);
                assert(i >= 0);
                return std::make_tuple<index, index>(index(BsizeByte * i), index(BsizeByte));
            }

            /// \brief returns start index in byte
            constexpr inline index operator()(index i) const
            {
                assert(i <= Length);
                assert(i >= 0);
                return index(BsizeByte * i);
            }

            /// \brief returns the overall length in byte, allows past-the end input
            constexpr inline int LengthByte() const { return Length * BsizeByte; }

            // LGhostMapping is actually const here
            void buildAsGhostAlltoall(const Indexer &indexer, const tMPI_intVec &pushingSizes, // pushingSizes in in bytes
                                      OffsetAscendIndexMapping &LGhostMapping,                 // pulling side structure
                                      const MPIInfo &mpi)
            {
                for (auto i : pushingSizes)
                {
                    assert(Bsize == i);
                }
                assert(mpi.size == LGhostMapping.ghostStart.size() - 1);
                Length = LGhostMapping.ghostStart[LGhostMapping.ghostStart.size() - 1];
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

// A varied size of batch of Ts, DNDS component type
namespace DNDS
{
    typedef std::function<rowsize(index)> tRowsizFunc;

    // struct IndexPairUniformFunc
    // {
    //     /// \arg size: size in bytes \return modified size in bytes
    //     inline index operator()(index size) const { return size; }                  // from stored to actual
    //     /// \arg in: (disp, size) in bytes, i: index in n-elements \return modified (disp, size) in bytes
    //     inline indexerPair operator()(indexerPair in, index i) const { return in; }  // from stored to actual
    //     /// \arg in: modified size in bytes \return size in bytes
    //     inline index operator[](index size) const { return size; }                    // from actual to stored
    //     // size conversion must meed inverse constraints
    // };

    // template <class T, class IndexModder = IndexPairUniformFunc> //disabled the IndexModder paradigm
    template <class T>
    class VarBatch
    {
    protected:
        T *data;
        index _size; // in unit of # T instances
        // static const IndexModder indexModder; //disabled the IndexModder paradigm

    public:
        static std::string Tname;

        typedef tpIndexVec tpRowstart;

        struct Context
        {
            static std::string Tname;

            index Length;
            tpRowstart pRowstart; // unit in bytes

            Context() : Length(indexMin){};

            // initializing using rowstart table unit in bytes
            // dummy int to not confuse with the moving constructor
            template <class TtpRowstart>
            Context(int dummy, TtpRowstart &&npRowstart) : Length(npRowstart->size() - 1), pRowstart(std::forward<TtpRowstart>(npRowstart)) {}

            // rowSizes unit in n-Ts!!!
            Context(const tRowsizFunc &rowSizes, index newLength) : Length(newLength)
            {
                // pRowstart.reset(); // abandon any hooked row info // not necessary
                pRowstart = std::make_shared<tIndexVec>(tIndexVec(Length + 1));
                (*pRowstart)[0] = 0;
                for (index i = 0; i < Length; i++)
                    (*pRowstart)[i + 1] = rowSizes(i) * sizeof(T) + (*pRowstart)[i];
            }
        };

        struct Indexer
        {
            static std::string Tname;

            index Length;
            tpRowstart pRowstart; // unit in bytes

            Indexer() : Length(INT64_MIN) {} // null constructor
            Indexer(const Context &context) : Length(context.Length), pRowstart(context.pRowstart) {}

            /// \brief returns start index and length in byte
            constexpr inline std::tuple<index, index> operator[](index i) const
            {
                assert(i < Length);
                assert(i >= 0);
                // return indexModder(
                //     std::make_tuple<index, index>(
                //         index((*pRowstart)[i]),
                //         index((*pRowstart)[i + 1] - (*pRowstart)[i])),
                //     i);                         //disabled the IndexModder paradigm
                return std::make_tuple<index, index>(
                    index((*pRowstart)[i]),
                    index((*pRowstart)[i + 1] - (*pRowstart)[i]));
            }

            /// \brief returns start index in byte, allows past-the end input
            constexpr inline index operator()(index i) const
            {
                assert(i <= Length);
                assert(i >= 0);
                //    return indexModder(
                //     std::make_tuple<index, index>(
                //         index((*pRowstart)[i]),
                //         index((*pRowstart)[i + 1] - (*pRowstart)[i])),
                //     i);                           //disabled the IndexModder paradigm
                return index((*pRowstart)[i]);
            }

            constexpr inline int LengthByte() const
            {
                // return std::get<0>(
                //     indexModder(
                //         std::make_tuple<index, index>(
                //             (*pRowstart)[Length], 0),
                //         Length));            //disabled the IndexModder paradigm
                return (*pRowstart)[Length];
            }

            // LGhostMapping is actually const here
            void buildAsGhostAlltoall(const Indexer &indexer, const tMPI_intVec &pushingSizes, // pushingSizes in in bytes
                                      OffsetAscendIndexMapping &LGhostMapping,                 // pulling side structure
                                      const MPIInfo &mpi)
            {
                assert(mpi.size == LGhostMapping.ghostStart.size() - 1);
                Length = LGhostMapping.ghostStart[LGhostMapping.ghostStart.size() - 1];
                // std::cout << LGhostMapping.gStarts().size() << std::endl;
                pRowstart.reset();
                pRowstart = std::make_shared<tIndexVec>(Length + 1);

                // obtain pulling sizes with pushing sizes
                tMPI_intVec pullingSizes(Length);
                MPI_Alltoallv(pushingSizes.data(), LGhostMapping.pushIndexSizes.data(), LGhostMapping.pushIndexStarts.data(), MPI_INT,
                              pullingSizes.data(), LGhostMapping.ghostSizes.data(), LGhostMapping.ghostStart.data(), MPI_INT,
                              mpi.comm);

                (*pRowstart)[0] = 0;
                // for (index i = 0; i < Length; i++)
                //     (*pRowstart)[i + 1] = (*pRowstart)[i] + indexModder[pullingSizes[i]];// disabled the IndexModder paradigm
                for (index i = 0; i < Length; i++)
                    (*pRowstart)[i + 1] = (*pRowstart)[i] + pullingSizes[i];
                // is actually pulling disps, but is contiguous anyway

                // InsertCheck(mpi);
                // std::cout << mpi.rank << " VEC ";
                // PrintVec(pullingSizes, std::cout);
                // std::cout << std::endl;
                // InsertCheck(mpi);

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
        } // a friend function can be defined within
    };

    // template <class T, class IndexModder>
    // std::string VarBatch<T, IndexModder>::Tname = std::string(typeid(T).name()) + " VarBatch";

    // template <class T, class IndexModder>
    // std::string VarBatch<T, IndexModder>::Indexer::Tname = VarBatch<T>::Tname + " Indexer";

    // template <class T, class IndexModder>
    // std::string VarBatch<T, IndexModder>::Context::Tname = VarBatch<T>::Tname + " Context";

    // template <class T, class IndexModder>
    // const IndexModder VarBatch<T, IndexModder>::indexModder = IndexModder(); // disabled the IndexModder paradigm

    template <class T>
    std::string VarBatch<T>::Tname = std::string(typeid(T).name()) + " VarBatch";

    template <class T>
    std::string VarBatch<T>::Indexer::Tname = VarBatch<T>::Tname + " Indexer";

    template <class T>
    std::string VarBatch<T>::Context::Tname = VarBatch<T>::Tname + " Context";
}

namespace DNDS
{

    typedef VarBatch<real> VReal;
    typedef Batch<real, 1> Real1;
    typedef Batch<real, 2> Real2;
    typedef Batch<real, 3> Real3;
}