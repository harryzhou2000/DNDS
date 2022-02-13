#pragma once

#include "DNDS_Defines.h"
#include <tuple>
#include <string>
#include <iostream>

// A constant size of batch of reals, DNDS component type
namespace DNDS
{
    template <uint32_t Bsize>
    class RealBatch
    {
        real *data;

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
            static const uint32_t BsizeByte = Bsize * sizeof(real);
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

        RealBatch(uint8_t *dataPos, index size, const Context &context, index i)
        {
            ConstructOn(dataPos, size, context, i);
        }

        inline void ConstructOn(uint8_t *dataPos, index size, const Context &context, index i)
        {
            data = (real *)(dataPos);
        }

        real &operator[](uint32_t i)
        {
            assert(i >= 0 && i < Bsize);
            return data[i];
        }

        friend std::ostream &operator<<(std::ostream &out, const RealBatch &rb)
        {
            for (uint32_t i = 0; i < Bsize; i++)
                out << rb.data[i] << outputDelim;
            return out;
        }
    };

    template <uint32_t Bsize>
    std::string RealBatch<Bsize>::Tname = "RealBatch<>";

    template <uint32_t Bsize>
    std::string RealBatch<Bsize>::Indexer::Tname = RealBatch<Bsize>::Tname + " Indexer";

    template <uint32_t Bsize>
    std::string RealBatch<Bsize>::Context::Tname = RealBatch<Bsize>::Tname + " Context";
}