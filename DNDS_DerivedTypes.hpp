#pragma once
#include "DNDS_BasicTypes.hpp"
#include "Eigen/Dense"

// some degraded objects
namespace DNDS
{
    class IndexOne : public Batch<index, 1>
    {
    public:
        using Batch<index, 1>::Batch;
        operator index &()
        {
            return Batch<index, 1>::operator[](0);
        }
    };
}

// some dense eigen objects
namespace DNDS
{
    template <uint32_t v_siz>
    class VecStaticBatch : public Batch<real, v_siz>
    {
    public:
        typedef Batch<real, v_siz> tBase;
        using  Batch<real, v_siz>::Batch;
        Eigen::Map<Eigen::Vector<real, v_siz>> p()
        {
            return Eigen::Map<Eigen::Vector<real, v_siz>>(tBase::data);
        }
    };
    typedef VecStaticBatch<3> Vec3DBatch;

    template <uint32_t row_siz, uint32_t col_siz>
    class MatStaticBatch : public Batch<real, row_siz * col_siz>
    {
    public:
        typedef Batch<real, row_siz * col_siz> tBase;
        using tBase::Batch;
        Eigen::Map<Eigen::Matrix<real, row_siz, col_siz>> m()
        {
            return Eigen::Map<Eigen::Matrix<real, row_siz, col_siz>>(tBase::data);
        }
    };
    typedef MatStaticBatch<3, 3> Mat3DBatch;

    class VarVector : public VarBatch<real>
    {
    public:
        using VarBatch<real>::VarBatch;
        Eigen::Map<Eigen::Vector<real, -1>> p()
        {
            return Eigen::Map<Eigen::Vector<real, -1>>(data, _size);
        }

        Eigen::Map<Eigen::Matrix<real, -1, -1>> m_by_ncol(index ncol)
        {
            return Eigen::Map<Eigen::Matrix<real, -1, -1>>(data, _size/ncol, ncol);
        }
    };

    class UniVector : public UniBatch<real>
    {
    public:
        using UniBatch<real>::UniBatch;
        Eigen::Map<Eigen::Vector<real, -1>> p()
        {
            return Eigen::Map<Eigen::Vector<real, -1>>(data, Bsize);
        }

        Eigen::Map<Eigen::Matrix<real, -1, -1>> m_by_ncol(index ncol)
        {
            // assert(Bsize % ncol == 0);
            return Eigen::Map<Eigen::Matrix<real, -1, -1>>(data, Bsize/ncol, ncol);
        }
    };

    // mat size of var * vsize
    template <uint32_t vsize>
    class SemiVarMatrix : public VarBatch<real>
    {
    public:
        using VarBatch<real>::VarBatch;
        Eigen::Map<Eigen::Matrix<real, -1, -1, Eigen::ColMajor>> m()
        {
            return Eigen::Map<Eigen::Matrix<real, -1, -1, Eigen::ColMajor>>(data, _size/vsize, vsize);
        }
    };

    /**
     * @brief autonomous matrix batch, not using any extra context information
     *
     */
    class SmallMatricesBatch : public VarBatch<uint8_t>
    {
    public:
        typedef int32_t tMatIndex;

        static const tMatIndex __size_of_index_element = sizeof(tMatIndex);
        static const tMatIndex __size_of_data_element = sizeof(real);

        /**
         * @brief derive the full size in bytes for the whole matrix batch
         * note that the elem size input for context constructor receives unit in bytes
         * \param matRowsCols matRowsCols[i * 2 +0],matRowsCols[i * 2 +1] is ith matrix size
         */
        template <class TArray>
        static tMatIndex predictSize(tMatIndex nMats, const TArray &matRowsCols)
        {
            tMatIndex indexFieldSize = (1 + nMats * 2) * __size_of_index_element;
            tMatIndex dataFieldSize = 0;
            for (tMatIndex i = 0; i < nMats; i++)
                dataFieldSize += (matRowsCols[i * 2 + 0] * matRowsCols[i * 2 + 1]) * __size_of_data_element;
            return dataFieldSize + indexFieldSize;
        }

        template <class TArray>
        static void initializeData(uint8_t *data, tMatIndex nMats, const TArray &matRowsCols)
        {
            *((tMatIndex *)(data)) = nMats;
            tMatIndex *nMij = (tMatIndex *)(data) + 1;
            for (tMatIndex i = 0; i < nMats; i++)
                nMij[i * 2 + 0] = matRowsCols[i * 2 + 0], nMij[i * 2 + 1] = matRowsCols[i * 2 + 1];
        }

        struct Context : public VarBatch<uint8_t>::Context
        {
            using VarBatch<uint8_t>::Context::Context;                      // completely inheriting the base context
            Context(const tRowsizFunc &rowSizes, index newLength) = delete; // apart from the not-initializing constructor
        };

        // using VarBatch<uint8_t>::VarBatch;

    private:
        tMatIndex nM;
        tMatIndex *nMij;
        std::vector<tMatIndex> matStarts; // in bytes

    public:
        SmallMatricesBatch(uint8_t *dataPos, index nsize, const Context &context, index i) // altering constructor
            : VarBatch<uint8_t>(dataPos, nsize, context, i)                                // must use base class construct on beforehand
        {
            ConstructOn_Extra();
        }

        inline void ConstructOn(uint8_t *dataPos, index nsize, const Context &context, index i)
        {
            VarBatch<uint8_t>::ConstructOn(dataPos, nsize, context, i);
            ConstructOn_Extra();
        }

        inline void ConstructOn_Extra()
        {
            nM = *((tMatIndex *)(data)); // c-language abstraction...
            nMij = (tMatIndex *)(data) + 1;
            assert(nM >= 0 && nM < 1024); // magical constraint for the sake of performance
            matStarts.resize(nM);
            if (nM == 0) // !warning: currently depends on the initialized data == bit 0, should add dumb initializer option?
                return;
            tMatIndex curPlc = (nM * 2 + 1) * __size_of_index_element;
            for (tMatIndex i = 0; i < nM; i++)
            {
                matStarts[i] = curPlc;
                curPlc += (nMij[i * 2 + 0] * nMij[i * 2 + 1]) * __size_of_data_element;
            }
            // std::cout << _size << " " << curPlc << " " << nM << std::endl;
            assert(_size == curPlc); // to not use overlapping data or underlapping data
        }

        template <class TArray>
        inline void Initialize(tMatIndex nMats, const TArray &matRowsCols)
        {
            *((tMatIndex *)(data)) = nMats;
            nM = *((tMatIndex *)(data));
            nMij = (tMatIndex *)(data) + 1;
            for (tMatIndex i = 0; i < nM; i++)
                nMij[i * 2 + 0] = matRowsCols[i * 2 + 0], nMij[i * 2 + 1] = matRowsCols[i * 2 + 1];
            ConstructOn_Extra();
        }

        tMatIndex getNMat()
        {
            return nM;
        }

        Eigen::Map<Eigen::Matrix<real, -1, -1>> m(tMatIndex iMat)
        {
            assert(iMat >= 0 && iMat < nM);
            return Eigen::Map<Eigen::Matrix<real, -1, -1>>((real *)(data + matStarts[iMat]), nMij[iMat * 2 + 0], nMij[iMat * 2 + 1]);
        }
    };

}