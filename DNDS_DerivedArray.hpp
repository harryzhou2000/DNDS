#pragma once

#include "DNDS_Array.hpp"

namespace DNDS
{

    // pair indexes ghost without referencing father's son, so multiple ghosts are allowed
    template <class T>
    class ArrayPair
    {
        bool connected = false;

    public:
        Array<T> &arr;
        Array<T> &arrGhost;

        std::vector<T> tPrebuild;

        template <class Tarr>
        ArrayPair(Tarr &&nArr, Tarr &&nArrGhost) : arr(std::forward<Tarr>(nArr)), arrGhost(std::forward<Tarr>(nArrGhost))
        {
            DNDS_assert(arrGhost.father == &arr);
            arr.connectWith(arrGhost);
            connected = true;
            tPrebuild.resize(size());
            for (typename decltype(tPrebuild)::size_type i = 0; i < tPrebuild.size(); i++)
            {
                if (i >= arr.size())
                    tPrebuild[i] = arrGhost[i - arr.size()];
                else
                    tPrebuild[i] = arr[i];
            }
        }

        T &operator[](index i)
        {
            DNDS_assert(i >= 0 && i < size());
            // if (i >= arr.size())
            //     return arrGhost[i - arr.size()];
            // return arr[i];
            return tPrebuild[i];
        }

        index size() const
        {
            return arr.size() + arrGhost.size();
        }

        // void connectPair()
        // {
        // }
    };
}

namespace DNDS
{
    /**
     * @brief f(T &element, index i)
     *
     */
    template <class T, class Tf>
    void forEachInArrayPair(ArrayPair<T> &arr, Tf &&f)
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
    void forEachBasicInArrayPair(ArrayPair<T> &arr, Tf &&f)
    {
        for (index i = 0; i < arr.size(); i++)
        {
            auto e = arr[i];
            for (index j = 0; j < e.size(); j++)
                f(e, i, e[j], j);
        }
    }
}

namespace DNDS
{

    template <class T>
    struct ArrayLocal
    {
        typedef Array<T> tArray;
        typedef ArrayPair<T> tPair;
        std::shared_ptr<Array<T>> dist;
        std::shared_ptr<Array<T>> ghost;
        std::shared_ptr<ArrayPair<T>> pair;

        // ArrayLocal

        void Copy(ArrayLocal<T> &R)
        {
            DNDS_assert(&R != this);
            dist = std::make_shared<Array<T>>(*R.dist);
            ghost = std::make_shared<Array<T>>(*R.ghost, dist.get());
            // ghost->BorrowGGIndexing(*R.ghost);
            // ghost->createMPITypes();
            MakePair();
        }

        template <class TR>
        void CreateGhostCopyComm(ArrayLocal<TR> &R)
        {
            DNDS_assert(dist);
            ghost = std::make_shared<Array<T>>(dist.get());
            ghost->BorrowGGIndexing(*R.ghost);
            ghost->createMPITypes();
            MakePair();
        }

        void MakePair()
        {
            DNDS_assert(dist && ghost);
            pair = std::make_shared<ArrayPair<T>>(*dist, *ghost);
        }

        void PullOnce()
        {
            DNDS_assert(pair && dist && ghost);
            ghost->pullOnce();
        }

        void PushOnce()
        {
            DNDS_assert(pair && dist && ghost);
            ghost->pushOnce();
        }

        void InitPersistentPullClean() { ghost->initPersistentPull(); }

        void StartPersistentPullClean() { ghost->startPersistentPull(); }

        void WaitPersistentPullClean() { ghost->waitPersistentPull(); }

        void ClearPersistentPullClean() { ghost->clearPersistentPull(); }

        // index the pair
        inline T &operator[](index i)
        {
            DNDS_assert(pair && dist && ghost);
            // if (i >= 0)
            return pair->operator[](i);
            // else
            //     return pair->operator[](1);
        }

        index size() const
        {
            DNDS_assert(pair && dist && ghost);
            return pair->size();
        }
    };

}

namespace DNDS
{
    template <uint32_t vsize>
    class ArrayDOF : public ArrayLocal<VecStaticBatch<vsize>>
    {
    public:
        typedef ArrayLocal<VecStaticBatch<vsize>> base;
        using ArrayLocal<VecStaticBatch<vsize>>::ArrayLocal;
        ArrayDOF() {}
        ArrayDOF(index distSize, const MPIInfo &mpi)
        {
            base::dist = std::make_shared<Array<VecStaticBatch<vsize>>>(
                typename VecStaticBatch<vsize>::Context(distSize), mpi);
            forEachInArray(*base::dist, [&](VecStaticBatch<vsize> &e, index i)
                           { e.p().setZero(); });
        }

        void resize(index nsize)
        {
            DNDS_assert(base::dist);
            MPIInfo mpi = base::dist->getMPI();
            base::dist = std::make_shared<Array<VecStaticBatch<vsize>>>(
                typename VecStaticBatch<vsize>::Context(nsize), mpi);
            forEachInArray(*base::dist, [&](VecStaticBatch<vsize> &e, index i)
                           { e.p().setZero(); });
        }

        void resize(index nsize, const MPIInfo &mpi)
        {
            base::dist = std::make_shared<Array<VecStaticBatch<vsize>>>(
                typename VecStaticBatch<vsize>::Context(nsize), mpi);
            forEachInArray(*base::dist, [&](VecStaticBatch<vsize> &e, index i)
                           { e.p().setZero(); });
        }

        void setConstant(real v)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](VecStaticBatch<vsize> &e, index i)
                           { e.p().setConstant(v); });
        }

        template <typename Tin>
        void setConstant(const Tin &in)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](VecStaticBatch<vsize> &e, index i)
                           { e.p() = in; });
        }

        void operator=(const ArrayDOF<vsize> &R)
        {
            forEachInArray(*base::dist, [&](VecStaticBatch<vsize> &e, index i)
                           { e.p() = (*R.dist)[i].p(); });
        }

        void operator+=(const ArrayDOF<vsize> &R)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](VecStaticBatch<vsize> &e, index i)
                           { e.p() += (*R.dist)[i].p(); });
        }

        void addTo(const ArrayDOF<vsize> &R, real r)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](VecStaticBatch<vsize> &e, index i)
                           { e.p() += (*R.dist)[i].p() * r; });
        }

        void operator-=(const ArrayDOF<vsize> &R)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](VecStaticBatch<vsize> &e, index i)
                           { e.p() -= (*R.dist)[i].p(); });
        }

        void operator*=(real r)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](VecStaticBatch<vsize> &e, index i)
                           { e.p() *= r; });
        }

        template <class VR>
        void operator*=(const VR &R)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](VecStaticBatch<vsize> &e, index i)
                           { e.p() *= R[i]; });
        }

        /**
         * @brief collective
         *
         */
        real norm2()
        {
            DNDS_assert(base::dist);
            real sqrSum{0}, sqrSumAll;
            forEachInArray(*base::dist, [&](VecStaticBatch<vsize> &e, index i)
                           { sqrSum += e.p().squaredNorm(); });
            MPI_Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, base::dist->getMPI().comm);
            // std::cout << "norm2is " << std::scientific << sqrSumAll << std::endl;
            return std::sqrt(sqrSumAll);
        }

        real dot(const ArrayDOF<vsize> &R)
        {
            DNDS_assert(base::dist);
            real sqrSum{0}, sqrSumAll;
            forEachInArray(*base::dist, [&](VecStaticBatch<vsize> &e, index i)
                           { sqrSum += e.p().dot((*R.dist)[i].p()); });
            MPI_Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, base::dist->getMPI().comm);
            return sqrSumAll;
        }

        // const Eigen::Map<Eigen::Vector<real, vsize>> &operator[](index i)
        // {
        //     return base::operator[](i).p();
        // }

        Eigen::Map<Eigen::Vector<real, vsize>> operator[](index i)
        {
            return base::operator[](i).p();
        }
    };
}

namespace DNDS
{
    template<int vsize = Eigen::Dynamic>
    class ArrayDOFV : public ArrayLocal<UniVector<vsize>>
    {
        
    public:
        typedef ArrayLocal<UniVector<vsize>> base;
        using TElem = UniVector<vsize>;
        using TArray = Array<UniVector<vsize>>;
        using ArrayLocal<UniVector<vsize>>::ArrayLocal;
        ArrayDOFV() {}
        ArrayDOFV(index distSize, const MPIInfo &mpi, index vecSize)
        {
            base::dist = std::make_shared<TArray>(
                typename TElem::Context(
                    distSize, vsize == Eigen::Dynamic ? vecSize : vsize),
                mpi);
            forEachInArray(*base::dist, [&](TElem &e, index i)
                           { e.p().setZero(); });
        }

        void resize(index nsize, index vecSize)
        {
            DNDS_assert(base::dist);
            MPIInfo mpi = base::dist->getMPI();
            base::dist = std::make_shared<TArray>(
                typename TElem::Context(
                    nsize, vsize == Eigen::Dynamic ? vecSize : vsize),
                mpi);
            forEachInArray(*base::dist, [&](TElem &e, index i)
                           { e.p().setZero(); });
        }

        void resize(index nsize, const MPIInfo &mpi, index vecSize)
        {
            base::dist = std::make_shared<TArray>(
                typename TElem::Context(
                    nsize, vsize == Eigen::Dynamic ? vecSize : vsize),
                mpi);
            forEachInArray(*base::dist, [&](TElem &e, index i)
                           { e.p().setZero(); });
        }

        void setConstant(real v)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](TElem &e, index i)
                           { e.p().setConstant(v); });
        }

        template <typename Tin>
        void setConstant(const Tin &in)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](TElem &e, index i)
                           { e.p() = in; });
        }

        void operator=(const ArrayDOFV<vsize> &R)
        {
            forEachInArray(*base::dist, [&](TElem &e, index i)
                           { e.p() = (*R.dist)[i].p(); });
        }

        void operator+=(const ArrayDOFV<vsize> &R)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](TElem &e, index i)
                           { e.p() += (*R.dist)[i].p(); });
        }

        void addTo(const ArrayDOFV<vsize> &R, real r)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](TElem &e, index i)
                           { e.p() += (*R.dist)[i].p() * r; });
        }

        void operator-=(const ArrayDOFV<vsize> &R)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](TElem &e, index i)
                           { e.p() -= (*R.dist)[i].p(); });
        }

        void operator*=(real r)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](TElem &e, index i)
                           { e.p() *= r; });
        }

        template <class VR>
        void operator*=(const VR &R)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](TElem &e, index i)
                           { e.p() *= R[i]; });
        }

        /**
         * @brief collective
         *
         */
        real norm2()
        {
            DNDS_assert(base::dist);
            real sqrSum{0}, sqrSumAll;
            forEachInArray(*base::dist, [&](TElem &e, index i)
                           { sqrSum += e.p().squaredNorm(); });
            MPI_Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, base::dist->getMPI().comm);
            // std::cout << "norm2is " << std::scientific << sqrSumAll << std::endl;
            return std::sqrt(sqrSumAll);
        }

        real dot(const ArrayDOFV<vsize> &R)
        {
            DNDS_assert(base::dist);
            real sqrSum{0}, sqrSumAll;
            forEachInArray(*base::dist, [&](TElem &e, index i)
                           { sqrSum += e.p().dot((*R.dist)[i].p()); });
            MPI_Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, base::dist->getMPI().comm);
            return sqrSumAll;
        }

        // const Eigen::Map<Eigen::Vector<real, vsize>> &operator[](index i)
        // {
        //     return base::operator[](i).p();
        // }

        Eigen::Map<Eigen::Vector<real, vsize>> operator[](index i)
        {
            return base::operator[](i).p();
        }

        template<int fixedVsize>
        Eigen::Map<Eigen::Vector<real, fixedVsize>> get(index i)
        {
            return base::operator[](i).template p_fixed<fixedVsize>();
        }
    };
}

namespace DNDS
{
    class ArrayRecV : public ArrayLocal<VarVector>
    {
        index _m_j;

    public:
        typedef ArrayLocal<VarVector> base;
        using ArrayLocal<VarVector>::ArrayLocal;
        ArrayRecV() {}

        template <class TFSize_I>
        ArrayRecV(index distSize, const MPIInfo &mpi, TFSize_I &&FSize_I, index msize_j)
            : _m_j(msize_j)
        {
            base::dist = std::make_shared<Array<VarVector>>(
                typename VarVector::Context([&](index i)
                                            { return msize_j * FSize_I(i); },
                                            distSize),
                mpi);
            forEachInArray(*base::dist, [&](VarVector &e, index i)
                           { e.p().setZero(); });
        }

        template <class TFSize_I>
        void resize(index nsize, const MPIInfo &mpi, TFSize_I &&FSize_I, index msize_j)
        {
            _m_j = msize_j;
            base::dist = std::make_shared<Array<VarVector>>(
                typename VarVector::Context([&](index i)
                                            { return msize_j * FSize_I(i); },
                                            nsize),
                mpi);
            forEachInArray(*base::dist, [&](VarVector &e, index i)
                           { e.p().setZero(); });
        }

        void setConstant(real v)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](VarVector &e, index i)
                           { e.p().setConstant(v); });
        }

        template <typename Tin>
        void setConstant(const Tin &in)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](VarVector &e, index i)
                           { e.p() = in; });
        }

        void operator=(const ArrayRecV &R)
        {
            forEachInArray(*base::dist, [&](VarVector &e, index i)
                           { e.p() = (*R.dist)[i].p(); });
        }

        void operator+=(const ArrayRecV &R)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](VarVector &e, index i)
                           { e.p() += (*R.dist)[i].p(); });
        }

        void addTo(const ArrayRecV &R, real r)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](VarVector &e, index i)
                           { e.p() += (*R.dist)[i].p() * r; });
        }

        void operator-=(const ArrayRecV &R)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](VarVector &e, index i)
                           { e.p() -= (*R.dist)[i].p(); });
        }

        void operator*=(real r)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](VarVector &e, index i)
                           { e.p() *= r; });
        }

        template <class VR>
        void operator*=(const VR &R)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](VarVector &e, index i)
                           { e.p() *= R[i]; });
        }

        /**
         * @brief collective
         *
         */
        real norm2__TODO()
        {
            DNDS_assert(base::dist);
            real sqrSum{0}, sqrSumAll;
            forEachInArray(*base::dist, [&](VarVector &e, index i)
                           { sqrSum += e.p().squaredNorm(); });
            MPI_Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, base::dist->getMPI().comm);
            // std::cout << "norm2is " << std::scientific << sqrSumAll << std::endl;
            return std::sqrt(sqrSumAll);
        }

        real dot__TODO(const ArrayRecV &R)
        {
            DNDS_assert(base::dist);
            real sqrSum{0}, sqrSumAll;
            forEachInArray(*base::dist, [&](VarVector &e, index i)
                           { sqrSum += e.p().dot((*R.dist)[i].p()); });
            MPI_Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, base::dist->getMPI().comm);
            return sqrSumAll;
        }

        // const Eigen::Map<Eigen::Vector<real, vsize>> &operator[](index i)
        // {
        //     return base::operator[](i).p();
        // }

        Eigen::Map<Eigen::Matrix<real, -1, -1>> operator[](index i)
        {
            return base::operator[](i).m_by_ncol(_m_j);
        }
    };
}

namespace DNDS
{
    template <uint32_t vsize>
    class ArrayVDOF : public ArrayLocal<SemiVarMatrix<vsize>> // TODO: update to sepearated arrays
    {
        typedef ArrayLocal<SemiVarMatrix<vsize>> base;
        using ArrayLocal<SemiVarMatrix<vsize>>::ArrayLocal;
        ArrayVDOF() {}

        void setConstant(real v)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](SemiVarMatrix<vsize> &e, index i)
                           { e.m().setConstant(v); });
        }

        void operator=(const ArrayVDOF<vsize> &R)
        {
            forEachInArray(*base::dist, [&](SemiVarMatrix<vsize> &e, index i)
                           { e.m() = (*R.dist)[i].m(); });
        }

        void operator+=(const ArrayVDOF<vsize> &R)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](SemiVarMatrix<vsize> &e, index i)
                           { e.m() += (*R.dist)[i].m(); });
        }

        void addTo(const ArrayVDOF<vsize> &R, real r)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](SemiVarMatrix<vsize> &e, index i)
                           { e.m() += (*R.dist)[i].m() * r; });
        }

        void operator-=(const ArrayVDOF<vsize> &R)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](SemiVarMatrix<vsize> &e, index i)
                           { e.m() -= (*R.dist)[i].m(); });
        }

        void operator*=(real r)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](SemiVarMatrix<vsize> &e, index i)
                           { e.m() *= r; });
        }

        template <class VR>
        void operator*=(const VR &R)
        {
            DNDS_assert(base::dist);
            forEachInArray(*base::dist, [&](SemiVarMatrix<vsize> &e, index i)
                           { e.m() *= R[i]; });
        }

        /**
         * @brief collective
         *
         */
        real norm2()
        {
            DNDS_assert(base::dist);
            real sqrSum{0}, sqrSumAll;
            forEachInArray(*base::dist, [&](SemiVarMatrix<vsize> &e, index i)
                           { sqrSum += e.m().squaredNorm(); });
            MPI_Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, base::dist->getMPI().comm);
            // std::cout << "norm2is " << std::scientific << sqrSumAll << std::endl;
            return std::sqrt(sqrSumAll);
        }

        real dot(const ArrayVDOF<vsize> &R)
        {
            DNDS_assert(base::dist);
            real sqrSum{0}, sqrSumAll;
            forEachInArray(*base::dist, [&](SemiVarMatrix<vsize> &e, index i)
                           { sqrSum += (e.m().array() * (*R.dist)[i].m().array()).sum(); });
            MPI_Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, base::dist->getMPI().comm);
            return sqrSumAll;
        }

        // const Eigen::Map<Eigen::Vector<real, vsize>> &operator[](index i)
        // {
        //     return base::operator[](i).p();
        // }

        Eigen::Map<Eigen::Matrix<real, -1, -1>> operator[](index i)
        {
            return base::operator[](i).m();
        }
    };
}