#pragma once

#include "DNDS_Array.hpp"

namespace DNDS
{
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
            assert(arrGhost.father == &arr);
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
            assert(i >= 0 && i < size());
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
            assert(&R != this);
            dist = std::make_shared<Array<T>>(*R.dist);
            ghost = std::make_shared<Array<T>>(*R.ghost, dist.get());
            // ghost->BorrowGGIndexing(*R.ghost);
            // ghost->createMPITypes();
            MakePair();
        }

        template <class TR>
        void CreateGhostCopyComm(ArrayLocal<TR> &R)
        {
            assert(dist);
            ghost = std::make_shared<Array<T>>(dist.get());
            ghost->BorrowGGIndexing(*R.ghost);
            ghost->createMPITypes();
            MakePair();
        }

        void MakePair()
        {
            assert(dist && ghost);
            pair = std::make_shared<ArrayPair<T>>(*dist, *ghost);
        }

        void PullOnce()
        {
            assert(pair && dist && ghost);
            ghost->pullOnce();
        }

        void PushOnce()
        {
            assert(pair && dist && ghost);
            ghost->pushOnce();
        }

        void InitPersistentPullClean() { ghost->initPersistentPull(); }

        void StartPersistentPullClean() { ghost->startPersistentPull(); }

        void WaitPersistentPullClean() { ghost->waitPersistentPull(); }

        void ClearPersistentPullClean() { ghost->clearPersistentPull(); }

        // index the pair
        inline T &operator[](index i)
        {
            assert(pair && dist && ghost);
            // if (i >= 0)
            return pair->operator[](i);
            // else
            //     return pair->operator[](1);
        }

        index size() const
        {
            assert(pair && dist && ghost);
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
            assert(base::dist);
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
            assert(base::dist);
            forEachInArray(*base::dist, [&](VecStaticBatch<vsize> &e, index i)
                           { e.p().setConstant(v); });
        }

        template <typename Tin>
        void setConstant(const Tin &in)
        {
            assert(base::dist);
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
            assert(base::dist);
            forEachInArray(*base::dist, [&](VecStaticBatch<vsize> &e, index i)
                           { e.p() += (*R.dist)[i].p(); });
        }

        void operator*=(real r)
        {
            assert(base::dist);
            forEachInArray(*base::dist, [&](VecStaticBatch<vsize> &e, index i)
                           { e.p() *= r; });
        }

        template <class VR>
        void operator*=(const VR &R)
        {
            assert(base::dist);
            forEachInArray(*base::dist, [&](VecStaticBatch<vsize> &e, index i)
                           { e.p() *= R[i]; });
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