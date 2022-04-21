#pragma once

#include "DNDS_Array.hpp"

namespace DNDS
{
    template <class T>
    class ArrayCascadePair
    {
    public:
        ArrayCascade<T> &arr;
        ArrayCascade<T> &arrGhost;

        template <class Tarr>
        ArrayCascadePair(Tarr &&nArr, Tarr &&nArrGhost) : arr(std::forward<Tarr>(nArr)), arrGhost(std::forward<Tarr>(nArrGhost)){};

        T operator[](index i)
        {
            assert(i >= 0 && i < size());
            if (i >= arr.size())
                return arrGhost[i - arr.size()];
            return arr[i];
        }

        index size() const
        {
            return arr.size() + arrGhost.size();
        }
    };
}

namespace DNDS
{
    /**
     * @brief f(T &element, index i)
     *
     */
    template <class T, class Tf>
    void forEachInArrayPair(ArrayCascadePair<T> &arr, Tf &&f)
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
    void forEachBasicInArrayPair(ArrayCascadePair<T> &arr, Tf &&f)
    {
        for (index i = 0; i < arr.size(); i++)
        {
            auto e = arr[i];
            for (index j = 0; j < e.size(); j++)
                f(e, i, e[j], j);
        }
    }

    template <class T>
    struct ArrayCascadeLocal
    {
        typedef ArrayCascade<T> tArray;
        typedef ArrayCascadePair<T> tPair;
        std::shared_ptr<ArrayCascade<T>> dist;
        std::shared_ptr<ArrayCascade<T>> ghost;
        std::shared_ptr<ArrayCascadePair<T>> pair;

        void Copy(ArrayCascadeLocal<T> &R)
        {
            dist = std::make_shared<ArrayCascade<T>>(R.dist);
            ghost = std::make_shared<ArrayCascade<T>>(dist.get());
            ghost->BorrowGGIndexing(*R.ghost);
            ghost->createMPITypes();
            MakePair();
        }

        template <class TR>
        void CreateGhostCopyComm(ArrayCascadeLocal<TR> &R)
        {
            assert(dist);
            ghost = std::make_shared<ArrayCascade<T>>(dist.get());
            ghost->BorrowGGIndexing(*R.ghost);
            ghost->createMPITypes();
            MakePair();
        }

        void MakePair()
        {
            pair = std::make_shared<ArrayCascadePair<T>>(*dist, *ghost);
        }

        void PullOnce()
        {
            ghost->pullOnce();
        }

        void PushOnce()
        {
            ghost->pushOnce();
        }

        void InitPersistentPullClean() { ghost->initPersistentPull(); }

        void StartPersistentPullClean() { ghost->startPersistentPull(); }

        void WaitPersistentPullClean() { ghost->waitPersistentPull(); }

        void ClearPersistentPullClean() { ghost->clearPersistentPull(); }

        // index the pair
        T operator[](index i)
        {
            assert(pair);
            return pair->operator[](i);
        }

        index size() const
        {
            return pair->size();
        }
    };

}