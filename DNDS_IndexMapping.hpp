#pragma once

#include <unordered_map>
#include <algorithm>

#include "DNDS_Defines.h"
#include "DNDS_MPI.hpp"

namespace DNDS
{ // mapping from rank-main place to global indices
    // should be global-identical, can broadcast

    class GlobalOffsetsMapping
    {
        tIndexVec RankLengths;
        tIndexVec RankOffsets;

    public:
        tIndexVec &RLengths() { return RankLengths; }

        /// \brief bcast, synchronize the rank lengths, then accumulate rank offsets
        void setMPIAlignBcast(const MPIInfo &mpi, index myLength)
        {
            RankLengths.resize(mpi.size);
            RankOffsets.resize(mpi.size + 1);
            RankLengths[mpi.rank] = myLength;

            // tMPI_reqVec bcastReqs(mpi.size); // for Ibcast

            for (MPI_int r = 0; r < mpi.size; r++)
            {
                // std::cout << mpi.rank << '\t' << myLength << std::endl;
                MPI_Bcast(RankLengths.data() + r, sizeof(index), MPI_BYTE, r, mpi.comm);
            }

            RankOffsets[0] = 0;
            for (auto i = 0; i < RankLengths.size(); i++)
                RankOffsets[i + 1] = RankOffsets[i] + RankLengths[i];
        }

        ///\brief inputs local index, outputs global
        index operator()(MPI_int rank, index val)
        {

            assert((rank >= 0 && rank <= RankLengths.size()) &&
                   (val >= 0 && val <= RankOffsets[rank + 1] - RankOffsets[rank]));
            return RankOffsets[rank] + val;
        }

        ///\brief inputs global index, outputs rank and local index, false if out of bound
        bool search(index globalQuery, MPI_int &rank, index &val) const
        {
            // find the first larger than query, for this is a [,) interval search, should find the ) position
            // example: RankOffsets = {0,1,3,3,4,4,5,5}
            //                 Rank = {0,1,2,3,4,5,6}
            // query 5 should be rank 7, which is out-of bound, returns false
            // query 4 should be rank 5, query 3 should be rank 3, query 2 should be rank 1
            auto place = std::lower_bound(RankOffsets.begin(), RankOffsets.end(), globalQuery, std::less_equal<index>());
            rank = place - 1 - RankOffsets.begin();
            if (rank < RankLengths.size() && rank >= 0)
            {
                val = globalQuery - RankOffsets[rank];
                return true;
            }
            return false;
        }
    };

    // mapping place from local main/ghost place to globalIndex or inverse
    // main data is a offset mapping while ghost indces are stored in ascending order
    // use 2-split search in ghost indexing, so the global indexing to local indexing must be ascending
    // warning!! due to MPI restrictions data inside are 32-bit
    class OffsetAscendIndexMapping
    {
        typedef MPI_int mapIndex;
        typedef std::vector<mapIndex> tMapIndexVec;
        index mainOffset;
        index mainSize;

    public:
        /// \brief ho many elements/global indexes at ghost for each other process
        /// ___ aka: pullIndexSizes
        tMPI_intVec ghostSizes;

        /// \brief the global indices for each element
        /// ___ aka: pullingIndexGlobal
        tIndexVec ghostIndex;

        /// \brief starting location for each (other) process's ghost space in ghostIndex
        /// ___ aka: pullIndexStarts
        tMapIndexVec ghostStart;

        tMPI_intVec pushIndexSizes;
        tMPI_intVec pushIndexStarts;
        tIndexVec pushingIndexGlobal;

        /// \brief the local indices for each element,
        /// the local ghost version of ghostIndex but does not exclude the local main part (which does not need communication)
        /// pulling request local also remains the input order from pullingIndexGlobal, or to say 1-on-1
        /// if stored value i is >=0, then means indexer[i], if stored value  i is <0, then means ghostIndexer[-i-1]
        tIndexVec pullingRequestLocal;

        template <class TpullSet>
        OffsetAscendIndexMapping(index nmainOffset, index nmainSize,
                                 TpullSet &&pullingIndexGlobal,
                                 const GlobalOffsetsMapping &LGlobalMapping,
                                 const MPIInfo &mpi)
            : mainOffset(nmainOffset),
              mainSize(nmainSize)
        {
            ghostSizes.assign(mpi.size, 0);
            for (auto i : pullingIndexGlobal)
            {
                MPI_int rank;
                index loc; // dummy here
                LGlobalMapping.search(i, rank, loc);
                // if (rank != mpi.rank)
                ghostSizes[rank]++;
            }

            ghostStart.resize(ghostSizes.size() + 1);
            ghostStart[0] = 0;
            for (index i = 0; i < ghostSizes.size(); i++)
                ghostStart[i + 1] = ghostStart[i] + ghostSizes[i];
            ghostIndex.reserve(ghostStart[ghostSizes.size()]);
            for (auto i : pullingIndexGlobal)
            {
                MPI_int rank;
                index loc; // dummy here
                LGlobalMapping.search(i, rank, loc);
                // if (rank != mpi.rank)
                ghostIndex.push_back(i);
            }
            ghostIndex.shrink_to_fit(); // only for safety
            sort();

            // obtain pushIndexSizes and actual push indices
            pushIndexSizes.resize(mpi.size);
            MPI_Alltoall(ghostSizes.data(), 1, MPI_INT, pushIndexSizes.data(), 1, MPI_INT, mpi.comm);
            AccumulateRowSize(pushIndexSizes, pushIndexStarts);
            pushingIndexGlobal.resize(pushIndexStarts[pushIndexStarts.size() - 1]);
            MPI_Alltoallv(ghostIndex.data(), ghostSizes.data(), ghostStart.data(), DNDS_MPI_INDEX,
                          pushingIndexGlobal.data(), pushIndexSizes.data(), pushIndexStarts.data(), DNDS_MPI_INDEX,
                          mpi.comm);

            //stores pullingRequest
            pullingRequestLocal = std::forward<TpullSet>(pullingIndexGlobal);
            for (auto &i : pullingRequestLocal)
            {
                MPI_int rank;
                index loc; // dummy here
                search(i, rank, loc);
                if (rank != -1)
                    i = -(1 + ghostStart[rank] + loc);
            }
        }

        // auto &ghost() { return ghostIndex; }

        // auto &gStarts() { return ghostStart; }

        // warning: using globally sorted condition
        void sort() { std::sort(ghostIndex.begin(), ghostIndex.end()); };

        index &ghostAt(MPI_int rank, index ighost)
        {
            assert(ighost >= 0 && ighost < (ghostStart[rank + 1] - ghostStart[rank]));
            return ghostIndex[ghostStart[rank] + ighost];
        }

        // TtMapIndexVec is a std::vector of someint
        // could overflow, accumulated to 32-bit
        // template <class TtMapIndexVec>
        // void allocateGhostIndex(const TtMapIndexVec &ghostSizes)
        // {
        // }

        bool searchInMain(index globalQuery, index &val) const
        {
            // std::cout << mainOffset << mainSize << std::endl;
            if (globalQuery >= mainOffset && globalQuery < mainSize + mainOffset)
            {
                val = globalQuery - mainOffset;
                return true;
            }
            return false;
        }

        bool searchInGhost(index globalQuery, MPI_int rank, index &val) const
        {
            assert((rank >= 0 && rank < ghostStart.size() - 1));
            auto place = std::lower_bound(
                ghostIndex.begin() + ghostStart[rank],
                ghostIndex.begin() + ghostStart[rank + 1],
                globalQuery);
            if (*place == globalQuery)
            {
                val = place - (ghostIndex.begin() + ghostStart[rank]);
                return true;
            }
            return false;
        }

        /// \brief returns rank and place in ghost of rank, rank==-1 means main data
        bool search(index globalQuery, MPI_int &rank, index &val) const
        {
            if (searchInMain(globalQuery, val))
            {
                rank = -1;
                return true;
            }
            // warning! linear on num of ranks here
            for (rank = 0; rank < (ghostStart.size() - 1); rank++)
                if (searchInGhost(globalQuery, rank, val))
                    return true;
            return false;
        }

        /// \brief if rank == -1, return the global index of local main data,
        /// or else return the ghosted index of local ghost data.
        index operator()(MPI_int rank, index val)
        {
            if (rank == -1)
            {
                assert(val >= 0 && val < mainSize);
                return val + mainOffset;
            }
            else
            {
                assert(
                    (rank >= 0 && rank < ghostStart.size() - 1) &&
                    (val >= 0 && val < ghostStart[rank + 1] - ghostStart[rank]));
                return ghostIndex[ghostStart[rank] + val];
            }
        }
    };

}