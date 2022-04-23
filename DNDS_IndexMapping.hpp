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
        index operator()(MPI_int rank, index val) const
        {
            // if (!((rank >= 0 && rank <= RankLengths.size()) &&
            //       (val >= 0 && val <= RankOffsets[rank + 1] - RankOffsets[rank])))
            // {
            //     PrintVec(RankOffsets, std::cout);
            //     std::cout << rank << " KK " << val << std::endl;
            // }
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
            if (RankOffsets.size() == 0) // in case the communicator is of size 0 ??
                return false;
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
                // if (rank != mpi.rank) // must not exclude local ones for the sake of scatter/gather
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

            // stores pullingRequest // ! now cancelled
            // pullingRequestLocal = std::forward<TpullSet>(pullingIndexGlobal);
            // for (auto &i : pullingRequestLocal)
            // {
            //     MPI_int rank;
            //     index loc; // dummy here
            //     search(i, rank, loc);
            //     if (rank != -1)
            //         i = -(1 + ghostStart[rank] + loc);
            // }
        }

        template <class TpushSet, class TpushStart>
        OffsetAscendIndexMapping(index nmainOffset, index nmainSize,
                                 TpushSet &&pushingIndexes, // which stores local index
                                 TpushStart &&pushingStarts,
                                 const GlobalOffsetsMapping &LGlobalMapping,
                                 const MPIInfo &mpi)
            : mainOffset(nmainOffset),
              mainSize(nmainSize)
        {
            assert(pushingStarts.size() == mpi.size + 1 && pushingIndexes.size() == pushingStarts[mpi.size]);
            pushIndexSizes.resize(mpi.size);
            pushIndexStarts.resize(mpi.size + 1, 0);
            for (index i = 0; i < mpi.size; i++)
                pushIndexSizes[i] = pushingStarts[i + 1] - pushingStarts[i],
                pushIndexStarts[i + 1] = pushingStarts[i + 1];
            pushingIndexGlobal.resize(pushingIndexes.size());
            std::forward<TpushStart>(pushingStarts); //! might delete
            for (index i = 0; i < pushingIndexGlobal.size(); i++)
                pushingIndexGlobal[i] = LGlobalMapping(mpi.rank, pushingIndexes[i]); // convert from local to global
            std::forward<TpushSet>(pushingIndexes);                                  //! might delete

            ghostSizes.assign(mpi.size, 0);
            MPI_Alltoall(pushIndexSizes.data(), 1, MPI_INT, ghostSizes.data(), 1, MPI_INT, mpi.comm); // inverse to the normal pulling
            ghostStart.resize(ghostSizes.size() + 1);
            ghostStart[0] = 0;
            for (index i = 0; i < ghostSizes.size(); i++)
                ghostStart[i + 1] = ghostStart[i] + ghostSizes[i];
            ghostIndex.resize(ghostStart[ghostSizes.size()]);
            MPI_Alltoallv(pushingIndexGlobal.data(), pushIndexSizes.data(), pushIndexStarts.data(), DNDS_MPI_INDEX,
                          ghostIndex.data(), ghostSizes.data(), ghostStart.data(), DNDS_MPI_INDEX,
                          mpi.comm); // inverse to the normal pulling

            // !doesn't store pullingRequest
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

        // returns place relative to ghostStart[rank]
        bool searchInGhost(index globalQuery, MPI_int rank, index &val) const
        {
            assert((rank >= 0 && rank < ghostStart.size() - 1));
            if ((ghostStart[rank + 1] - ghostStart[rank]) == 0)
                return false; // size = 0 could result in seg error doing search
            auto start = ghostIndex.begin() + ghostStart[rank];
            auto end = ghostIndex.begin() + ghostStart[rank + 1];
            auto place = std::lower_bound(start, end, globalQuery);
            if (place != end && *place == globalQuery) // dereferencing end could result in seg error
            {
                val = place - (ghostIndex.begin() + ghostStart[rank]);
                return true;
            }
            return false;
        }

        /// \brief returns rank and place in ghost array, rank==-1 means main data
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
                {
                    val += ghostStart[rank];
                    return true;
                }
            return false;
        }

        /// \brief returns rank and place in ghost array, rank==-1 means main data
        /// returned val is used for pair indexing
        bool search_indexAppend(index globalQuery, MPI_int &rank, index &val) const
        {
            if (searchInMain(globalQuery, val))
            {
                rank = -1;
                return true;
            }
            // warning! linear on num of ranks here
            for (rank = 0; rank < (ghostStart.size() - 1); rank++)
                if (searchInGhost(globalQuery, rank, val))
                {
                    // std::cout << mainSize << std::endl;
                    val += ghostStart[rank] + mainSize;
                    return true;
                }
            return false;
        }

        /// \brief returns rank and place in ghost of rank, rank==-1 means main data
        /// \warning search returns index that applies to local ghost array, this only goes for the ith of rank
        bool search_indexRank(index globalQuery, MPI_int &rank, index &val) const
        {
            if (searchInMain(globalQuery, val))
            {
                rank = -1;
                return true;
            }
            // warning! linear on num of ranks here
            for (rank = 0; rank < (ghostStart.size() - 1); rank++)
                if (searchInGhost(globalQuery, rank, val))
                {
                    return true;
                }
            return false;
        }

        /// \brief if rank == -1, return the global index of local main data,
        /// or else return the ghosted global index of local ghost data.
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