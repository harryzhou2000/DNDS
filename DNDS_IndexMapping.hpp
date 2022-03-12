#pragma once

#include <unordered_map>
#include <algorithm>

#include "DNDS_Defines.h"
#include "DNDS_MPI.hpp"

namespace DNDS
{

    // mapping place from local main/ghost place to globalIndex or inverse
    // main data is a offset mapping while ghost indces are stored in ascending order
    // use 2-split search in ghost indexing
    // warning!! due to MPI restrictions data inside are 32-bit
    class OffsetAscendIndexMapping
    {
        typedef MPI_int mapIndex;
        typedef std::vector<mapIndex> tMapIndexVec;
        index mainOffset;
        index mainSize;
        tIndexVec ghostIndex;
        tMapIndexVec ghostStart;

    public:
        OffsetAscendIndexMapping(index nmainOffset, index nmainSize) : mainOffset(nmainOffset), mainSize(nmainSize) { ghostIndex.resize(0); }

        auto &ghost() { return ghostIndex; }

        auto &gStarts() { return ghostStart; }

        void sort() { std::sort(ghostIndex.begin(), ghostIndex.end()); };

        index &ghostAt(MPI_int rank, index ighost)
        {
            assert(ighost >= 0 && ighost < (ghostStart[rank + 1] - ghostStart[rank]));
            return ghostIndex[ghostStart[rank] + ighost];
        }

        // TtMapIndexVec is a std::vector of someint
        // could overflow, accumulated to 32-bit
        template <class TtMapIndexVec>
        void allocateGhostIndex(const TtMapIndexVec &ghostSizes)
        {
            ghostStart.resize(ghostSizes.size() + 1);
            ghostStart[0] = 0;
            for (index i = 0; i < ghostSizes.size(); i++)
                ghostStart[i + 1] = ghostStart[i] + ghostSizes[i];
            ghostIndex.resize(ghostStart[ghostSizes.size()]);
        }

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
            auto place = std::lower_bound(ghostIndex.begin() + ghostStart[rank], ghostIndex.begin() + ghostStart[rank + 1], globalQuery);
            if (*place == globalQuery)
            {
                val = place - (ghostIndex.begin() + ghostStart[rank]);
                return true;
            }
            return false;
        }

        // returns rank and place, rank==-1 means main data
        bool search(index globalQuery, MPI_int &rank, index &val) const
        {
            if (searchInMain(globalQuery, val))
            {
                rank = -1;
                return true;
            }
            for (rank = 0; rank < (ghostStart.size() - 1); rank++)
                if (searchInGhost(globalQuery, rank, val))
                    return true;
            return false;
        }

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

    // mapping from rank-main place to global indices
    // should be global-identical, can broadcast

    class GlobalOffsetsMapping
    {
        tIndexVec RankLengths;
        tIndexVec RankOffsets;

    public:
        tIndexVec &RLengths() { return RankLengths; }

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

        index operator()(MPI_int rank, index val)
        {

            assert((rank >= 0 && rank <= RankLengths.size()) &&
                   (val >= 0 && val <= RankOffsets[rank + 1] - RankOffsets[rank]));
            return RankOffsets[rank] + val;
        }

        bool search(index globalQuery, MPI_int &rank, index &val) const
        {
            auto place = std::lower_bound(RankOffsets.begin(), RankOffsets.end(), globalQuery, std::less_equal<index>());
            rank = place - 1  - RankOffsets.begin();
            if (rank < RankLengths.size())
            {
                val = globalQuery - RankOffsets[rank];
                return true;
            }
            return false;
        }
    };

}