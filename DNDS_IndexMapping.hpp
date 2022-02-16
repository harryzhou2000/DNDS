#pragma once

#include <unordered_map>
#include <algorithm>

#include "DNDS_Defines.h"

namespace DNDS
{

    // mapping place from main/ghost place to globalIndex or inverse
    // main data is a offset mapping while ghost indces are stored in ascending order
    // use 2-split search in ghost indexing
    class OffsetAscendIndexMapping
    {
        index mainOffset;
        index mainSize;
        tIndexVec ghostIndex;
        tIndexVec ghostStart;

    public:
        OffsetAscendIndexMapping(index nmainOffset, index nmainSize) : mainOffset(nmainOffset), mainSize(nmainSize) { ghostIndex.resize(0); }

        tIndexVec &ghost()
        {
            return ghostIndex;
        }

        index &ghostAt(index rank, index ighost)
        {
            assert(ighost >= 0 && ighost < (ghostStart[rank + 1] - ghostStart[rank]));
            return ghostIndex[ghostStart[rank] + ighost];
        }

        void allocateGhostIndex(const tIndexVec &ghostSizes)
        {
            ghostStart.resize(ghostSizes.size() + 1);
            ghostStart[0] = 0;
            for (index i = 0; i < ghostSizes.size(); i++)
                ghostStart[i + 1] = ghostStart[i] + ghostSizes[i];
            ghostIndex.resize(ghostStart[ghostSizes.size()]);
        }

        bool searchInMain(index globalQuery, index &val) const
        {
            if (globalQuery >= mainOffset && globalQuery < mainSize)
            {
                val = globalQuery - mainOffset;
                return true;
            }
            return false;
        }

        bool searchInGhost(index globalQuery, index rank, index &val) const
        {
            auto place = std::lower_bound(ghostIndex.begin() + ghostStart[rank], ghostIndex.begin() + ghostStart[rank + 1], globalQuery);
            if (*place == globalQuery)
            {
                val = place - (ghostIndex.begin() + ghostStart[rank]);
                return true;
            }
            return false;
        }

        // returns rank and place, rank==-1 means main data
        bool search(index globalQuery, index &rank, index &val) const
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
    };

    // mapping from rank-main place to global indices
    // should be global-identical, can broadcast

    class GlobalOffsetsMapping
    {
        
    };

}