#include "DNDS_Elements_Prebuild.h"


namespace DNDS
{
    namespace Elem
    {
        bool ElementManager::NBufferInit = false;
        std::vector<std::vector<std::vector<tDiFj>>> ElementManager::NBuffer[DNDS_ELEM_TYPE_NUM]; // is this init safe?
    }
}