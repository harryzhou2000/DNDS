#pragma once
#include "DNDS_Defines.h"
#include "rapidjson/filereadstream.h"
// #include "rapidjson/istreamwrapper.h"
#include "rapidjson/document.h"
#include <cstdio>

namespace DNDS
{
    namespace JSON
    {
        void ReadFile(const std::string &fname, rapidjson::Document &d)
        {
            using namespace rapidjson;
#if defined(__WINDOWS_) || defined(_WIN32)
            FILE *fp = fopen(fname.c_str(), "rb");
#else
            FILE *fp = fopen(fname.c_str(), "r"); // non-Windows use "r"
#endif
            char readBuffer[65536];
            FileReadStream is(fp, readBuffer, sizeof(readBuffer));
            d.ParseStream(is);
            fclose(fp);
        }


    }


    namespace Python3
    {
        
    }

}