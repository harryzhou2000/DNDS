#pragma once
#include "DNDS_Defines.h"
#include "DNDS_MPI.hpp"
#include "Eigen/Dense"
#include "rapidjson/filereadstream.h"
// #include "rapidjson/istreamwrapper.h"
#include "rapidjson/document.h"
#include <cstdio>

namespace DNDS
{
    namespace JSON
    {
        inline void ReadFile(const std::string &fname, rapidjson::Document &d)
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

        class ParamParser // ! bad!
        {
            MPIInfo mpi;
            enum ItemType
            {
                Object = 0,
                Int = 1,
                DNDS_Real = 2,
                Bool = 3,
                std_String = 4,
                Eigen_RealVec = 5
            };
            // rapidjson::Document *doc = nullptr;
            // rapidjson::Value::Object obj;
            // std::vector<std::string> prefix;
            typedef std::tuple<ItemType, void *, std::string> listComponent;
            std::vector<listComponent> list;

        public:
            // ParamParser(rapidjson::Document *nDoc, const MPIInfo &nMpi)
            //     : mpi(nMpi), doc(nDoc) {}
            // ParamParser(rapidjson::Document *nDoc, const std::vector<std::string> &nPrefix, const MPIInfo &nMpi)
            //     : mpi(nMpi), doc(nDoc), prefix(nPrefix) {}

            // ParamParser(const rapidjson::Value::Object &Nobj, const MPIInfo &nMpi)
            //     : mpi(nMpi), obj(Nobj) {}

            ParamParser(const MPIInfo &nMpi)
                : mpi(nMpi) {}

            ~ParamParser()
            {
                // while (!list.empty())
                // {
                //     if (std::get<0>(list.back()) == ItemType::Object)
                //     {
                //         ParamParser *pParser = (ParamParser *)(std::get<1>(list.back()));
                //         delete pParser;
                //     }

                //     list.pop_back();
                // }
            }

            void AddObject(const std::string &name, ParamParser *pParser)
            {
                // std::vector<std::string> newPrefix = prefix;
                // newPrefix.push_back(name);

                // assert(obj[name.c_str()].IsObject());
                // ParamParser *pParser = new ParamParser(obj[name.c_str()].GetObject(), mpi);
                list.push_back(std::make_tuple(Object, (void *)(pParser), name));
            }

            void AddInt(const std::string &name, int *dest)
            {
                list.push_back(std::make_tuple(ItemType::Int, (void *)(dest), name));
            }

            void AddDNDS_Real(const std::string &name, real *dest)
            {
                list.push_back(std::make_tuple(ItemType::DNDS_Real, (void *)(dest), name));
            }

            void AddBool(const std::string &name, bool *dest)
            {
                list.push_back(std::make_tuple(ItemType::Bool, (void *)(dest), name));
            }

            void Addstd_String(const std::string &name, std::string *dest)
            {
                list.push_back(std::make_tuple(ItemType::std_String, (void *)(dest), name));
            }

            void AddEigen_RealVec(const std::string &name, Eigen::VectorXd *dest)
            {
                list.push_back(std::make_tuple(ItemType::Eigen_RealVec, (void *)(dest), name));
            }

            void Parse(const rapidjson::Value::Object &cObj, int iden);
            
        };

    }

    namespace Python3
    {

    }

}