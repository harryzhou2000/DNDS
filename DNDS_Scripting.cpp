#include"DNDS_Scripting.hpp"

namespace DNDS
{
    namespace JSON
    {
        void ParamParser::Parse(const rapidjson::Value::Object &cObj, int iden)
        {
            for (auto &i : list)
            {
                auto type = std::get<0>(i);
                auto ptr = std::get<1>(i);
                auto &name = std::get<2>(i);
                auto &post = std::get<3>(i);
                switch (type)
                {
                case ItemType::Object:
                {
                    ParamParser *pParser = (ParamParser *)ptr;
                    if (!cObj.HasMember(name.c_str()) ||
                        !(cObj[name.c_str()].IsObject()))
                    {
                        log() << "JSON: ";
                        for (int den = 0; den < iden; den++)
                            log() << "    ";
                        log() << name << std::endl;
                        log() << "=== !! Failed !! not object ===" << std::endl;
                        assert(false);
                    }

                    if (mpi.rank == 0)
                    {
                        log() << "JSON: ";
                        for (int den = 0; den < iden; den++)
                            log() << "    ";
                        log() << name << "." << std::endl;
                    }

                    pParser->Parse(cObj[name.c_str()].GetObject(), iden + 1);
                }
                break;
                case ItemType::Int:
                {
                    int *dest = (int *)ptr;
                    if (!cObj.HasMember(name.c_str()) ||
                        !(cObj[name.c_str()].IsInt()))
                    {
                        log() << "JSON: ";
                        for (int den = 0; den < iden; den++)
                            log() << "    ";
                        log() << name << std::endl;
                        log() << "=== !! Failed !! not int ===" << std::endl;
                        assert(false);
                    }
                    *dest = cObj[name.c_str()].GetInt();
                    if (mpi.rank == 0)
                    {
                        log() << "JSON: ";
                        for (int den = 0; den < iden; den++)
                            log() << "    ";
                        log() << name << " = " << *dest << std::endl;
                    }
                }
                break;
                case ItemType::DNDS_Real:
                {
                    real *dest = (real *)ptr;
                    if (!cObj.HasMember(name.c_str()) ||
                        !(cObj[name.c_str()].IsNumber()))
                    {
                        log() << "JSON: ";
                        for (int den = 0; den < iden; den++)
                            log() << "    ";
                        log() << name << std::endl;
                        log() << "=== !! Failed !! not real ===" << std::endl;
                        assert(false);
                    }
                    *dest = cObj[name.c_str()].GetDouble();
                    if (mpi.rank == 0)
                    {
                        log() << "JSON: ";
                        for (int den = 0; den < iden; den++)
                            log() << "    ";
                        log() << name << " = " << *dest << std::endl;
                    }
                }
                break;
                case ItemType::Bool:
                {
                    bool *dest = (bool *)ptr;
                    if (!cObj.HasMember(name.c_str()) ||
                        !(cObj[name.c_str()].IsBool()))
                    {
                        log() << "JSON: ";
                        for (int den = 0; den < iden; den++)
                            log() << "    ";
                        log() << name << std::endl;
                        log() << "=== !! Failed !! not bool ===" << std::endl;
                        assert(false);
                    }
                    *dest = cObj[name.c_str()].GetBool();
                    if (mpi.rank == 0)
                    {
                        log() << "JSON: ";
                        for (int den = 0; den < iden; den++)
                            log() << "    ";
                        log() << name << " = " << ((*dest) ? "true" : "false") << std::endl;
                    }
                }
                break;
                case ItemType::Eigen_RealVec:
                {
                    Eigen::VectorXd *dest = (Eigen::VectorXd *)ptr;
                    if (!cObj.HasMember(name.c_str()) ||
                        !(cObj[name.c_str()].IsArray()))
                    {
                        log() << "JSON: ";
                        for (int den = 0; den < iden; den++)
                            log() << "    ";
                        log() << name << std::endl;
                        log() << "=== !! Failed !! not array ===" << std::endl;
                        assert(false);
                    }
                    const auto &arr = cObj[name.c_str()].GetArray();
                    dest->resize(arr.Size());
                    for (int ii = 0; ii < arr.Size(); ii++)
                    {
                        assert(arr[ii].IsNumber());
                        dest->operator[](ii) = arr[ii].GetDouble();
                    }

                    // *dest = cObj[name.c_str()].GetDouble();
                    if (mpi.rank == 0)
                    {
                        log() << "JSON: ";
                        for (int den = 0; den < iden; den++)
                            log() << "    ";
                        log() << name << " = [" << dest->transpose() << "] " << std::endl;
                    }
                }
                break;

                case ItemType::std_String:
                {
                    std::string *dest = (std::string *)ptr;
                    if (!cObj.HasMember(name.c_str()) ||
                        !(cObj[name.c_str()].IsString()))
                    {
                        log() << "JSON: ";
                        for (int den = 0; den < iden; den++)
                            log() << "    ";
                        log() << name << std::endl;
                        log() << "=== !! Failed !! not string ===" << std::endl;
                        assert(false);
                    }
                    *dest = cObj[name.c_str()].GetString();
                    if (mpi.rank == 0)
                    {
                        log() << "JSON: ";
                        for (int den = 0; den < iden; den++)
                            log() << "    ";
                        log() << name << " = \"" << *dest << "\" " << std::endl;
                    }
                }
                break;

                default:
                    assert(false);
                    break;
                }
                post();
            }
        }
    }
}