#pragma once

#include "DNDS_Defines.h"
#include "DNDS_Elements.hpp"
#include <map>
#include <fstream>
#include <regex>
#include <sstream>

namespace DNDS
{
    struct GmshElem
    {
        Elem::ElemType elemType;
        int phyGrp;
        std::vector<index> indexList;
    };

    struct GmshPhyGrp
    {
        uint8_t dim;
        std::string name;
    };

    struct SerialGmshReader2d
    {
        std::map<index, Elem::tPoint> readPoints;
        std::map<index, GmshElem> readElems;
        std::map<index, GmshPhyGrp> readPhyGrps;

        void FileRead(const std::string &fname)
        {
            readPoints.clear(), readElems.clear();
            std::ifstream fin(fname);
            std::regex regComments("Comments");
            std::regex regNodes("Nodes");
            std::regex regElems("Elements");
            std::regex regPhyGrps("PhysicalNames");
            std::regex regEnd("End");
            std::regex regPhyContent("(\\d+)\\s+(\\d+)\\s+\\\"(.*)\\\"");

            while (!fin.eof())
            {
                std::string line;
                std::getline(fin, line);
                if (line[0] == '$')
                    if (std::regex_search(line.c_str(), regComments))
                    {
                        while (!fin.eof())
                        {
                            std::getline(fin, line);
                            if (line[0] == '$' && std::regex_search(line, regEnd))
                                break;
                        }
                        if (fin.eof())
                        {
                            log() << "FileRead Error, comment not concluded\n"
                                  << std::endl;
                            assert(false);
                        }
                    }
                    else if (std::regex_search(line, regNodes))
                    {
                        index nnodes;
                        std::getline(fin, line);
                        std::stringstream lineStream;
                        lineStream << line;
                        lineStream >> nnodes;
                        log() << "Node Number = [" << nnodes << "]" << std::endl;

                        while (!fin.eof())
                        {
                            std::getline(fin, line);
                            if (line[0] == '$' && std::regex_search(line, regEnd))
                                break;
                            std::stringstream lineStream;
                            lineStream << line;
                            index inode;
                            Elem::tPoint p;
                            lineStream >> inode >> p[0] >> p[1] >> p[2];
                            readPoints[inode] = p;
                        }
                        if (fin.eof())
                        {
                            log() << "FileRead Error, nodes not concluded\n"
                                  << std::endl;
                            assert(false);
                        }
                    }
                    else if (std::regex_search(line, regElems))
                    {
                        index nelems;
                        std::getline(fin, line);
                        std::stringstream lineStream;
                        lineStream << line;
                        lineStream >> nelems;
                        log() << "Elem Number = [" << nelems << "]" << std::endl;

                        while (!fin.eof())
                        {
                            std::getline(fin, line);
                            if (line[0] == '$' && std::regex_search(line, regEnd))
                                break;
                            std::stringstream lineStream;
                            lineStream << line;
                            index ielem, gmshType, ntag, iphy, iegrp;
                            lineStream >> ielem >> gmshType >> ntag >> iphy >> iegrp;
                            assert(ntag == 2);                              // warning: why???
                            Elem::ElemType type = Elem::ElemType(gmshType); // direct mapping of type
                            Elem::ElementManager e(type, 0);
                            int nnode = e.getNNode();

                            readElems[ielem].elemType = type;
                            readElems[ielem].indexList.resize(nnode);
                            readElems[ielem].phyGrp = iphy;
                            for (int in = 0; in < nnode; in++)
                                lineStream >> readElems[ielem].indexList[in];
                        }
                        if (fin.eof())
                        {
                            log() << "FileRead Error, elems not concluded\n"
                                  << std::endl;
                            assert(false);
                        }
                    }
                    else if (std::regex_search(line, regPhyGrps))
                    {
                        index nphys;
                        std::getline(fin, line);
                        std::stringstream lineStream;
                        lineStream << line;
                        lineStream >> nphys;
                        log() << "PhyGrp Number = [" << nphys << "]" << std::endl;

                        while (!fin.eof())
                        {
                            std::getline(fin, line);
                            if (line[0] == '$' && std::regex_search(line, regEnd))
                                break;
                            std::cmatch m;
                            auto ret = std::regex_search(line.c_str(), m, regPhyContent);
                            assert(ret);

                            index dim, iphy;
                            std::string name;
                            std::stringstream con;
                            con << m[1].str() + " " + m[2].str();
                            con >> dim >> iphy;
                            name = m[3].str();

                            readPhyGrps[iphy].dim = dim;
                            readPhyGrps[iphy].name = name;
                        }
                        if (fin.eof())
                        {
                            log() << "FileRead Error, phygrp not concluded\n"
                                  << std::endl;
                            assert(false);
                        }
                    }
            }

            fin.close();
        }
    };

}