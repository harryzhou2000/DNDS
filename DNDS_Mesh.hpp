#pragma once

#include "DNDS_Defines.h"
#include "DNDS_Elements.hpp"
#include <map>
#include <algorithm>
#include <fstream>
#include <regex>
#include <sstream>
#include <iomanip>

namespace DNDS
{
    struct GmshElem
    {
        int phyGrp;
        std::vector<index> nodeList;
        Elem::ElemType elemType;
        inline bool operator==(const GmshElem &R) const
        {
            if (R.elemType != elemType)
                return false;
            Elem::ElementManager e(elemType, 0);
            std::vector<index> n1(nodeList.begin(), nodeList.begin() + e.getNVert());
            std::vector<index> n2(R.nodeList.begin(), R.nodeList.begin() + e.getNVert());
            std::sort(n1.begin(), n1.end()), std::sort(n2.begin(), n2.end());
            for (int i = 0; i < e.getNVert(); i++)
                if (n1[i] != n2[i])
                    return false;
            return true;
        }
    };

    struct GmshPhyGrp
    {
        uint8_t dim;
        std::string name;
    };

    struct SerialGmshReader2d
    {
        // std::map<index, Elem::tPoint> readPoints;
        // std::map<index, GmshElem> readElems;
        std::map<index, GmshPhyGrp> readPhyGrps;

        std::vector<Elem::tPoint> readPoints;
        std::vector<GmshElem> readElems;
        // std::vector<Elem::tPoint> readPhyGrps;

        std::vector<GmshElem> volElems;
        std::vector<std::vector<index>> vol2face;

        std::vector<GmshElem> bndElems;

        std::vector<GmshElem> faceElems;
        std::vector<std::pair<index, index>> face2vol;

        inline void FileRead(const std::string &fname)
        {
            readPoints.clear(), readElems.clear();
            std::ifstream fin(fname);
            if (!fin)
            {
                log() << "Error: FileRead open \"" << fname << "\" failure" << std::endl;
                assert(false);
            }
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
                        readPoints.resize(nnodes);

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
                            inode--; // to 0 based
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
                        readElems.resize(nelems);

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

                            ielem--; // use 0 based index
                            readElems[ielem].elemType = type;
                            readElems[ielem].nodeList.resize(nnode);
                            readElems[ielem].phyGrp = iphy;
                            for (int in = 0; in < nnode; in++)
                            {
                                lineStream >> readElems[ielem].nodeList[in];
                                readElems[ielem].nodeList[in]--; // use 0 based index
                            }
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

        inline void InterpolateTopology()
        {
            // count and divide
            index nVolElem = 0;
            index nBndElem = 0;
            for (auto &i : readElems)
            {
                if (readPhyGrps[i.phyGrp].dim == 2) // magical 2d for volume
                    nVolElem++;
                else if (readPhyGrps[i.phyGrp].dim == 1) // magical 2d for face
                    nBndElem++;
                else
                    assert(false);
            };
            log() << "GmshGrid: found volume elements: [" << nVolElem << "]\n";
            log() << "GmshGrid: found boundary elements: [" << nBndElem << "]\n";
            volElems.reserve(nVolElem), bndElems.reserve(nBndElem);
            for (auto &i : readElems)
            {
                if (readPhyGrps[i.phyGrp].dim == 2) // magical 2d for volume
                    volElems.push_back(i);
                else if (readPhyGrps[i.phyGrp].dim == 1) // magical 2d for face
                    bndElems.push_back(i);
                else
                    assert(false);
            };
            assert(volElems.size() == nVolElem && bndElems.size() == nBndElem);

            faceElems.reserve(nVolElem * 3);
            face2vol.reserve(nVolElem * 3);
            vol2face.resize(volElems.size());
            for (index i = 0; i < volElems.size(); i++)
                vol2face[i].resize(Elem::ElementManager(volElems[i].elemType, 0).getNFace());

            // establish vol2face
            std::vector<index> node2faceSiz(readPoints.size(), 0);
            for (auto &i : volElems)
            {
                auto e = Elem::ElementManager(i.elemType, 0);
                for (int iface = 0; iface < e.getNFace(); iface++)
                {
                    auto ef = e.ObtainFace(iface, 0);
                    std::vector<index> faceNodes(ef.getNNode());
                    e.SubstractFaceNodes(iface, ef, i.nodeList, faceNodes);
                    for (auto n : faceNodes)
                        node2faceSiz[n]++;
                }
            };
            std::vector<std::vector<index>> node2face(readPoints.size());
            for (index i = 0; i < readPoints.size(); i++)
                node2face[i].reserve(node2faceSiz[i]);
            for (index iv = 0; iv < volElems.size(); iv++)
            {
                auto &i = volElems[iv];
                auto e = Elem::ElementManager(i.elemType, 0);
                for (int iface = 0; iface < e.getNFace(); iface++)
                {
                    auto ef = e.ObtainFace(iface, 0);
                    std::vector<index> faceNodes(ef.getNNode());
                    e.SubstractFaceNodes(iface, ef, i.nodeList, faceNodes);
                    index checkNode = faceNodes[0];
                    GmshElem faceElem;
                    faceElem.elemType = ef.getType();
                    faceElem.phyGrp = -1;
                    faceElem.nodeList = faceNodes;
                    bool found = false;
                    for (auto f : node2face[checkNode])
                        if (faceElems[f] == faceElem)
                        {
                            assert(face2vol[f].second == -1);
                            face2vol[f].second = iv;
                            found = true;
                        }
                    if (!found)
                    {
                        faceElems.push_back(faceElem);
                        for (auto n : faceNodes)
                            node2face[n].push_back(faceElems.size() - 1);
                        face2vol.push_back(std::make_pair(iv, index(-1)));
                    }
                }
            }

            for (auto &i : bndElems)
            {
                index checkNode = i.nodeList[0];
                bool found = false;
                index ffound;
                for (auto f : node2face[checkNode])
                    if (faceElems[f] == i)
                    {
                        assert(face2vol[f].second == -1); // for must be a boundary
                        ffound = f;
                        found = true;
                    }
                assert(found);
                faceElems[ffound].phyGrp = i.phyGrp;
            }
            // std::vector<std::vector<index>> volAtNode;
        }

        inline void WriteMeshDebugTecASCII(const std::string &fname)
        {
            if (!Elem::ElementManager::NBufferInit)
                Elem::ElementManager::InitNBuffer();
            Elem::tIntScheme schemeTri = Elem::INT_SCHEME_TRI_7;
            Elem::tIntScheme schemeQuad = Elem::INT_SCHEME_QUAD_16;
            std::ofstream fout(fname);
            if (!fout)
            {
                log() << "Error: WriteMeshDebugTecASCII open \"" << fname << "\" failure" << std::endl;
                assert(false);
            }
            fout << "VARIABLES = \"x\", \"y\", \"volume\"\n" // 2d mesh so only x y
                 << "Zone N =" << readPoints.size() << ","
                 << " E = " << volElems.size() << ","
                 << "VARLOCATION=([1-2]=NODAL,[3]=CELLCENTERED)"
                 << "\n,"
                 << "DATAPACKING=BLOCK, ZONETYPE=FEQUADRILATERAL"
                 << "\n";
            fout << std::setprecision(16);
            for (auto &p : readPoints)
                fout << p[0] << "\n";
            for (auto &p : readPoints)
                fout << p[1] << "\n";
            for (auto &i : volElems)
            {
                Elem::ElementManager e(i.elemType, 0);
                switch (e.getPspace())
                {
                case Elem::ParamSpace::TriSpace:
                    e.setType(i.elemType, schemeTri);
                    break;
                case Elem::ParamSpace::QuadSpace:
                    e.setType(i.elemType, schemeQuad);
                    break;
                default:
                    assert(false);
                }
                real v = 0.0;
                Eigen::MatrixXd coords(3, e.getNNode());
                for (int in = 0; in < e.getNNode(); in++)
                    coords(Eigen::all, in) = readPoints[i.nodeList[in]];
                //, std::cout << i.nodeList[in] << " [" << readPoints[i.nodeList[in]].transpose << std::endl;

                e.Integration(v,
                              [&](double &vinc, int m, DNDS::Elem::tPoint &p, DNDS::Elem::tDiFj &DiNj) -> void
                              {
                                  vinc = DNDS::Elem::DiNj2Jacobi(DiNj, coords)({0, 1}, {0, 1}).determinant();
                                  if (vinc < 0)
                                      log() << "Error: 2d vol orientation wrong or distorted" << std::endl;
                                  assert(vinc > 0);
                              });
                fout << v << "\n";
            }
            for (auto &i : volElems)
            {
                Elem::ElementManager e(i.elemType, 0);
                switch (e.getPspace())
                {
                case Elem::ParamSpace::TriSpace:
                    fout << i.nodeList[0] + 1 << " " << i.nodeList[1] + 1 << " " << i.nodeList[2] + 1 << " " << i.nodeList[2] + 1 << '\n';
                    break;
                case Elem::ParamSpace::QuadSpace:
                    fout << i.nodeList[0] + 1 << " " << i.nodeList[1] + 1 << " " << i.nodeList[2] + 1 << " " << i.nodeList[3] + 1 << '\n';
                    break;
                default:
                    assert(false);
                }
            }
            fout.close();
        }
    };

}