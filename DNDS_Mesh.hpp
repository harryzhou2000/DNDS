#pragma once

#include "DNDS_Defines.h"
#include "DNDS_Elements.hpp"
#include "DNDS_Array.hpp"
#include "DNDS_DerivedTypes.hpp"
#include "DNDS_DerivedArray.hpp"

#include <map>
#include <algorithm>
#include <fstream>
#include <regex>
#include <sstream>
#include <iomanip>
#include <metis.h>

namespace DNDS
{
    const index FACE_2_VOL_EMPTY = INT64_MIN;

    enum BoundaryType
    {
        Inner = -1,
        Unknown = 0,
        Wall = 1,
        Farfield = 2,

    };
}

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
        std::vector<std::vector<index>> vol2face; // could CSR

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
                            std::cout << line << std::endl;
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
                            assert(face2vol[f].second == FACE_2_VOL_EMPTY);
                            face2vol[f].second = iv;
                            found = true;
                            vol2face[iv][iface] = f;
                        }
                    if (!found)
                    {
                        faceElems.push_back(faceElem);
                        for (auto n : faceNodes)
                            node2face[n].push_back(faceElems.size() - 1);
                        vol2face[iv][iface] = faceElems.size() - 1;
                        face2vol.push_back(std::make_pair(iv, FACE_2_VOL_EMPTY));
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
                        assert(face2vol[f].second == FACE_2_VOL_EMPTY); // for must be a boundary
                        ffound = f;
                        found = true;
                    }
                assert(found);
                faceElems[ffound].phyGrp = i.phyGrp;
                // std::cout << faceElems[ffound].phyGrp << std::endl;
            }
            // std::vector<std::vector<index>> volAtNode;
            faceElems.shrink_to_fit();
            face2vol.shrink_to_fit();
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
            real vsum = 0.0;
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
                vsum += v;
            }
            log() << "Sum Volume [" << vsum << "]" << std::endl;
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

/******************************************************************************************************************************/
/**
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 * /
/******************************************************************************************************************************/

namespace DNDS
{

    /**
     * @brief convert arraySerialAdj's content to global (with partition J), then distribute it with partition I
     *
     */
    template <class TAdjBatch>
    void ConvertAdjSerial2Global(std::shared_ptr<ArrayCascade<TAdjBatch>> &arraySerialAdj,
                                 const std::vector<index> &partitionJSerial2Global,
                                 const MPIInfo &mpi)
    {
        IndexArray JSG(IndexArray::tContext(partitionJSerial2Global.size()), mpi);
        forEachInArray(
            JSG, [&](IndexArray::tComponent &e, index i)
            { e[0] = partitionJSerial2Global[i]; });
        IndexArray JSGGhost(&JSG);
        JSGGhost.createGlobalMapping();
        std::vector<index> ghostJSerialQuery;
        // InsertCheck(mpi);
        // forEachInArray(
        //     JSG, [&](IndexArray::tComponent &e, index i)
        //     { std::cout << e[0]; });
        // std::cout << std::endl;

        // get ghost
        index nGhost = 0;
        forEachBasicInArray(
            *arraySerialAdj,
            [&](TAdjBatch &e, index i, index v, index j)
            {
                if (v == FACE_2_VOL_EMPTY)
                    return;
                MPI_int rank;
                index val;
                JSGGhost.pLGlobalMapping->search(v, rank, val);
                if (rank != mpi.rank)
                    nGhost++;
            });
        ghostJSerialQuery.reserve(nGhost);
        forEachBasicInArray(
            *arraySerialAdj,
            [&](TAdjBatch &e, index i, index v, index j)
            {
                if (v == FACE_2_VOL_EMPTY)
                    return;
                MPI_int rank;
                index val;
                JSGGhost.pLGlobalMapping->search(v, rank, val);
                if (rank != mpi.rank)
                    ghostJSerialQuery.push_back(v);
            });
        PrintVec(ghostJSerialQuery, std::cout);
        JSGGhost.createGhostMapping(ghostJSerialQuery);
        // std::cout

        JSGGhost.createMPITypes();
        JSGGhost.pullOnce();
        forEachBasicInArray(
            *arraySerialAdj,
            [&](TAdjBatch &e, index i, index &v, index j)
            {
                if (v == FACE_2_VOL_EMPTY)
                    return;
                MPI_int rank;
                index val;
                JSGGhost.pLGhostMapping->search(v, rank, val);
                if (rank == -1)
                    v = JSG[val];
                else
                    v = JSGGhost[val];
                // std::cout << "nv" << v << " " << rank << std::endl;
            });
    }

    template <class TComp>
    void DistributeByPushLocal(std::shared_ptr<ArrayCascade<TComp>> &arraySerial,
                               std::shared_ptr<ArrayCascade<TComp>> &arrayDist,
                               const std::vector<index> &partitionIPushLocal, const std::vector<index> &partitionIPushLocalStarts)
    {
        arrayDist = std::make_shared<ArrayCascade<TComp>>(arraySerial.get()); // arraySerialAdj->arrayDistAdj
        arrayDist->createGlobalMapping();
        arrayDist->createGhostMapping(partitionIPushLocal, partitionIPushLocalStarts);
        arrayDist->createMPITypes();
        // InsertCheck(mpi);
        arrayDist->pullOnce();
        // InsertCheck(mpi);
    }

    template <class TPartitionIdx>
    void Partition2Serial2Global(const std::vector<TPartitionIdx> &partition, std::vector<index> &serial2Global, const MPIInfo &mpi, MPI_int nPart)
    {
        serial2Global.resize(partition.size());
        index iFill = 0;
        /****************************************/
        std::vector<index> numberAtLocal(nPart, 0);
        for (auto r : partition)
            numberAtLocal[r]++;
        std::vector<index> numberTotal(nPart), numberPrev(nPart);
        MPI_Allreduce(numberAtLocal.data(), numberTotal.data(), nPart, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        MPI_Scan(numberAtLocal.data(), numberPrev.data(), nPart, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        std::vector<index> numberTotalPlc(nPart + 1);
        numberTotalPlc[0] = 0;
        for (MPI_int r = 0; r < nPart; r++)
            numberTotalPlc[r + 1] = numberTotalPlc[r] + numberTotal[r], numberPrev[r] -= numberAtLocal[r];
        // 2 things here: accumulate total and substract local from prev
        /****************************************/
        numberAtLocal.assign(numberAtLocal.size(), 0);
        for (auto r : partition)
            serial2Global[iFill++] = (numberAtLocal[r]++) + numberTotalPlc[r] + numberPrev[r];
    }

    // for one-shot usage, partition data corresponds to mpi
    template <class TPartitionIdx>
    void Partition2LocalIdx(const std::vector<TPartitionIdx> &partition, std::vector<index> &localPush, std::vector<index> &localPushStart, const MPIInfo &mpi)
    {
        // localPushStart.resize(mpi.size);
        std::vector<index> localPushSizes(mpi.size, 0);
        for (auto r : partition)
        {
            localPushSizes[r]++;
            assert(r < mpi.size);
        }
        AccumulateRowSize(localPushSizes, localPushStart);
        localPush.resize(localPushStart[mpi.size]);
        localPushSizes.assign(mpi.size, 0);
        assert(partition.size() == localPush.size());
        for (index i = 0; i < partition.size(); i++)
            localPush[localPushStart[partition[i]] + (localPushSizes[partition[i]]++)] = i;
    }
}
/*


















*/

namespace DNDS
{

    struct ElemAttributes
    {
        index iPhy;
        Elem::ElemType type;
        Elem::tIntScheme intScheme;
    };

    typedef ArrayCascade<VarBatch<index>> tAdjArrayCascade;
    typedef ArrayCascade<Batch<index, 2>> tAdjStatic2ArrayCascade;
    typedef ArrayCascade<Batch<ElemAttributes, 1>> tElemAtrArrayCascade;
    typedef ArrayCascade<Vec3DBatch> tVec3DArrayCascade;

    class CompactFacedMeshSerialRW
    {
    public:
        MPIInfo mpi;
        // std::map<index, GmshPhyGrp> phyGrps;
        // std::vector<Elem::tPoint> points;
        // std::vector<GmshElem> volElems;
        // std::vector<GmshElem>
        index numCellGlobal;
        index numFaceGlobal;
        index numNodeGlobal;

        std::shared_ptr<tAdjArrayCascade> cell2node; // serial node index
        std::shared_ptr<tAdjArrayCascade> cell2face; // serial face index
        std::shared_ptr<tElemAtrArrayCascade> cellAtr;

        std::shared_ptr<tAdjArrayCascade> face2node;        // serial node index
        std::shared_ptr<tAdjStatic2ArrayCascade> face2cell; // serial cell index
        std::shared_ptr<tElemAtrArrayCascade> faceAtr;

        // std::shared_ptr<tAdjStatic2ArrayCascade> face2cellGlobal; // global cell index
        // // ! currently only cell needs to be indexed explicitly in global order.
        // // ! for only cell data are currently run-time communicated

        std::shared_ptr<tVec3DArrayCascade> nodeCoords;

        // std::vector<index> iCellsSerialDist;

        std::shared_ptr<tAdjArrayCascade> cell2nodeDist;
        std::shared_ptr<tAdjArrayCascade> cell2faceDist;
        std::shared_ptr<tElemAtrArrayCascade> cellAtrDist;

        std::shared_ptr<tAdjArrayCascade> face2nodeDist;
        std::shared_ptr<tAdjStatic2ArrayCascade> face2cellDist;
        std::shared_ptr<tElemAtrArrayCascade> faceAtrDist;

        std::shared_ptr<tVec3DArrayCascade> nodeCoordsDist;

        std::shared_ptr<GlobalOffsetsMapping> pCellGlobalMapping;
        std::shared_ptr<GlobalOffsetsMapping> pFaceGlobalMapping;
        std::shared_ptr<GlobalOffsetsMapping> pNodeGlobalMapping;
        std::shared_ptr<OffsetAscendIndexMapping> pCellGhostMapping;
        std::shared_ptr<OffsetAscendIndexMapping> pFaceGhostMapping;
        std::shared_ptr<OffsetAscendIndexMapping> pNodeGhostMapping;

        std::shared_ptr<tAdjArrayCascade> cell2nodeDistGhost;
        std::shared_ptr<tAdjArrayCascade> cell2faceDistGhost;
        std::shared_ptr<tElemAtrArrayCascade> cellAtrDistGhost;
        std::shared_ptr<ArrayCascadePair<tAdjArrayCascade::tComponent>> cell2facePair;
        ArrayCascadeLocal<tAdjArrayCascade::tComponent> cell2faceLocal;
        std::shared_ptr<ArrayCascadePair<tAdjArrayCascade::tComponent>> cell2nodePair;
        ArrayCascadeLocal<tAdjArrayCascade::tComponent> cell2nodeLocal;
        std::shared_ptr<ArrayCascadePair<tElemAtrArrayCascade::tComponent>> cellAtrPair;
        ArrayCascadeLocal<tElemAtrArrayCascade::tComponent> cellAtrLocal;

        std::shared_ptr<tAdjArrayCascade> face2nodeDistGhost;
        std::shared_ptr<tAdjStatic2ArrayCascade> face2cellDistGhost;
        std::shared_ptr<tElemAtrArrayCascade> faceAtrDistGhost;
        std::shared_ptr<ArrayCascadePair<tAdjStatic2ArrayCascade::tComponent>> face2cellPair;
        ArrayCascadeLocal<tAdjStatic2ArrayCascade::tComponent> face2cellLocal;
        std::shared_ptr<ArrayCascadePair<tAdjArrayCascade::tComponent>> face2nodePair;
        ArrayCascadeLocal<tAdjArrayCascade::tComponent> face2nodeLocal;
        std::shared_ptr<ArrayCascadePair<tElemAtrArrayCascade::tComponent>> faceAtrPair;
        ArrayCascadeLocal<tElemAtrArrayCascade::tComponent> faceAtrLocal;

        std::shared_ptr<tVec3DArrayCascade> nodeCoordsDistGhost;
        std::shared_ptr<ArrayCascadePair<tVec3DArrayCascade::tComponent>> nodeCoordsPair;
        ArrayCascadeLocal<tVec3DArrayCascade::tComponent> nodeCoordsLocal;

        ArrayCascadeLocal<tAdjStatic2ArrayCascade::tComponent> face2cellRefLocal;

        // std::shared_ptr<tAdjArrayCascade> cell2Localnode;
        // std::shared_ptr<tAdjArrayCascade> cell2Localface;
        // std::shared_ptr<tAdjStatic2ArrayCascade> face2Localcell;
        // std::shared_ptr<tAdjArrayCascade> face2Localnode;

        CompactFacedMeshSerialRW(SerialGmshReader2d &gmshReader, const MPIInfo &nmpi) : mpi(nmpi)
        {
            assert(gmshReader.vol2face.size() == gmshReader.volElems.size());

            // copy cell2node
            cell2node = std::make_shared<tAdjArrayCascade>(
                tAdjArrayCascade::tContext(
                    [&](index i) -> rowsize
                    {
                        assert(gmshReader.volElems[i].nodeList.size() == Elem::ElementManager(gmshReader.volElems[i].elemType, 0).getNNode());
                        return Elem::ElementManager(gmshReader.volElems[i].elemType, 0).getNNode();
                    },
                    gmshReader.volElems.size()),
                mpi);
            for (index iv = 0; iv < cell2node->size(); iv++)
            {
                auto e = (*cell2node)[iv];
                for (rowsize in = 0; in < e.size(); in++)
                    e[in] = gmshReader.volElems[iv].nodeList[in];
            }

            // copy face2node
            face2node = std::make_shared<tAdjArrayCascade>(
                tAdjArrayCascade::tContext(
                    [&](index i) -> rowsize
                    {
                        assert(gmshReader.faceElems[i].nodeList.size() == Elem::ElementManager(gmshReader.faceElems[i].elemType, 0).getNNode());
                        return Elem::ElementManager(gmshReader.faceElems[i].elemType, 0).getNNode();
                    },
                    gmshReader.faceElems.size()),
                mpi);
            for (index iff = 0; iff < face2node->size(); iff++)
            {
                auto e = (*face2node)[iff];
                for (rowsize in = 0; in < e.size(); in++)
                    e[in] = gmshReader.faceElems[iff].nodeList[in];
            }

            // copy cell2face
            cell2face = std::make_shared<tAdjArrayCascade>(
                tAdjArrayCascade::tContext(
                    [&](index i) -> rowsize
                    {
                        assert(gmshReader.vol2face[i].size() == Elem::ElementManager(gmshReader.volElems[i].elemType, 0).getNFace());
                        return gmshReader.vol2face[i].size();
                    },
                    gmshReader.vol2face.size()),
                mpi);
            for (index iv = 0; iv < cell2face->size(); iv++)
            {
                auto e = (*cell2face)[iv];
                for (rowsize in = 0; in < e.size(); in++)
                    e[in] = gmshReader.vol2face[iv][in];
            }
            assert(cell2face->size() == cell2node->size());

            // copy face2cell
            face2cell = std::make_shared<tAdjStatic2ArrayCascade>(
                tAdjStatic2ArrayCascade::tContext(
                    gmshReader.faceElems.size()),
                mpi);
            for (index iff = 0; iff < face2cell->size(); iff++)
            {
                auto e = (*face2cell)[iff];
                e[0] = gmshReader.face2vol[iff].first;
                e[1] = gmshReader.face2vol[iff].second;
            }

            // copy cell atr
            cellAtr = std::make_shared<tElemAtrArrayCascade>(
                tElemAtrArrayCascade::tContext(cell2face->size()),
                mpi);
            for (index iv = 0; iv < cellAtr->size(); iv++)
            {
                auto &e = (*cellAtr)[iv][0];
                e.iPhy = gmshReader.volElems[iv].phyGrp;
                e.type = gmshReader.volElems[iv].elemType;
                e.intScheme = -1; // init value
            }

            // copy face atr
            faceAtr = std::make_shared<tElemAtrArrayCascade>(
                tElemAtrArrayCascade::tContext(face2cell->size()),
                mpi);
            for (index iff = 0; iff < face2cell->size(); iff++)
            {
                auto &e = (*faceAtr)[iff][0];
                if (gmshReader.readPhyGrps[gmshReader.faceElems[iff].phyGrp].name == "bc-1")
                    e.iPhy = index(BoundaryType::Wall);
                else if (gmshReader.readPhyGrps[gmshReader.faceElems[iff].phyGrp].name == "bc-2")
                    e.iPhy = index(BoundaryType::Farfield);
                else if (gmshReader.faceElems[iff].phyGrp == -1)
                    e.iPhy = index(BoundaryType::Inner);
                else
                {
                    std::cout << gmshReader.readPhyGrps[gmshReader.faceElems[iff].phyGrp].name << std::endl;
                    assert(false);
                }
                e.type = gmshReader.faceElems[iff].elemType;
                e.intScheme = -1; // init value
                // std::cout << e.iPhy << std::endl;
            }

            // copy points
            nodeCoords = std::make_shared<tVec3DArrayCascade>(
                tVec3DArrayCascade::tContext(gmshReader.readPoints.size()),
                mpi);
            for (index ip = 0; ip < nodeCoords->size(); ip++)
                (*nodeCoords)[ip].p() = gmshReader.readPoints[ip];

            numNodeGlobal = nodeCoords->obtainTotalSize();
            numFaceGlobal = face2node->obtainTotalSize();
            numCellGlobal = cell2node->obtainTotalSize();
        }

        void MetisSerialPartitionKWay(MPI_int oprank)
        {
            if (mpi.rank == oprank)
            {
                /*******************************************************/
                // derive cell2cell graph
                std::vector<index> cell2cellSiz(cell2node->size(), 0);
                for (index iv = 0; iv < cell2face->size(); iv++)
                {
                    auto faceList = (*cell2face)[iv];
                    for (index iff = 0; iff < faceList.size(); iff++)
                        if ((*face2cell)[faceList[iff]][1] != FACE_2_VOL_EMPTY) // then face connects another cell
                            cell2cellSiz[iv]++;
                }
                std::vector<idx_t> cell2cellStarts(cell2cellSiz.size() + 1);
                cell2cellStarts[0] = 0;
                for (index iv = 0; iv < cell2cellSiz.size(); iv++)
                    cell2cellStarts[iv + 1] = cell2cellStarts[iv] + cell2cellSiz[iv];
                std::vector<idx_t> cell2cell(cell2cellStarts[cell2cellSiz.size()]);

                index icell2cellfill = 0;
                for (index iv = 0; iv < cell2cellSiz.size(); iv++)
                {
                    auto faceList = (*cell2face)[iv];
                    for (index iff = 0; iff < faceList.size(); iff++)
                        if ((*face2cell)[faceList[iff]][1] != FACE_2_VOL_EMPTY) // then face connects another cell
                            if ((*face2cell)[faceList[iff]][0] == iv)
                                cell2cell[icell2cellfill++] = (*face2cell)[faceList[iff]][1];
                            else
                                cell2cell[icell2cellfill++] = (*face2cell)[faceList[iff]][0];
                }
                assert(icell2cellfill == cell2cell.size());
                /*******************************************************/

                idx_t ncell = cell2cellSiz.size();
                idx_t ncons = 1;
                idx_t nparts = mpi.size;
                idx_t options[METIS_NOPTIONS];
                options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
                options[METIS_OPTION_CTYPE] = METIS_CTYPE_RM;
                options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_GROW;
                options[METIS_OPTION_RTYPE] = METIS_RTYPE_FM;
                options[METIS_OPTION_NO2HOP] = 0;
                options[METIS_OPTION_NCUTS] = 1;
                options[METIS_OPTION_NITER] = 10;
                options[METIS_OPTION_UFACTOR] = 30;
                options[METIS_OPTION_MINCONN] = 0;
                options[METIS_OPTION_CONTIG] = 1; // ! forcing contigious partition now ? necessary?
                options[METIS_OPTION_SEED] = 0;   // ! seeding 0 for determined result
                options[METIS_OPTION_NUMBERING] = 0;
                options[METIS_OPTION_DBGLVL] = METIS_DBG_TIME | METIS_DBG_IPART;

                idx_t objval;
                std::vector<idx_t> partition;
                partition.resize(cell2cellSiz.size());
                std::vector<idx_t> iGlobal; // iGlobal[iCell_Serial] = iCell_Global
                iGlobal.reserve(cell2cellSiz.size());

                if (nparts > 1)
                {
                    int ret = METIS_PartGraphKway(&ncell, &ncons, cell2cellStarts.data(), cell2cell.data(), NULL, NULL, NULL, &nparts, NULL, NULL, options, &objval, partition.data());
                    if (ret != METIS_OK)
                    {
                        log() << "METIS returned not OK: [" << ret << "]" << std::endl;
                        assert(false);
                    }
                }
                else
                    partition.assign(partition.size(), 0);

                std::vector<idx_t> partitionFace(face2cell->size());
                std::vector<idx_t> partitionNode(nodeCoords->size());
                forEachBasicInArray(
                    *cell2face,
                    [&](tAdjArrayCascade::tComponent &e, index i, index v, index j)
                    {
                        partitionFace[v] = partition[i];
                    });
                forEachBasicInArray(
                    *cell2node,
                    [&](tAdjArrayCascade::tComponent &e, index i, index v, index j)
                    {
                        partitionNode[v] = partition[i];
                    });

                std::vector<index> localIdxNode, localIdxStartNode, localIdxFace, localIdxStartFace, localIdx, localIdxStart;
                Partition2LocalIdx(partition, localIdx, localIdxStart, mpi);
                Partition2LocalIdx(partitionNode, localIdxNode, localIdxStartNode, mpi);
                Partition2LocalIdx(partitionFace, localIdxFace, localIdxStartFace, mpi);
                std::vector<index> SG, SGNode, SGFace;
                Partition2Serial2Global(partition, SG, mpi, mpi.size);
                Partition2Serial2Global(partitionNode, SGNode, mpi, mpi.size);
                Partition2Serial2Global(partitionFace, SGFace, mpi, mpi.size);

                auto pncell2face = std::make_shared<tAdjArrayCascade>(*cell2face);
                auto pncell2node = std::make_shared<tAdjArrayCascade>(*cell2node);
                auto pnface2cell = std::make_shared<tAdjStatic2ArrayCascade>(*face2cell);
                auto pnface2node = std::make_shared<tAdjArrayCascade>(*face2node);

                ConvertAdjSerial2Global(pncell2face, SGFace, mpi);
                ConvertAdjSerial2Global(pncell2node, SGNode, mpi);
                ConvertAdjSerial2Global(pnface2cell, SG, mpi);
                ConvertAdjSerial2Global(pnface2node, SGNode, mpi);

                DistributeByPushLocal(pncell2face, cell2faceDist, localIdx, localIdxStart);
                DistributeByPushLocal(pncell2node, cell2nodeDist, localIdx, localIdxStart);
                DistributeByPushLocal(cellAtr, cellAtrDist, localIdx, localIdxStart);

                DistributeByPushLocal(pnface2cell, face2cellDist, localIdxFace, localIdxStartFace);
                DistributeByPushLocal(pnface2node, face2nodeDist, localIdxFace, localIdxStartFace);
                DistributeByPushLocal(faceAtr, faceAtrDist, localIdxFace, localIdxStartFace);

                DistributeByPushLocal(nodeCoords, nodeCoordsDist, localIdxNode, localIdxStartNode);
            }
            else
            {
                std::vector<idx_t> partitionFace(0);
                std::vector<idx_t> partitionNode(0);
                std::vector<idx_t> partition(0);
                assert(cell2node->size() == 0 && face2node->size() == 0 && nodeCoords->size() == 0);

                std::vector<index> localIdxNode, localIdxStartNode, localIdxFace, localIdxStartFace, localIdx, localIdxStart;
                Partition2LocalIdx(partition, localIdx, localIdxStart, mpi);
                Partition2LocalIdx(partitionNode, localIdxNode, localIdxStartNode, mpi);
                Partition2LocalIdx(partitionFace, localIdxFace, localIdxStartFace, mpi);
                std::vector<index> SG, SGNode, SGFace;
                Partition2Serial2Global(partition, SG, mpi, mpi.size);
                Partition2Serial2Global(partitionNode, SGNode, mpi, mpi.size);
                Partition2Serial2Global(partitionFace, SGFace, mpi, mpi.size);

                auto pncell2face = std::make_shared<tAdjArrayCascade>(*cell2face);
                auto pncell2node = std::make_shared<tAdjArrayCascade>(*cell2node);
                auto pnface2cell = std::make_shared<tAdjStatic2ArrayCascade>(*face2cell);
                auto pnface2node = std::make_shared<tAdjArrayCascade>(*face2node);

                ConvertAdjSerial2Global(pncell2face, SGFace, mpi);
                ConvertAdjSerial2Global(pncell2node, SGNode, mpi);
                ConvertAdjSerial2Global(pnface2cell, SG, mpi);
                ConvertAdjSerial2Global(pnface2node, SGNode, mpi);

                DistributeByPushLocal(pncell2face, cell2faceDist, localIdx, localIdxStart);
                DistributeByPushLocal(pncell2node, cell2nodeDist, localIdx, localIdxStart);
                DistributeByPushLocal(cellAtr, cellAtrDist, localIdx, localIdxStart);

                DistributeByPushLocal(pnface2cell, face2cellDist, localIdxFace, localIdxStartFace);
                DistributeByPushLocal(pnface2node, face2nodeDist, localIdxFace, localIdxStartFace);
                DistributeByPushLocal(faceAtr, faceAtrDist, localIdxFace, localIdxStartFace);

                DistributeByPushLocal(nodeCoords, nodeCoordsDist, localIdxNode, localIdxStartNode); // OPT could optimize the mapping sharing here
            }
        }

        void ClearSerial()
        {
            cell2face.reset();
            cell2node.reset();
            face2cell.reset();
            face2node.reset();
            cellAtr.reset();
            faceAtr.reset();
            nodeCoords.reset();

            cell2faceDist->ForgetFather();
            cell2nodeDist->ForgetFather();
            face2cellDist->ForgetFather();
            face2nodeDist->ForgetFather();
            cellAtrDist->ForgetFather();
            faceAtrDist->ForgetFather();
            nodeCoordsDist->ForgetFather();
        }

        /**
         * @brief builds serial: cell2node, nodeCoords and cellAtr
         * all data serial output should borrow GGindexings from here
         */
        void BuildSerialOut(MPI_int oprank)
        {
            cell2node = std::make_shared<tAdjArrayCascade>(cell2nodeDist.get());
            cellAtr = std::make_shared<tElemAtrArrayCascade>(cellAtrDist.get());
            nodeCoords = std::make_shared<tVec3DArrayCascade>(nodeCoordsDist.get());
            std::vector<index> serialPullCell;
            std::vector<index> serialPullNode;
            assert(cell2nodeDist->obtainTotalSize() == numCellGlobal);
            if (mpi.rank == oprank)
            {
                serialPullCell.resize(numCellGlobal);
                serialPullNode.resize(numNodeGlobal);
                for (index i = 0; i < serialPullCell.size(); i++)
                    serialPullCell[i] = i;
                for (index i = 0; i < serialPullNode.size(); i++)
                    serialPullNode[i] = i;
            }
            else
            {
            }
            cell2node->createGlobalMapping();
            cell2node->createGhostMapping(serialPullCell);
            cell2node->createMPITypes();
            cell2node->pullOnce();

            cellAtr->BorrowGGIndexing(*cell2node);
            cellAtr->createMPITypes();
            cellAtr->pullOnce();

            nodeCoords->createGlobalMapping();
            nodeCoords->createGhostMapping(serialPullNode);
            nodeCoords->createMPITypes();
            nodeCoords->pullOnce();
        }

        /**
         * @brief builds ghost for cell, then for coords
         *
         */
        void BuildGhosts()
        {
            face2cellDistGhost = std::make_shared<tAdjStatic2ArrayCascade>(face2cellDist.get());
            face2cellDistGhost->createGlobalMapping();
            pFaceGlobalMapping = face2cellDistGhost->pLGlobalMapping;

            // get ghost face set
            index nghostFaces = 0;
            forEachBasicInArray(
                *cell2faceDist,
                [&](tAdjArrayCascade::tComponent &c2f, index ic, index ifg, index icf)
                {
                    MPI_int rank;
                    index val;
                    bool found = pFaceGlobalMapping->search(ifg, rank, val);
                    assert(found);
                    if (rank != mpi.rank)
                        nghostFaces++;
                });
            std::vector<index> ghostFaces;
            ghostFaces.reserve(nghostFaces);
            forEachBasicInArray(
                *cell2faceDist,
                [&](tAdjArrayCascade::tComponent &c2f, index ic, index ifg, index icf)
                {
                    MPI_int rank;
                    index val;
                    bool found = pFaceGlobalMapping->search(ifg, rank, val);
                    assert(found);
                    if (rank != mpi.rank)
                        ghostFaces.push_back(ifg);
                });

            face2cellDistGhost->createGhostMapping(ghostFaces);
            face2cellDistGhost->createMPITypes();
            face2cellDistGhost->pullOnce();

            pFaceGhostMapping = face2cellDistGhost->pLGhostMapping;
            face2cellPair = std::make_shared<decltype(face2cellPair)::element_type>(*face2cellDist, *face2cellDistGhost);

            face2nodeDistGhost = std::make_shared<tAdjArrayCascade>(face2nodeDist.get());
            face2nodeDistGhost->BorrowGGIndexing(*face2cellDistGhost);
            face2nodeDistGhost->createMPITypes();
            face2nodeDistGhost->pullOnce();
            face2nodePair = std::make_shared<decltype(face2nodePair)::element_type>(*face2nodeDist, *face2nodeDistGhost);

            faceAtrDistGhost = std::make_shared<tElemAtrArrayCascade>(faceAtrDist.get());
            faceAtrDistGhost->BorrowGGIndexing(*face2cellDistGhost);
            faceAtrDistGhost->createMPITypes();
            faceAtrDistGhost->pullOnce();
            faceAtrPair = std::make_shared<decltype(faceAtrPair)::element_type>(*faceAtrDist, *faceAtrDistGhost);

            cell2faceDistGhost = std::make_shared<tAdjArrayCascade>(cell2faceDist.get());
            cell2faceDistGhost->createGlobalMapping();
            pCellGlobalMapping = cell2faceDistGhost->pLGlobalMapping;

            // get ghost cell set
            index nghostCells = 0;
            forEachInArray(
                *cell2faceDist,
                [&](tAdjArrayCascade::tComponent &c2f, index ic)
                {
                    for (int iff = 0; iff < c2f.size(); iff++)
                    {
                        MPI_int rank;
                        index val;
                        bool rt0 = pFaceGhostMapping->search_indexAppend(c2f[iff], rank, val);
                        // if (mpi.rank == 0)
                        //     std::cout << val << std::endl;
                        assert(rt0);
                        auto f2c = (*face2cellPair)[val];

                        if (f2c[1] == FACE_2_VOL_EMPTY)
                            continue;
                        index icGlob = pCellGlobalMapping->operator()(mpi.rank, ic);
                        index bCellGlob = f2c[0] == icGlob ? f2c[1] : f2c[0];

                        bool rt = pCellGlobalMapping->search(bCellGlob, rank, val);
                        assert(rt);
                        if (rank != mpi.rank)
                            nghostCells++;
                    }
                });

            std::vector<index> ghostCells;
            ghostCells.reserve(nghostCells);
            forEachInArray(
                *cell2faceDist,
                [&](tAdjArrayCascade::tComponent &c2f, index ic)
                {
                    for (int iff = 0; iff < c2f.size(); iff++)
                    {
                        MPI_int rank;
                        index val;
                        bool rt0 = pFaceGhostMapping->search_indexAppend(c2f[iff], rank, val);
                        assert(rt0);
                        auto f2c = (*face2cellPair)[val];

                        if (f2c[1] == FACE_2_VOL_EMPTY)
                            continue;
                        index icGlob = pCellGlobalMapping->operator()(mpi.rank, ic);
                        index bCellGlob = f2c[0] == icGlob ? f2c[1] : f2c[0];

                        bool rt = pCellGlobalMapping->search(bCellGlob, rank, val);
                        assert(rt);
                        if (rank != mpi.rank)
                            ghostCells.push_back(bCellGlob);
                    }
                });

            cell2faceDistGhost->createGhostMapping(ghostCells);
            cell2faceDistGhost->createMPITypes();
            cell2faceDistGhost->pullOnce(); //! cell2face's ghost is redundant here, as no ghost face is actually done
            pCellGhostMapping = cell2faceDistGhost->pLGhostMapping;
            cell2facePair = std::make_shared<decltype(cell2facePair)::element_type>(*cell2faceDist, *cell2faceDistGhost);

            cell2nodeDistGhost = std::make_shared<tAdjArrayCascade>(cell2nodeDist.get()); //! note: don't write as std::shared_ptr<>() which mistakes as a pointer sharing
            cell2nodeDistGhost->BorrowGGIndexing(*cell2faceDistGhost);
            cell2nodeDistGhost->createMPITypes();
            cell2nodeDistGhost->pullOnce();
            cell2nodePair = std::make_shared<decltype(cell2nodePair)::element_type>(*cell2nodeDist, *cell2nodeDistGhost);

            cellAtrDistGhost = std::make_shared<tElemAtrArrayCascade>(cellAtrDist.get());
            cellAtrDistGhost->BorrowGGIndexing(*cell2faceDistGhost);
            cellAtrDistGhost->createMPITypes();
            cellAtrDistGhost->pullOnce();
            cellAtrPair = std::make_shared<decltype(cellAtrPair)::element_type>(*cellAtrDist, *cellAtrDistGhost);

            nodeCoordsDistGhost = std::make_shared<tVec3DArrayCascade>(nodeCoordsDist.get());
            nodeCoordsDistGhost->createGlobalMapping();
            pNodeGlobalMapping = nodeCoordsDistGhost->pLGlobalMapping;

            // get ghost node set
            index nghostNode = 0;
            forEachBasicInArrayPair(
                *cell2nodePair,
                [&](tAdjArrayCascade::tComponent &c2n, index ic, index in, index icn)
                {
                    MPI_int rank;
                    index val;
                    bool rt = pNodeGlobalMapping->search(in, rank, val);
                    assert(rt);
                    if (rank != mpi.rank)
                        nghostNode++;
                });
            std::vector<index> ghostNodes;
            ghostNodes.reserve(nghostNode);
            forEachBasicInArrayPair(
                *cell2nodePair,
                [&](tAdjArrayCascade::tComponent &c2n, index ic, index in, index icn)
                {
                    MPI_int rank;
                    index val;
                    bool rt = pNodeGlobalMapping->search(in, rank, val);
                    assert(rt);
                    if (rank != mpi.rank)
                        ghostNodes.push_back(in);
                });
            nodeCoordsDistGhost->createGhostMapping(ghostNodes);
            nodeCoordsDistGhost->createMPITypes();
            nodeCoordsDistGhost->pullOnce();
            nodeCoordsPair = std::make_shared<decltype(nodeCoordsPair)::element_type>(*nodeCoordsDist, *nodeCoordsDistGhost);
            pNodeGhostMapping = nodeCoordsDistGhost->pLGhostMapping;

            // convert Adj arrays to point to local arrays
            // cell2face only main part is converted , !! ghost part ignored
            // cell2node all is converted
            // !all Adj after this should not be communicated

            forEachBasicInArrayPair(
                *cell2facePair,
                [&](tAdjArrayCascade::tComponent &c2f, index ic, index &iff, index icf)
                {
                    MPI_int rank;
                    index val;
                    bool found = pFaceGhostMapping->search_indexAppend(iff, rank, val);
                    if (found)
                        iff = val; // global -> local_pair
                    else           // !else not local face, remains -1- global_face
                        iff = -1 - iff;
                });
            forEachBasicInArrayPair(
                *cell2nodePair,
                [&](tAdjArrayCascade::tComponent &c2n, index ic, index &in, index icn)
                {
                    MPI_int rank;
                    index val;
                    bool found = pNodeGhostMapping->search_indexAppend(in, rank, val);
                    assert(found); // every cell in cell pair must have all nodes in pair
                    in = val;
                });
            forEachBasicInArrayPair(
                *face2cellPair,
                [&](tAdjStatic2ArrayCascade::tComponent &f2c, index iff, index &ic, index ifc)
                {
                    // if (mpi.rank == 0)
                    //     std::cout << ic << std::endl;
                    if (ic == FACE_2_VOL_EMPTY)
                        return;
                    MPI_int rank;
                    index val;
                    bool found = pCellGhostMapping->search_indexAppend(ic, rank, val);
                    assert(found); // every face in adjacent set must have non empty adj cell in cell pair
                    ic = val;
                });
            forEachBasicInArrayPair(
                *face2nodePair,
                [&](tAdjArrayCascade::tComponent &f2n, index iff, index &in, index ifn)
                {
                    MPI_int rank;
                    index val;
                    bool found = pNodeGhostMapping->search_indexAppend(in, rank, val);
                    assert(found); // every face in adjacent set must have all nodes in pair
                    in = val;
                });

            // now the mesh data:
            //  cell2nodePair, cell2faceDist(not pair), face2nodePair, face2cellPair, , nodeCoordsPair, cellAtrPair, faceAtrPair

            // convert to Local triplets to make copying convenient
            cell2nodeLocal.dist = cell2nodeDist;
            cell2nodeLocal.ghost = cell2nodeDistGhost;
            cell2nodeLocal.pair = cell2nodePair;
            cellAtrLocal.dist = cellAtrDist;
            cellAtrLocal.ghost = cellAtrDistGhost;
            cellAtrLocal.pair = cellAtrPair;
            cell2faceLocal.dist = cell2faceDist;
            cell2faceLocal.ghost = cell2faceDistGhost;
            cell2faceLocal.pair = cell2facePair;

            face2nodeLocal.dist = face2nodeDist;
            face2nodeLocal.ghost = face2nodeDistGhost;
            face2nodeLocal.pair = face2nodePair;
            face2cellLocal.dist = face2cellDist;
            face2cellLocal.ghost = face2cellDistGhost;
            face2cellLocal.pair = face2cellPair;
            faceAtrLocal.dist = faceAtrDist;
            faceAtrLocal.ghost = faceAtrDistGhost;
            faceAtrLocal.pair = faceAtrPair;

            nodeCoordsLocal.dist = nodeCoordsDist;
            nodeCoordsLocal.ghost = nodeCoordsDistGhost;
            nodeCoordsLocal.pair = nodeCoordsPair;

            // face2cellRefLocal
            face2cellRefLocal.dist = std::make_shared<tAdjStatic2ArrayCascade>(tAdjStatic2ArrayCascade::tContext(face2cellLocal.dist->size()), mpi);
            face2cellRefLocal.CreateGhostCopyComm(face2cellLocal);
            // std::cout << "FUCKED" << face2cellRefLocal.pair->size() << std::endl;
            forEachInArrayPair(
                *face2cellRefLocal.pair,
                [&](tAdjStatic2ArrayCascade::tComponent &f2cr, index iff)
                {
                    auto f2c = face2cellLocal[iff];
                    // cell 0
                    auto c2f = cell2faceLocal[f2c[0]];
                    int ic2f;
                    for (ic2f = 0; ic2f < c2f.size(); ic2f++)
                        if (c2f[ic2f] == iff)
                        {
                            f2cr[0] = ic2f;
                            break;
                        }
                    assert(ic2f < c2f.size());
                    // cell 1
                    if (f2c[1] == FACE_2_VOL_EMPTY)
                        return;
                    c2f = cell2faceLocal[f2c[1]];
                    for (ic2f = 0; ic2f < c2f.size(); ic2f++)
                        if (c2f[ic2f] == iff)
                        {
                            f2cr[1] = ic2f;
                            break;
                        }
                    // std::cout << "R Cell = " << ic2f << std::endl;
                    assert(ic2f < c2f.size());
                });
        }

        // c2n must be pointing to local
        void LoadCoords(const tAdjArrayCascade::tComponent &c2n, Eigen::MatrixXd &coords)
        {
            coords.resize(3, c2n.size());
            for (int in = 0; in < c2n.size(); in++)
                coords(Eigen::all, in) = nodeCoordsLocal[c2n[in]].p();
        }

        void FacePParam2Cell(index iCell, index iF2C, index iFace, const tAdjArrayCascade::tComponent &f2n, Elem::ElementManager &eFace, const Elem::tPoint &pFace, Elem::tPoint &pCell)
        {
            Elem::ElementManager eCell(this->cellAtrLocal[iCell][0].type, 0); // int scheme is not relevant here
            index iFaceAtCell = this->face2cellRefLocal[iFace][iF2C];
            std::vector<index> fNodeSTD(eFace.getNNode());
            eCell.SubstractFaceNodes(iFaceAtCell, eFace, this->cell2nodeLocal[iCell], fNodeSTD);
            // if (mpi.rank == 0)
            // {
            //     std::cout << iCell << outputDelim << iFace << outputDelim;
            //     PrintVec(fNodeSTD, std::cout);
            //     std::cout << f2n << std::endl;
            // }
            eCell.FaceSpace2VolSpace(iFaceAtCell, pFace, pCell, f2n, fNodeSTD);
        }

        // struct CellIterContext
        // {
        //     index iCell; // local-ghost appended index
        //     VarBatch<index> c2f;
        //     VarBatch<index> c2n;
        //     VarBatch<index> c2c;
        //     Eigen::MatrixXd coords;
        //     Batch<ElemAttributes, 1> atr;
        //     Elem::ElementManager elemMan;
        // };

        void PrintSerialPartPltASCIIDBG(const std::string &fname, MPI_int oprank) //! supports 2d here
        {
            if (mpi.rank != oprank)
                return;

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
            fout << "VARIABLES = \"x\", \"y\", \"volume\", \"iPart\"\n" // 2d mesh so only x y
                 << "Zone N =" << nodeCoords->size() << ","
                 << " E = " << cell2node->size() << ","
                 << "VARLOCATION=([1-2]=NODAL,[3-4]=CELLCENTERED)"
                 << "\n,"
                 << "DATAPACKING=BLOCK, ZONETYPE=FEQUADRILATERAL"
                 << "\n";
            fout << std::setprecision(16);
            forEachInArray(
                *nodeCoords,
                [&](tVec3DArrayCascade::tComponent &e, index i)
                {
                    fout << e.p()(0) << "\n";
                });
            forEachInArray(
                *nodeCoords,
                [&](tVec3DArrayCascade::tComponent &e, index i)
                {
                    fout << e.p()(1) << "\n";
                });
            real vsum = 0.0;

            forEachInArray(
                *cell2node,
                [&](tAdjArrayCascade::tComponent &c2n, index iv)
                {
                    auto atr = (*cellAtr)[iv][0];
                    Elem::ElementManager elemMan(atr.type, 0);
                    switch (elemMan.getPspace())
                    {
                    case Elem::ParamSpace::TriSpace:
                        elemMan.setType(atr.type, schemeTri);
                        break;
                    case Elem::ParamSpace::QuadSpace:
                        elemMan.setType(atr.type, schemeQuad);
                        break;
                    default:
                        assert(false);
                    }
                    real v = 0.0;
                    Eigen::MatrixXd coords(3, elemMan.getNNode());
                    for (int in = 0; in < elemMan.getNNode(); in++)
                        coords(Eigen::all, in) = (*nodeCoords)[c2n[in]].p(); // the coords data ordering is conforming with that of cell2node

                    elemMan.Integration(
                        v,
                        [&](real &vinc, int m, DNDS::Elem::tPoint &p, DNDS::Elem::tDiFj &DiNj) -> void
                        {
                            vinc = DNDS::Elem::DiNj2Jacobi(DiNj, coords)({0, 1}, {0, 1}).determinant();
                            if (vinc < 0)
                                log() << "Error: 2d vol orientation wrong or distorted" << std::endl;
                            assert(vinc > 0);
                        });
                    fout << v << "\n";
                    vsum += v;
                });
            log() << "Sum Volume [" << vsum << "]" << std::endl;
            for (index iv = 0; iv < numCellGlobal; iv++)
            {
                MPI_int r;
                index v;
                cell2node->pLGlobalMapping->search(iv, r, v);
                fout << r << "\n";
            }

            forEachInArray(
                *cell2node,
                [&](tAdjArrayCascade::tComponent &c2n, index iv)
                {
                    Elem::ElementManager elemMan((*cellAtr)[iv][0].type, 0);
                    switch (elemMan.getPspace())
                    {
                    case Elem::ParamSpace::TriSpace:
                        fout << c2n[0] + 1 << " " << c2n[1] + 1 << " " << c2n[2] + 1 << " " << c2n[2] + 1 << '\n';
                        break;
                    case Elem::ParamSpace::QuadSpace:
                        fout << c2n[0] + 1 << " " << c2n[1] + 1 << " " << c2n[2] + 1 << " " << c2n[3] + 1 << '\n';
                        break;
                    default:
                        assert(false);
                    }
                });
            fout.close();
        }

        /**
         * @brief names(idata) data(idata, ivolume)
         *
         */
        template <class FNAMES, class FDATA>
        void PrintSerialPartPltASCIIDataArray(const std::string &fname, MPI_int oprank,
                                              int arraySiz,
                                              FNAMES &&names,
                                              FDATA &&data) //! supports 2d here
        {
            if (mpi.rank != oprank)
                return;

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
            fout << "VARIABLES = \"x\", \"y\", \"iPart\"";
            for (int idata = 0; idata < arraySiz; idata++)
                fout << ", \"" << names(idata) << "\"";
            fout << "\n" // 2d mesh so only x y
                 << "Zone N =" << nodeCoords->size() << ","
                 << " E = " << cell2node->size() << ","
                 << "VARLOCATION=([1-2]=NODAL,[3-" << arraySiz + 3 << "]=CELLCENTERED)"
                 << "\n,"
                 << "DATAPACKING=BLOCK, ZONETYPE=FEQUADRILATERAL"
                 << "\n";
            fout << std::setprecision(16);
            forEachInArray(
                *nodeCoords,
                [&](tVec3DArrayCascade::tComponent &e, index i)
                {
                    fout << e.p()(0) << "\n";
                });
            forEachInArray(
                *nodeCoords,
                [&](tVec3DArrayCascade::tComponent &e, index i)
                {
                    fout << e.p()(1) << "\n";
                });
            for (index iv = 0; iv < numCellGlobal; iv++)
            {
                MPI_int r;
                index v;
                cell2node->pLGlobalMapping->search(iv, r, v);
                fout << r << "\n";
            }
            for (int idata = 0; idata < arraySiz; idata++)
            {
                for (index iv = 0; iv < numCellGlobal; iv++)
                {
                    fout << data(idata, iv) << "\n";
                }
            }

            forEachInArray(
                *cell2node,
                [&](tAdjArrayCascade::tComponent &c2n, index iv)
                {
                    Elem::ElementManager elemMan((*cellAtr)[iv][0].type, 0);
                    switch (elemMan.getPspace())
                    {
                    case Elem::ParamSpace::TriSpace:
                        fout << c2n[0] + 1 << " " << c2n[1] + 1 << " " << c2n[2] + 1 << " " << c2n[2] + 1 << '\n';
                        break;
                    case Elem::ParamSpace::QuadSpace:
                        fout << c2n[0] + 1 << " " << c2n[1] + 1 << " " << c2n[2] + 1 << " " << c2n[3] + 1 << '\n';
                        break;
                    default:
                        assert(false);
                    }
                });
            fout.close();
        }

        void LogStatusSerialPart()
        {
            MPI_Barrier(mpi.comm);
            if (mpi.rank == 0)
                log() << "CompactFacedMeshSerialRW::LogStatus() Synchronized\n"
                      << std::endl;

            if (mpi.rank == 0)
                log() << "CompactFacedMeshSerialRW::cell2node\n";
            if (cell2node)
                cell2node->LogStatus();

            if (mpi.rank == 0)
                log() << "CompactFacedMeshSerialRW::cell2face\n";
            if (cell2face)
                cell2face->LogStatus();

            if (mpi.rank == 0)
                log() << "CompactFacedMeshSerialRW::cellAtr\n";
            if (cellAtr)
                cellAtr->LogStatus();

            if (mpi.rank == 0)
                log() << "CompactFacedMeshSerialRW::face2node\n";
            if (face2node)
                face2node->LogStatus();

            if (mpi.rank == 0)
                log() << "CompactFacedMeshSerialRW::face2cell\n";
            if (face2cell)
                face2cell->LogStatus();

            if (mpi.rank == 0)
                log() << "CompactFacedMeshSerialRW::faceAtr\n";
            if (faceAtr)
                faceAtr->LogStatus();

            if (mpi.rank == 0)
                log() << "CompactFacedMeshSerialRW::nodeCoords\n";
            if (nodeCoords)
                nodeCoords->LogStatus();
        }

        void LogStatusDistPart()
        {
            MPI_Barrier(mpi.comm);
            if (mpi.rank == 0)
                log() << "CompactFacedMeshSerialRW::LogStatus() Synchronized\n"
                      << std::endl;

            if (mpi.rank == 0)
                log() << "CompactFacedMeshSerialRW::cell2nodeDist\n";
            if (cell2nodeDist)
                cell2nodeDist->LogStatus();

            if (mpi.rank == 0)
                log() << "CompactFacedMeshSerialRW::cell2faceDist\n";
            if (cell2face)
                cell2faceDist->LogStatus();

            if (mpi.rank == 0)
                log() << "CompactFacedMeshSerialRW::cellAtrDist\n";
            if (cellAtr)
                cellAtrDist->LogStatus();

            if (mpi.rank == 0)
                log() << "CompactFacedMeshSerialRW::face2nodeDist\n";
            if (face2node)
                face2nodeDist->LogStatus();

            if (mpi.rank == 0)
                log() << "CompactFacedMeshSerialRW::face2cellDist\n";
            if (face2cell)
                face2cellDist->LogStatus();

            if (mpi.rank == 0)
                log() << "CompactFacedMeshSerialRW::faceAtrDist\n";
            if (faceAtr)
                faceAtrDist->LogStatus();

            if (mpi.rank == 0)
                log() << "CompactFacedMeshSerialRW::nodeCoordsDist\n";
            if (nodeCoords)
                nodeCoordsDist->LogStatus();
        }
    };
}