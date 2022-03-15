#pragma once
#include "DNDS_Defines.h"

/**
 * \file DNDS_Elements.hpp
 * \brief some basic routines for 3d element calculation
 * operating dim fixed for 3
 * 
 */

namespace DNDS
{
    namespace Elem
    {
        /// \brief Elem Type complies with Gmsh convention
        /// https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
        /// Node Ordering:
        /**
         *
Line:                 Line3:          Line4:

      v
      ^
      |
      |
0-----+-----1 --> u   0----2----1     0---2---3---1




Triangle:               Triangle6:          Triangle9/10:          Triangle12/15:

v
^                                                                   2
|                                                                   | \
2                       2                    2                      9   8
|`\                     |`\                  | \                    |     \
|  `\                   |  `\                7   6                 10 (14)  7
|    `\                 5    `4              |     \                |         \
|      `\               |      `\            8  (9)  5             11 (12) (13) 6
|        `\             |        `\          |         \            |             \
0----------1 --> u      0-----3----1         0---3---4---1          0---3---4---5---1

F0 = 0,1, F1 = 1,2, F2 = 2,0

Quadrangle:            Quadrangle8:            Quadrangle9:

      v
      ^
      |
3-----------2          3-----6-----2           3-----6-----2
|     |     |          |           |           |           |
|     |     |          |           |           |           |
|     +---- | --> u    7           5           7     8     5
|           |          |           |           |           |
|           |          |           |           |           |
0-----------1          0-----4-----1           0-----4-----1

F0 = 0,1, F1 = 1,2, F2 = 2,3, F3 = 3,0


Tetrahedron:                          Tetrahedron10:

                   v
                 .
               ,/
              /
           2                                     2
         ,/|`\                                 ,/|`\
       ,/  |  `\                             ,/  |  `\
     ,/    '.   `\                         ,6    '.   `5
   ,/       |     `\                     ,/       8     `\
 ,/         |       `\                 ,/         |       `\
0-----------'.--------1 --> u         0--------4--'.--------1
 `\.         |      ,/                 `\.         |      ,/
    `\.      |    ,/                      `\.      |    ,9
       `\.   '. ,/                           `7.   '. ,/
          `\. |/                                `\. |/
             `3                                    `3
                `\.
                   ` w

F0 = 0,2,1, F1 = 0,1,3, F2 = 0,3,2, F3 = 3,1,2




Hexahedron:             Hexahedron20:          Hexahedron27:

       v
3----------2            3----13----2           3----13----2
|\     ^   |\           |\         |\          |\         |\
| \    |   | \          | 15       | 14        |15    24  | 14
|  \   |   |  \         9  \       11 \        9  \ 20    11 \
|   7------+---6        |   7----19+---6       |   7----19+---6
|   |  +-- |-- | -> u   |   |      |   |       |22 |  26  | 23|
0---+---\--1   |        0---+-8----1   |       0---+-8----1   |
 \  |    \  \  |         \  17      \  18       \ 17    25 \  18
  \ |     \  \ |         10 |        12|        10 |  21    12|
   \|      w  \|           \|         \|          \|         \|
    4----------5            4----16----5           4----16----5

F0 = 0,3,2,1, F1 = 0,1,5,4, F2 = 0,4,7,3
F3 = 1,2,6,5, F4 = 2,3,7,6, F5 = 4,5,6,7



Prism:                      Prism15:               Prism18:

           w
           ^
           |
           3                       3                      3
         ,/|`\                   ,/|`\                  ,/|`\
       ,/  |  `\               12  |  13              12  |  13
     ,/    |    `\           ,/    |    `\          ,/    |    `\
    4------+------5         4------14-----5        4------14-----5
    |      |      |         |      8      |        |      8      |
    |    ,/|`\    |         |      |      |        |    ,/|`\    |
    |  ,/  |  `\  |         |      |      |        |  15  |  16  |
    |,/    |    `\|         |      |      |        |,/    |    `\|
   ,|      |      |\        10     |      11       10-----17-----11
 ,/ |      0      | `\      |      0      |        |      0      |
u   |    ,/ `\    |    v    |    ,/ `\    |        |    ,/ `\    |
    |  ,/     `\  |         |  ,6     `7  |        |  ,6     `7  |
    |,/         `\|         |,/         `\|        |,/         `\|
    1-------------2         1------9------2        1------9------2

F0 = 0,2,1
F1 = 0,1,4,3, F2 = 0,3,5,2, F3 = 1,2,5,4
F4 = 3,4,5



Pyramid:                     Pyramid13:                   Pyramid14:

               4                            4                            4
             ,/|\                         ,/|\                         ,/|\
           ,/ .'|\                      ,/ .'|\                      ,/ .'|\
         ,/   | | \                   ,/   | | \                   ,/   | | \
       ,/    .' | `.                ,/    .' | `.                ,/    .' | `.
     ,/      |  '.  \             ,7      |  12  \             ,7      |  12  \
   ,/       .' w |   \          ,/       .'   |   \          ,/       .'   |   \
 ,/         |  ^ |    \       ,/         9    |    11      ,/         9    |    11
0----------.'--|-3    `.     0--------6-.'----3    `.     0--------6-.'----3    `.
 `\        |   |  `\    \      `\        |      `\    \     `\        |      `\    \
   `\     .'   +----`\ - \ -> v  `5     .'        10   \      `5     .' 13     10   \
     `\   |    `\     `\  \        `\   |           `\  \       `\   |           `\  \
       `\.'      `\     `\`          `\.'             `\`         `\.'             `\`
          1----------------2            1--------8-------2           1--------8-------2
                    `\
                       u

F0 = 0,3,2,1
F1 = 0,1,4, F2 = 0,4,3, F3 = 1,2,4, F4 = 2,3,4

         */
        enum ElemType
        {
            UnknownElem = 0,
            Line1 = 1,
            Line2 = 8,

            Tri3 = 2,
            Tri6 = 9,
            Quad4 = 3,
            Quad9 = 10,

            Tet4 = 4,
            Tet10 = 11,
            Hex8 = 5,
            Hex27 = 12,
            Prism6 = 6,
            Prism18 = 13,
            Pyramid5 = 7,
            Pyramid14 = 14
        };

        enum ParamSpace
        {
            UnknownPSpace = 0,
            LineSpace,

            TriSpace,
            QuadSpace,

            TetSpace,
            HexSpace,
            PrismSpace,
            PyramidSpace
        };

        const ParamSpace paramSpaceList[15] = {
            UnknownPSpace, // 0
            LineSpace,     // 1
            TriSpace,      // 2
            QuadSpace,     // 3
            TetSpace,      // 4
            HexSpace,      // 5
            PrismSpace,    // 6
            PyramidSpace,  // 7
            LineSpace,     // 8
            TriSpace,      // 9
            QuadSpace,     // 10
            TetSpace,      // 11
            HexSpace,      // 12
            PrismSpace,    // 13
            PyramidSpace,  // 14
        };

        const ElemType FaceTypeList[15][6] = {
            {UnknownElem},                                  // 0
            {UnknownElem, UnknownElem, UnknownElem},        // 1
            {Line1, Line1, Line1, UnknownElem},             // 2
            {Line1, Line1, Line1, Line1, UnknownElem},      // 3
            {Tri3, Tri3, Tri3, Tri3, UnknownElem},          // 4
            {Quad4, Quad4, Quad4, Quad4, Quad4, Quad4},     // 5
            {Tri3, Quad4, Quad4, Quad4, Tri3, UnknownElem}, // 6
            {Quad4, Tri3, Tri3, Tri3, Tri3, UnknownElem},   // 7
            {UnknownElem, UnknownElem, UnknownElem},        // 8
            {Line2, Line2, Line2, UnknownElem},             // 9
            {Line2, Line2, Line2, Line2, UnknownElem},      // 10
            {Tri6, Tri6, Tri6, Tri6, UnknownElem},          // 11
            {Quad9, Quad9, Quad9, Quad9, Quad9, Quad9},     // 12
            {Tri6, Quad9, Quad9, Quad9, Tri6, UnknownElem}, // 13
            {Quad9, Tri6, Tri6, Tri6, Tri6, UnknownElem},   // 14
        };

        const int FaceNodeList[15][6][9] =
            {
                {{-1}, {-1}, {-1}, {-1}, {-1}, {-1}},
                {{-1}, {-1}, {-1}, {-1}, {-1}, {-1}},                         // 1
                {{0, 1, -1}, {1, 2, -1}, {2, 0, -1}, {-1}, {-1}, {-1}},       // 2
                {{0, 1, -1}, {1, 2, -1}, {2, 3, -1}, {3, 0, -1}, {-1}, {-1}}, // 3
                {{0, 2, 1, -1},
                 {0, 1, 3, -1},
                 {0, 3, 2, -1},
                 {3, 1, 2, -1},
                 {-1},
                 {-1}}, // 4
                {{0, 3, 2, 1, -1},
                 {0, 1, 5, 4, -1},
                 {0, 4, 7, 3, -1},
                 {1, 2, 6, 5, -1},
                 {2, 3, 7, 6, -1},
                 {4, 5, 6, 7, -1}}, // 5
                {{0, 2, 1, -1},
                 {0, 1, 4, 3, -1},
                 {0, 3, 5, 2, -1},
                 {1, 2, 5, 4, -1},
                 {3, 4, 5, -1},
                 {-1, -1}}, // 6
                {{0, 3, 2, 1, -1},
                 {0, 1, 4, -1},
                 {0, 4, 3, -1},
                 {1, 2, 4, -1},
                 {2, 3, 4, -1},
                 {-1, -1}},                                                               // 7
                {{-1}, {-1}, {-1}, {-1}, {-1}, {-1}},                                     // 8
                {{0, 1, 3, -1}, {1, 2, 4, -1}, {2, 0, 5, -1}, {-1}, {-1}, {-1}},          // 9
                {{0, 1, 4, -1}, {1, 2, 5, -1}, {2, 3, 6, -1}, {3, 0, 7, -1}, {-1}, {-1}}, // 10
                {{0, 2, 1, 6, 5, 4 - 1},
                 {0, 1, 3, 4, 9, 7, -1},
                 {0, 3, 2, 7, 8, 6, -1},
                 {3, 1, 2, 9, 5, 8, -1},
                 {-1},
                 {-1}}, // 11
                {{0, 3, 2, 1, 9, 13, 11, 8, 20},
                 {0, 1, 5, 4, 8, 12, 16, 10, 21},
                 {0, 4, 7, 3, 10, 17, 15, 9, 22},
                 {1, 2, 6, 5, 11, 14, 18, 12, 23},
                 {2, 3, 7, 6, 13, 15, 19, 14, 24},
                 {4, 5, 6, 7, 16, 18, 19, 17, 25}}, // 12
                {{0, 2, 1, 7, 9, 6, -1},
                 {0, 1, 4, 3, 6, 10, 12, 8, 15},
                 {0, 3, 5, 2, 8, 13, 11, 7, 16},
                 {1, 2, 5, 4, 9, 11, 14, 10, 17},
                 {3, 4, 5, 12, 14, 13, -1},
                 {-1, -1}}, // 13
                {{0, 3, 2, 1, 6, 10, 8, 5, 13},
                 {0, 1, 4, 5, 9, 7, -1},
                 {0, 4, 3, 7, 12, 6, -1},
                 {1, 2, 4, 8, 11, 9, -1},
                 {2, 3, 4, 10, 12, 11, -1},
                 {-1, -1}}, // 14
        };

        const int DimOrderListNVNNNF[15][5] = {
            {-1, -1, -1, -1, -1}, // 0
            {1, 1, 2, 2, 0},      // 1 Line1
            {2, 1, 3, 3, 3},      // 2 Tri3
            {2, 1, 4, 4, 4},      // 3 Quad4
            {3, 1, 4, 4, 4},      // 4 Tet4
            {3, 1, 8, 8, 6},      // 5 Hex8
            {3, 1, 6, 6, 5},      // 6 Prism6
            {3, 1, 5, 5, 5},      // 7 Pyramid5
            {1, 2, 2, 3, 0},      // 8 Line2
            {2, 2, 3, 6, 3},      // 9 Tri6
            {2, 2, 4, 9, 4},      // 10 Quad9
            {3, 2, 4, 10, 4},     // 11 Tet10
            {3, 2, 8, 27, 6},     // 12 Hex27
            {3, 2, 6, 18, 5},     // 13 Prism18
            {3, 2, 5, 14, 5},     // 14 Pyramid14
        };

        /// Gauss-Legendre integral coords and weights for [-1,1] intervals
        const real GaussLine1[2][1] = {{0}, {2}};

        const real GaussLine2[2][2] = {{-1. / std::sqrt(3), 1. / std::sqrt(3)}, {1, 1}};

        const real GaussLine3[2][3] = {{-0.7745966692414833, 0, 0.7745966692414833}, {5. / 9., 8. / 9., 5. / 9.}};

        const real GaussLine4[2][4] = {{-0.861136311594054, -0.339981043584857, 0.339981043584856, 0.861136311594053},
                                       {0.347854845137452, 0.652145154862546, 0.652145154862546, 0.347854845137453}};

        const real GaussLine5[2][5] = {{-0.906179845938664, -0.538469310105683, 0, 0.538469310105683, 0.906179845938664},
                                       {0.236926885056188, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056190}};

        /// Hammer integral coords for [0,1] [0,1] triangle
        const real HammerTri1[3][1] = {{1. / 3.}, {1. / 3.}, {1. / 2.}};

        const real HammerTri3[3][3] = {
            {2. / 3., 1. / 6., 1. / 6.},
            {1. / 6., 2. / 3., 1. / 6.},
            {1. / 6., 1. / 6., 1. / 6.}};

        const real HammerTri4[3][4] = {
            {1. / 3., 0.6, 0.2, 0.2},
            {1. / 3., 0.2, 0.6, 0.2},
            {-27. / 96., 25. / 96., 25. / 96., 25. / 96.}};

        const real HammerTri7A1 = 0.059715871789770;
        const real HammerTri7B1 = 0.470142064105115;
        const real HammerTri7W1 = 0.132394152788506 * 0.5;
        const real HammerTri7A2 = 0.797426985353087;
        const real HammerTri7B2 = 0.101286507323456;
        const real HammerTri7W2 = 0.125939180544827 * 0.5;
        const real HammerTri7[3][7] = {
            {1. / 3., HammerTri7A1, HammerTri7B1, HammerTri7B1, HammerTri7A2, HammerTri7B2, HammerTri7B2},
            {1. / 3., HammerTri7B1, HammerTri7A1, HammerTri7B1, HammerTri7B2, HammerTri7A2, HammerTri7B2},
            {9. / 80., HammerTri7W1, HammerTri7W1, HammerTri7W1, HammerTri7W2, HammerTri7W2, HammerTri7W2},
        };

        ///\brief A utility class, for organizing element basics or getting integral routine
        class Elem
        {
            ElemType elemType = UnknownElem;
            ParamSpace paramSpace = UnknownPSpace;
            int dim = 0;
            int order = 0;
            int Nvert = 0;
            int Nnode = 0;
            int Nface = 0;

        public:
            inline Elem(ElemType ntype) { setType(ntype); }
            inline ElemType getType() { return elemType; }
            inline ParamSpace getPspace() { return paramSpace; }
            inline int getDim() { return dim; }
            inline int getOrder() { return order; }
            inline int getNFace() { return Nface; }
            inline int getNNode() { return Nnode; }
            inline int getNVert() { return Nvert; }
            inline void setType(ElemType ntype)
            {
                elemType = ntype;
                paramSpace = paramSpaceList[ntype];
                dim = DimOrderListNVNNNF[ntype][0];
                order = DimOrderListNVNNNF[ntype][1];
                Nvert = DimOrderListNVNNNF[ntype][2];
                Nnode = DimOrderListNVNNNF[ntype][3];
                Nface = DimOrderListNVNNNF[ntype][4];
            }
            inline Elem ObtainFace(int iface)
            {
                assert(iface < Nface);
                return Elem(FaceTypeList[elemType][iface]);
            }
            template <class TArray>
            void SubstractFaceNodes(int iface, const Elem &faceElem, const TArray &nodes, TArray &faceNodes)
            {
                assert(iface < Nface);
                for (int i = 0; i < faceElem.Nnode; i++)
                    faceNodes[i] = nodes[FaceNodeList[elemType][iface][i]];
            }

            // TODO: Integration method ("Eigen-Flavored Template")
        };

    }
}