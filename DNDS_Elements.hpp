#pragma once
#include "DNDS_Defines.h"
#include "Eigen/Dense"
#include "EigenTensor.hpp"

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
* Line:                 Line3:          Line4:
*
*       v
*       ^
*       |
*       |
* 0-----+-----1 --> u   0----2----1     0---2---3---1
*
*
*
*
* Triangle:               Triangle6:          Triangle9/10:          Triangle12/15:
*
* v
* ^                                                                   2
* |                                                                   | \
* 2                       2                    2                      9   8
* |`\                     |`\                  | \                    |     \
* |  `\                   |  `\                7   6                 10 (14)  7
* |    `\                 5    `4              |     \                |         \
* |      `\               |      `\            8  (9)  5             11 (12) (13) 6
* |        `\             |        `\          |         \            |             \
* 0----------1 --> u      0-----3----1         0---3---4---1          0---3---4---5---1
*
* F0 = 0,1, F1 = 1,2, F2 = 2,0
*
* Quadrangle:            Quadrangle8:            Quadrangle9:
*
*       v
*       ^
*       |
* 3-----------2          3-----6-----2           3-----6-----2
* |     |     |          |           |           |           |
* |     |     |          |           |           |           |
* |     +---- | --> u    7           5           7     8     5
* |           |          |           |           |           |
* |           |          |           |           |           |
* 0-----------1          0-----4-----1           0-----4-----1
*
* F0 = 0,1, F1 = 1,2, F2 = 2,3, F3 = 3,0
*
*
* Tetrahedron:                          Tetrahedron10:
*
*                    v
*                  .
*                ,/
*               /
*            2                                     2
*          ,/|`\                                 ,/|`\
*        ,/  |  `\                             ,/  |  `\
*      ,/    '.   `\                         ,6    '.   `5
*    ,/       |     `\                     ,/       8     `\
*  ,/         |       `\                 ,/         |       `\
* 0-----------'.--------1 --> u         0--------4--'.--------1
*  `\.         |      ,/                 `\.         |      ,/
*     `\.      |    ,/                      `\.      |    ,9
*        `\.   '. ,/                           `7.   '. ,/
*           `\. |/                                `\. |/
*              `3                                    `3
*                 `\.
*                    ` w
*
* F0 = 0,2,1, F1 = 0,1,3, F2 = 0,3,2, F3 = 3,1,2
*
*
*
*
* Hexahedron:             Hexahedron20:          Hexahedron27:
*
*        v
* 3----------2            3----13----2           3----13----2
* |\     ^   |\           |\         |\          |\         |\
* | \    |   | \          | 15       | 14        |15    24  | 14
* |  \   |   |  \         9  \       11 \        9  \ 20    11 \
* |   7------+---6        |   7----19+---6       |   7----19+---6
* |   |  +-- |-- | -> u   |   |      |   |       |22 |  26  | 23|
* 0---+---\--1   |        0---+-8----1   |       0---+-8----1   |
*  \  |    \  \  |         \  17      \  18       \ 17    25 \  18
*   \ |     \  \ |         10 |        12|        10 |  21    12|
*    \|      w  \|           \|         \|          \|         \|
*     4----------5            4----16----5           4----16----5
*
* F0 = 0,3,2,1, F1 = 0,1,5,4, F2 = 0,4,7,3
* F3 = 1,2,6,5, F4 = 2,3,7,6, F5 = 4,5,6,7
*
*
*
* Prism:                      Prism15:               Prism18:
*
*            w
*            ^
*            |
*            3                       3                      3
*          ,/|`\                   ,/|`\                  ,/|`\
*        ,/  |  `\               12  |  13              12  |  13
*      ,/    |    `\           ,/    |    `\          ,/    |    `\
*     4------+------5         4------14-----5        4------14-----5
*     |      |      |         |      8      |        |      8      |
*     |    ,/|`\    |         |      |      |        |    ,/|`\    |
*     |  ,/  |  `\  |         |      |      |        |  15  |  16  |
*     |,/    |    `\|         |      |      |        |,/    |    `\|
*    ,|      |      |\        10     |      11       10-----17-----11
*  ,/ |      0      | `\      |      0      |        |      0      |
* u   |    ,/ `\    |    v    |    ,/ `\    |        |    ,/ `\    |
*     |  ,/     `\  |         |  ,6     `7  |        |  ,6     `7  |
*     |,/         `\|         |,/         `\|        |,/         `\|
*     1-------------2         1------9------2        1------9------2
*
* F0 = 0,2,1
* F1 = 0,1,4,3, F2 = 0,3,5,2, F3 = 1,2,5,4
* F4 = 3,4,5
*
*
*
* Pyramid:                     Pyramid13:                   Pyramid14:
*
*                4                            4                            4
*              ,/|\                         ,/|\                         ,/|\
*            ,/ .'|\                      ,/ .'|\                      ,/ .'|\
*          ,/   | | \                   ,/   | | \                   ,/   | | \
*        ,/    .' | `.                ,/    .' | `.                ,/    .' | `.
*      ,/      |  '.  \             ,7      |  12  \             ,7      |  12  \
*    ,/       .' w |   \          ,/       .'   |   \          ,/       .'   |   \
*  ,/         |  ^ |    \       ,/         9    |    11      ,/         9    |    11
* 0----------.'--|-3    `.     0--------6-.'----3    `.     0--------6-.'----3    `.
*  `\        |   |  `\    \      `\        |      `\    \     `\        |      `\    \
*    `\     .'   +----`\ - \ -> v  `5     .'        10   \      `5     .' 13     10   \
*      `\   |    `\     `\  \        `\   |           `\  \       `\   |           `\  \
*        `\.'      `\     `\`          `\.'             `\`         `\.'             `\`
*           1----------------2            1--------8-------2           1--------8-------2
*                     `\
*                        u
*
* F0 = 0,3,2,1
* F1 = 0,1,4, F2 = 0,4,3, F3 = 1,2,4, F4 = 2,3,4

         */

        typedef Eigen::Vector3d tPoint;
        typedef Eigen::Matrix3d tJacobi;

        static const int DNDS_ELEM_TYPE_NUM = 15;
        static const int DNDS_ELEM_MAX_FACE_NUM = 6;
        static const int DNDS_ELEM_MAX_FACENODE_NUM = 9;

        static const int DNDS_PARAM_SPACE_NUM = 8;

        enum ElemType
        {
            UnknownElem = 0,
            Line2 = 1,
            Line3 = 8,

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
            Pyramid14 = 14,
            // Point1 = 15,
            // Quad8 = 16,
        };

        enum ParamSpace
        {
            UnknownPSpace = 0,
            LineSpace = 1,

            TriSpace = 2,
            QuadSpace = 3,

            TetSpace = 4,
            HexSpace = 5,
            PrismSpace = 6,
            PyramidSpace = 7
        };

        // TODO: elevate 3d elements
        static const int paramSpaceNumIntScheme[8] = {
            0, 5, 4, 4, 0, 0, 0, 0};

        static const ParamSpace paramSpaceList[DNDS_ELEM_TYPE_NUM] = {
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
        // auto a = sizeof(ParamSpace);

        static const ElemType FaceTypeList[DNDS_ELEM_TYPE_NUM][DNDS_ELEM_MAX_FACE_NUM] = {
            {UnknownElem},                                  // 0
            {UnknownElem, UnknownElem, UnknownElem},        // 1
            {Line2, Line2, Line2, UnknownElem},             // 2
            {Line2, Line2, Line2, Line2, UnknownElem},      // 3
            {Tri3, Tri3, Tri3, Tri3, UnknownElem},          // 4
            {Quad4, Quad4, Quad4, Quad4, Quad4, Quad4},     // 5
            {Tri3, Quad4, Quad4, Quad4, Tri3, UnknownElem}, // 6
            {Quad4, Tri3, Tri3, Tri3, Tri3, UnknownElem},   // 7
            {UnknownElem, UnknownElem, UnknownElem},        // 8
            {Line3, Line3, Line3, UnknownElem},             // 9
            {Line3, Line3, Line3, Line3, UnknownElem},      // 10
            {Tri6, Tri6, Tri6, Tri6, UnknownElem},          // 11
            {Quad9, Quad9, Quad9, Quad9, Quad9, Quad9},     // 12
            {Tri6, Quad9, Quad9, Quad9, Tri6, UnknownElem}, // 13
            {Quad9, Tri6, Tri6, Tri6, Tri6, UnknownElem},   // 14
        };

        static const int FaceNodeList[DNDS_ELEM_TYPE_NUM][DNDS_ELEM_MAX_FACE_NUM][DNDS_ELEM_MAX_FACENODE_NUM] =
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

        static const int DimOrderListNVNNNF[DNDS_ELEM_TYPE_NUM][5] = {
            {-1, -1, -1, -1, -1}, // 0
            {1, 1, 2, 2, 0},      // 1 Line2
            {2, 1, 3, 3, 3},      // 2 Tri3
            {2, 1, 4, 4, 4},      // 3 Quad4
            {3, 1, 4, 4, 4},      // 4 Tet4
            {3, 1, 8, 8, 6},      // 5 Hex8
            {3, 1, 6, 6, 5},      // 6 Prism6
            {3, 1, 5, 5, 5},      // 7 Pyramid5
            {1, 2, 2, 3, 0},      // 8 Line3
            {2, 2, 3, 6, 3},      // 9 Tri6
            {2, 2, 4, 9, 4},      // 10 Quad9
            {3, 2, 4, 10, 4},     // 11 Tet10
            {3, 2, 8, 27, 6},     // 12 Hex27
            {3, 2, 6, 18, 5},     // 13 Prism18
            {3, 2, 5, 14, 5},     // 14 Pyramid14
        };

        /// Gauss-Legendre integral coords and weights for [-1,1] intervals
        static const real GaussLine1[2][1] = {{0}, {2}};

        static const real GaussLine2[2][2] = {{-1. / std::sqrt(3.), 1. / std::sqrt(3.)}, {1, 1}};

        static const real GaussLine3[2][3] = {{-0.7745966692414833, 0, 0.7745966692414833}, {5. / 9., 8. / 9., 5. / 9.}};

        static const real GaussLine4[2][4] = {{-0.861136311594054, -0.339981043584857, 0.339981043584856, 0.861136311594053},
                                              {0.347854845137452, 0.652145154862546, 0.652145154862546, 0.347854845137453}};

        static const real GaussLine5[2][5] = {{-0.906179845938664, -0.538469310105683, 0, 0.538469310105683, 0.906179845938664},
                                              {0.236926885056188, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056190}};

        static const real GaussQuad1[3][1] = {{0}, {0}, {2}};

        static const real GaussQuad4[3][4] = {{-1. / std::sqrt(3.), 1. / std::sqrt(3.), -1. / std::sqrt(3.), 1. / std::sqrt(3.)},
                                              {-1. / std::sqrt(3.), -1. / std::sqrt(3.), 1. / std::sqrt(3.), 1. / std::sqrt(3.)},
                                              {1, 1, 1, 1}};
        static const real GaussQuad9[3][9] = {{-0.7745966692414833, 0, 0.7745966692414833, -0.7745966692414833, 0, 0.7745966692414833, -0.7745966692414833, 0, 0.7745966692414833},
                                              {-0.7745966692414833, -0.7745966692414833, -0.7745966692414833, 0, 0, 0, 0.7745966692414833, 0.7745966692414833, 0.7745966692414833},
                                              {25. / 81., 40. / 81., 25. / 81., 40. / 81., 64. / 81., 40. / 81., 25. / 81., 40. / 81., 25. / 81.}};

        static const real GaussQuad16[3][16] = {{-0.861136311594054, -0.339981043584857, 0.339981043584856, 0.861136311594053,
                                                 -0.861136311594054, -0.339981043584857, 0.339981043584856, 0.861136311594053,
                                                 -0.861136311594054, -0.339981043584857, 0.339981043584856, 0.861136311594053,
                                                 -0.861136311594054, -0.339981043584857, 0.339981043584856, 0.861136311594053},
                                                {-0.861136311594054, -0.861136311594054, -0.861136311594054, -0.861136311594054,
                                                 -0.339981043584857, -0.339981043584857, -0.339981043584857, -0.339981043584857,
                                                 0.339981043584856, 0.339981043584856, 0.339981043584856, 0.339981043584856,
                                                 0.861136311594053, 0.861136311594053, 0.861136311594053, 0.861136311594053},
                                                {0.347854845137452 * 0.347854845137452, 0.652145154862546 * 0.347854845137452, 0.652145154862546 * 0.347854845137452, 0.347854845137453 * 0.347854845137452,
                                                 0.347854845137452 * 0.652145154862546, 0.652145154862546 * 0.652145154862546, 0.652145154862546 * 0.652145154862546, 0.347854845137453 * 0.652145154862546,
                                                 0.347854845137452 * 0.652145154862546, 0.652145154862546 * 0.652145154862546, 0.652145154862546 * 0.652145154862546, 0.347854845137453 * 0.652145154862546,
                                                 0.347854845137452 * 0.347854845137452, 0.652145154862546 * 0.347854845137452, 0.652145154862546 * 0.347854845137452, 0.347854845137453 * 0.347854845137452}};

        /// Hammer integral coords for [0,1] [0,1] triangle
        static const real HammerTri1[3][1] = {{1. / 3.}, {1. / 3.}, {1. / 2.}};

        static const real HammerTri3[3][3] = {
            {2. / 3., 1. / 6., 1. / 6.},
            {1. / 6., 2. / 3., 1. / 6.},
            {1. / 6., 1. / 6., 1. / 6.}};

        static const real HammerTri4[3][4] = {
            {1. / 3., 0.6, 0.2, 0.2},
            {1. / 3., 0.2, 0.6, 0.2},
            {-27. / 96., 25. / 96., 25. / 96., 25. / 96.}};

        static const real HammerTri7A1 = 0.059715871789770;
        static const real HammerTri7B1 = 0.470142064105115;
        static const real HammerTri7W1 = 0.132394152788506 * 0.5;
        static const real HammerTri7A2 = 0.797426985353087;
        static const real HammerTri7B2 = 0.101286507323456;
        static const real HammerTri7W2 = 0.125939180544827 * 0.5;
        static const real HammerTri7[3][7] = {
            {1. / 3., HammerTri7A1, HammerTri7B1, HammerTri7B1, HammerTri7A2, HammerTri7B2, HammerTri7B2},
            {1. / 3., HammerTri7B1, HammerTri7A1, HammerTri7B1, HammerTri7B2, HammerTri7A2, HammerTri7B2},
            {9. / 80., HammerTri7W1, HammerTri7W1, HammerTri7W1, HammerTri7W2, HammerTri7W2, HammerTri7W2},
        };

        typedef int8_t tIntScheme;
        static const tIntScheme INT_SCHEME_LINE_1 = 0;
        static const tIntScheme INT_SCHEME_LINE_2 = 1;
        static const tIntScheme INT_SCHEME_LINE_3 = 2;
        static const tIntScheme INT_SCHEME_LINE_4 = 3;
        static const tIntScheme INT_SCHEME_LINE_5 = 4;

        static const tIntScheme INT_SCHEME_TRI_1 = 0;
        static const tIntScheme INT_SCHEME_TRI_3 = 1;
        static const tIntScheme INT_SCHEME_TRI_4 = 2;
        static const tIntScheme INT_SCHEME_TRI_7 = 3;

        static const tIntScheme INT_SCHEME_QUAD_1 = 0;
        static const tIntScheme INT_SCHEME_QUAD_4 = 1;
        static const tIntScheme INT_SCHEME_QUAD_9 = 2;
        static const tIntScheme INT_SCHEME_QUAD_16 = 3;

        static const int IntSchemeSize[DNDS_PARAM_SPACE_NUM][5] = {
            {},
            {1, 2, 3, 4, 5},
            {1, 3, 4, 7},
            {1, 4, 9, 16},
        };

        static const real *IntSchemeBuffPos[DNDS_PARAM_SPACE_NUM][5] = {
            {nullptr, nullptr, nullptr, nullptr, nullptr},
            {GaussLine1[0], GaussLine2[0], GaussLine3[0], GaussLine4[0], GaussLine5[0]},
            {HammerTri1[0], HammerTri3[0], HammerTri4[0], HammerTri7[0], nullptr},
            {GaussQuad1[0], GaussQuad4[0], GaussQuad9[0], GaussQuad16[0], nullptr},
        };

        // D_i is diff operator, xxx = 000, xxy = yxx = yxy = 001, use this code to ascend-order D_i

        /// including up to 3 orders or diffs
        static const int ndiff = 3;
        static const int ndiffSiz = 20;
        static const int ndiffSiz2D = 10;
        static const int diffOperatorOrderList[ndiffSiz][3] =
            {
                //{diffOrderX_0, diffOrderX_1, diffOrder_X2} // indexPlace, diffSeq //*diff seq is ascending, like ddd/dydxdz -> 012
                {0, 0, 0}, // 00     0
                {1, 0, 0}, // 01 0
                {0, 1, 0}, // 02 1
                {0, 0, 1}, // 03 2
                {2, 0, 0}, // 04 00
                {1, 1, 0}, // 05 01
                {1, 0, 1}, // 06 02
                {0, 2, 0}, // 07 11
                {0, 1, 1}, // 08 12
                {0, 0, 2}, // 09 22
                {3, 0, 0}, // 10 000
                {2, 1, 0}, // 11 001
                {2, 0, 1}, // 12 002
                {1, 2, 0}, // 13 011
                {1, 1, 1}, // 14 012
                {1, 0, 2}, // 15 022
                {0, 3, 0}, // 16 111
                {0, 2, 1}, // 17 112
                {0, 1, 2}, // 18 122
                {0, 0, 3}, // 19 222
        };
        static const int diffOperatorOrderList2D[ndiffSiz2D][3] = {
            {0, 0, 0}, // 00 00
            {1, 0, 0}, // 01 01 0
            {0, 1, 0}, // 02 02 1
            {2, 0, 0}, // 03 04 00
            {1, 1, 0}, // 04 05 01
            {0, 2, 0}, // 05 07 11
            {3, 0, 0}, // 06 10 000
            {2, 1, 0}, // 07 11 001
            {1, 2, 0}, // 08 13 011
            {0, 3, 0}, // 09 16 111
        };

        constexpr inline int diffOperatorOrder2Plc(int d0, int d1, int d2)
        {
            int b = 1;
            int number = 0; // calculates number as diffSeq's 3-based integer (each digit added by 1, 012 == 1*3^2 + 2*3^1 + 3*3^0)
            for (int i = 0; i < d2; i++)
            {
                number += 3 * b;
                b *= 3;
            }
            for (int i = 0; i < d1; i++)
            {
                number += 2 * b;
                b *= 3;
            }
            for (int i = 0; i < d0; i++)
            {
                number += 1 * b;
                b *= 3;
            }
            switch (number)
            {
            case 0:
            case 1:
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
                return number;
            case 8:
                return 7;
            case 9:
                return 8;
            case 12:
                return 9;
            case 13:
                return 10;
            case 14:
                return 11;
            case 15:
                return 12;
            case 17:
                return 13;
            case 18:
                return 14;
            case 21:
                return 15;
            case 26:
                return 16;
            case 27:
                return 17;
            case 30:
                return 18;
            case 39:
                return 19;
            default:
                assert(false);
                return -1;
                break;
            }
        }

#define DNDS_DIFF2Dto3DMAP {0, 1, 2, 4, 5, 7, 10, 11, 13, 16}
        static const int diff2Dto3Dmap[ndiffSiz2D] = {0, 1, 2, 4, 5, 7, 10, 11, 13, 16};
        static const int diff3Dto2Dmap[ndiffSiz] = {0, 1, 2, -0xfffffff, 3, 4, -0xfffffff, 5, -0xfffffff, -0xfffffff, 6, 7, -0xfffffff, 8, -0xfffffff, -0xfffffff, 9, -0xfffffff, -0xfffffff, -0xfffffff};

        static const int factorials[ndiff * 3 + 1] = {
            1,
            1,
            1 * 2,
            1 * 2 * 3,
            1 * 2 * 3 * 4,
            1 * 2 * 3 * 4 * 5,
            1 * 2 * 3 * 4 * 5 * 6,
            1 * 2 * 3 * 4 * 5 * 6 * 7,
            1 * 2 * 3 * 4 * 5 * 6 * 7 * 8,
            1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9,
        };
        static const int diffNCombs2D[ndiffSiz2D]{
            1, 1, 1, 1, 2, 1, 1, 3, 3, 1};

        static const int dFactorials[ndiff + 1][ndiff + 1] = {
            {1, 0, 0, 0},
            {1, 1, 0, 0},
            {1, 2, 2, 0},
            {1, 3, 6, 6}};

        inline real iPow(int p, real x)
        {
            switch (p)
            {
            case 0:
                return 1.;
            case 1:
                return x;
            case 2:
                return x * x;
            case 3:
                return x * x * x;
            default:
                return 1e300;
                break;
            }
        }

        static const int diffOperatorOrderRange[ndiff + 2] = {
            0, 1, 4, 10, 20};

        static const int diffOperatorOrderRange2D[ndiff + 2] = {
            0, 1, 3, 6, 10};

        inline int getDiffOrderFromDiffSize(int diffSize)
        {
            for (int i = 0; i <= ndiff; i++)
                if (diffOperatorOrderRange[i + 1] == diffSize)
                    return i;
            log() << "Error === getDiffOrderFromDiffSize got diffs size not good\n"
                  << std::endl;
            assert(false);
            return -1;
        }

        typedef Eigen::Matrix<real, -1, -1, Eigen::RowMajor> tDiFj;
        static const real sqrt3 = std::sqrt(3.);
        static const real sqrt2 = std::sqrt(2.);
        static const real sqrt6 = std::sqrt(6.);

        ///\brief A utility class, for organizing element basics or getting integral routine
        class ElementManager
        {
            /// \brief NBuffer[elemType][iface+1 or 0 for volume][iInt][igaussPoint] = D_i(N_j)
            /// basically GetDiNj()'s buffer, for GetDiNj() only needs to buffer globally

            ElemType elemType = UnknownElem;
            ParamSpace paramSpace = UnknownPSpace;
            int dim = 0;
            int order = 0;
            int Nvert = 0;
            int Nnode = 0;
            int Nface = 0;
            tIntScheme iInt = -1;
            int nIntPoint = -1;
            const real *bufInt = nullptr;

        public:
            static std::vector<std::vector<std::vector<tDiFj>>> NBuffer[DNDS_ELEM_TYPE_NUM];
            static bool NBufferInit;
            static inline void InitNBuffer() // TODO: main check
            {
                for (int elemType = 0; elemType < DNDS_ELEM_TYPE_NUM; elemType++)
                {
                    // int dim = DimOrderListNVNNNF[elemType][0];
                    // int order = DimOrderListNVNNNF[elemType][1];
                    // int Nvert = DimOrderListNVNNNF[elemType][2];
                    int Nnode = DimOrderListNVNNNF[elemType][3];
                    int Nface = DimOrderListNVNNNF[elemType][4];
                    auto paramSpace = paramSpaceList[elemType];
                    if (Nface < 0 || paramSpaceNumIntScheme[paramSpace] <= 0) // latter condition means the element is not implemented with gauss int
                        continue;
                    NBuffer[elemType].resize(Nface + 1);

                    // at vol gauss points
                    NBuffer[elemType][0].resize(paramSpaceNumIntScheme[paramSpace]);
                    for (tIntScheme iInt = 0; iInt < paramSpaceNumIntScheme[paramSpace]; iInt++)
                    {
                        auto e = ElementManager(ElemType(elemType), iInt);
                        NBuffer[elemType][0][iInt].resize(e.nIntPoint);

                        for (int ig = 0; ig < e.nIntPoint; ig++)
                        {
                            tPoint p;
                            e.GetIntPoint(ig, p);
                            NBuffer[elemType][0][iInt][ig].resize(ndiffSiz, Nnode);
                            e.GetDiNj(p, NBuffer[elemType][0][iInt][ig]);
                        }
                    }

                    // at face gauss points
                    for (int iface = 0; iface < Nface; iface++)
                    {
                        auto e = ElementManager(ElemType(elemType), 0);
                        auto ef = e.ObtainFace(iface, 0);
                        NBuffer[elemType][iface + 1].resize(paramSpaceNumIntScheme[ef.paramSpace]);
                        for (tIntScheme iInt = 0; iInt < paramSpaceNumIntScheme[ef.paramSpace]; iInt++)
                        {
                            auto eff = e.ObtainFace(iface, iInt);
                            NBuffer[elemType][iface + 1][iInt].resize(eff.nIntPoint);
                            for (int ig = 0; ig < eff.nIntPoint; ig++)
                            {
                                tPoint p;
                                eff.GetIntPoint(ig, p);
                                tPoint pc;
                                NBuffer[elemType][iface + 1][iInt][ig].resize(ndiffSiz, e.Nnode);
                                e.FaceSpace2VolSpace(iface, p, pc);
                                e.GetDiNj(pc, NBuffer[elemType][iface + 1][iInt][ig]);
                                // std::cout << "ELEMTYPE:" << eff.elemType << " " << e.elemType << " Face:" << iface << std::endl;
                                // std::cout << p.transpose() << std::endl;
                                // std::cout << pc.transpose() << std::endl;
                            }
                        }
                    }
                }
                NBufferInit = true;
            }

            inline ElementManager(ElemType ntype, tIntScheme NIntSchemeIndex) { setType(ntype, NIntSchemeIndex); }
            inline ElemType getType() { return elemType; }
            inline ParamSpace getPspace() { return paramSpace; }
            inline int getDim() { return dim; }
            inline int getOrder() { return order; }
            inline int getNFace() { return Nface; }
            inline int getNNode() { return Nnode; }
            inline int getNVert() { return Nvert; }
            inline tIntScheme getIIntScheme() { return iInt; }
            inline int getNInt() { return nIntPoint; }

            inline void setType(ElemType ntype, tIntScheme NIntSchemeIndex)
            {
                elemType = ntype;
                dim = DimOrderListNVNNNF[ntype][0];
                order = DimOrderListNVNNNF[ntype][1];
                Nvert = DimOrderListNVNNNF[ntype][2];
                Nnode = DimOrderListNVNNNF[ntype][3];
                Nface = DimOrderListNVNNNF[ntype][4];
                paramSpace = paramSpaceList[ntype];

                if (NIntSchemeIndex < 0)
                    NIntSchemeIndex = 0; // <- means doesn't matter which//! or could refuse to initialize int scheme
                iInt = NIntSchemeIndex;
                nIntPoint = IntSchemeSize[paramSpace][iInt];
                bufInt = IntSchemeBuffPos[paramSpace][iInt];
                // assert(nIntPoint > 0 && iInt < 5 && iInt >= 0 && bufInt && bufInt[dim * nIntPoint - 1] > 0. && bufInt[dim * nIntPoint - 1] <= 4.);
            }
            inline tPoint getCenterPParam()
            {
                switch (paramSpace)
                {
                case ParamSpace::LineSpace:
                case ParamSpace::HexSpace:
                case ParamSpace::QuadSpace:
                    return tPoint{0, 0, 0};
                case ParamSpace::TriSpace:
                    return tPoint{1. / 3., 1. / 3., 0};
                case ParamSpace::TetSpace:
                    return tPoint{0.25, 0.25, 0.25};
                case ParamSpace::PyramidSpace:
                    return tPoint{0, 0, 0.25};
                case ParamSpace::PrismSpace:
                    return tPoint{1. / 3., 1. / 3., 0};
                default:
                    assert(false);
                    return tPoint{1e100, 1e100, 1e100};
                }
            }
            inline ElementManager ObtainFace(int iface, tIntScheme faceiInt)
            {
                assert(iface < Nface);
                return ElementManager(FaceTypeList[elemType][iface], faceiInt);
            }

            // returns param coord of int point
            inline void GetIntPoint(int iGpoint, tPoint &p)
            {
                assert(iGpoint < nIntPoint);
                int d;
                for (d = 0; d < dim; d++)
                    p[d] = bufInt[d * nIntPoint + iGpoint];
                for (; d < 3; d++) // magical 3 is because the physical world mostly involves 3 spacial dims
                    p[d] = 0.;
            }

            /// \brief convert point in certain STD face of the element to element space
            template <class TP>
            void FaceSpace2VolSpace(int iface, const TP &pface, TP &pvol, bool invert = false)
            {
                assert(iface >= 0 && iface < Nface);
                // // TODO: implement now
                switch (paramSpace)
                {
                case ParamSpace::TriSpace:
                    if (!invert)
                        switch (iface)
                        {
                        case 0:
                            pvol[0] = pface[0], pvol[1] = pvol[2] = 0.;
                            break;
                        case 1:
                            pvol[0] = 1 - pface[0], pvol[1] = pface[0], pvol[2] = 0.;
                            break;
                        case 2:
                            pvol[0] = 0, pvol[1] = 1 - pface[0], pvol[2] = 0.;
                            break;
                        default:
                            assert(false);
                            break;
                        }
                    else
                        switch (iface)
                        {
                        case 0:
                            pvol[0] = 1 - pface[0], pvol[1] = pvol[2] = 0.;
                            break;
                        case 1:
                            pvol[0] = pface[0], pvol[1] = 1 - pface[0], pvol[2] = 0.;
                            break;
                        case 2:
                            pvol[0] = 0, pvol[1] = pface[0], pvol[2] = 0.;
                            break;
                        default:
                            assert(false);
                            break;
                        }
                    break;

                case ParamSpace::QuadSpace:
                    if (!invert)
                        switch (iface)
                        {
                        case 0:
                            pvol[0] = pface[0], pvol[1] = -1, pvol[2] = 0.;
                            break;
                        case 1:
                            pvol[0] = 1, pvol[1] = pface[0], pvol[2] = 0.;
                            break;
                        case 2:
                            pvol[0] = -pface[0], pvol[1] = 1, pvol[2] = 0.;
                            break;
                        case 3:
                            pvol[0] = -1, pvol[1] = -pface[0], pvol[2] = 0.;
                            break;
                        default:
                            assert(false);
                            break;
                        }
                    else
                        switch (iface)
                        {
                        case 0:
                            pvol[0] = -pface[0], pvol[1] = -1, pvol[2] = 0.;
                            break;
                        case 1:
                            pvol[0] = 1, pvol[1] = -pface[0], pvol[2] = 0.;
                            break;
                        case 2:
                            pvol[0] = pface[0], pvol[1] = 1, pvol[2] = 0.;
                            break;
                        case 3:
                            pvol[0] = -1, pvol[1] = pface[0], pvol[2] = 0.;
                            break;
                        default:
                            assert(false);
                            break;
                        }
                    break;
                default:
                    log() << "FaceSpace2VolSpace: space not supported!\n"
                          << std::endl;
                    assert(false);
                    break;
                }
            }

            /// \brief convert point in certain face of the element to element space, as face is not standard need extra conversion
            template <class TP, class TArray, class TArray2>
            void FaceSpace2VolSpace(int iface, const TP &pface, TP &pvol, const TArray &faceNodes, const TArray2 &faceSTDNodes)
            {
                assert(iface >= 0 && iface < Nface);
                // // TODO: implement now
                switch (paramSpace)
                {
                case ParamSpace::TriSpace:
                    assert((faceNodes[0] == faceSTDNodes[0] && faceNodes[1] == faceSTDNodes[1]) ||
                           (faceNodes[0] == faceSTDNodes[1] && faceNodes[1] == faceSTDNodes[0]));
                    if (faceNodes[0] == faceSTDNodes[0] && faceNodes[1] == faceSTDNodes[1])
                        switch (iface)
                        {
                        case 0:
                            pvol[0] = pface[0], pvol[1] = pvol[2] = 0.;
                            break;
                        case 1:
                            pvol[0] = 1 - pface[0], pvol[1] = pface[0], pvol[2] = 0.;
                            break;
                        case 2:
                            pvol[0] = 0, pvol[1] = 1 - pface[0], pvol[2] = 0.;
                            break;
                        default:
                            assert(false);
                            break;
                        }
                    else
                        switch (iface)
                        {
                        case 0:
                            pvol[0] = 1 - pface[0], pvol[1] = pvol[2] = 0.;
                            break;
                        case 1:
                            pvol[0] = pface[0], pvol[1] = 1 - pface[0], pvol[2] = 0.;
                            break;
                        case 2:
                            pvol[0] = 0, pvol[1] = pface[0], pvol[2] = 0.;
                            break;
                        default:
                            assert(false);
                            break;
                        }
                    break;

                case ParamSpace::QuadSpace:
                    assert((faceNodes[0] == faceSTDNodes[0] && faceNodes[1] == faceSTDNodes[1]) ||
                           (faceNodes[0] == faceSTDNodes[1] && faceNodes[1] == faceSTDNodes[0]));
                    if (faceNodes[0] == faceSTDNodes[0])
                        switch (iface)
                        {
                        case 0:
                            pvol[0] = pface[0], pvol[1] = -1, pvol[2] = 0.;
                            break;
                        case 1:
                            pvol[0] = 1, pvol[1] = pface[0], pvol[2] = 0.;
                            break;
                        case 2:
                            pvol[0] = -pface[0], pvol[1] = 1, pvol[2] = 0.;
                            break;
                        case 3:
                            pvol[0] = -1, pvol[1] = -pface[0], pvol[2] = 0.;
                            break;
                        default:
                            assert(false);
                            break;
                        }
                    else
                        switch (iface)
                        {
                        case 0:
                            pvol[0] = -pface[0], pvol[1] = -1, pvol[2] = 0.;
                            break;
                        case 1:
                            pvol[0] = 1, pvol[1] = -pface[0], pvol[2] = 0.;
                            break;
                        case 2:
                            pvol[0] = pface[0], pvol[1] = 1, pvol[2] = 0.;
                            break;
                        case 3:
                            pvol[0] = -1, pvol[1] = pface[0], pvol[2] = 0.;
                            break;
                        default:
                            assert(false);
                            break;
                        }
                    break;
                default:
                    log() << "FaceSpace2VolSpace: space not supported!\n"
                          << std::endl;
                    assert(false);
                    break;
                }
            }

            /// \brief guassMAP[igaussPointInFace] = iguassPointInFaceSTD
            /// assumes faceNodes and faceSTDNodes represent same-place elements
            template <class TArray, class TArrayG>
            void FaceIGauss2STDMap(int iface, ElementManager &ef, const TArray &faceNodes, const TArray &faceSTDNodes, TArrayG &gaussMAP, bool invert = false)
            {
                assert(iface < Nface && iface >= 0);
                assert(faceNodes.size() >= ef.getNVert() && faceSTDNodes.size() >= ef.getNVert() && gaussMAP.size() >= ef.nIntPoint);
                switch (elemType)
                {
                case ElemType::Tri3:
                case ElemType::Tri6:
                case ElemType::Quad4:
                case ElemType::Quad9: // as planar faces always have line faces, cases are simple
                    assert((faceNodes[0] == faceSTDNodes[0] && faceNodes[1] == faceSTDNodes[1]) ||
                           (faceNodes[0] == faceSTDNodes[1] && faceNodes[1] == faceSTDNodes[0]));
                    if (faceNodes[0] == faceSTDNodes[0])
                        for (int i = 0; i < ef.nIntPoint; i++)
                            gaussMAP[i] = invert ? ef.nIntPoint - 1 - i : i;
                    else
                        for (int i = 0; i < ef.nIntPoint; i++)
                            gaussMAP[i] = (!invert) ? ef.nIntPoint - 1 - i : i;
                    break;
                default:
                    assert(false);
                    break;
                }
            }

            /** \brief returns faceSTD in  faceNodes, should be norming out
             *
             * \param faceNodes returned value, face nodes
             * \param nodes my nodes
             * \param faceElem face elem returned by ObtainFace()
             * */
            template <class TArray, class TArray2>
            void SubstractFaceNodes(int iface, const ElementManager &faceElem, const TArray &nodes, TArray2 &faceNodes)
            {
                assert(iface < Nface && iface >= 0);
                for (int i = 0; i < faceElem.Nnode; i++)
                    faceNodes[i] = nodes[FaceNodeList[elemType][iface][i]];
            }

            /**
             * \brief shape function evaluator
             * this does not require concrete coords, but GetDiPhiJ could require
             * \param p param coord of point
             * \param DiNj = diff_i(N_j),  DiNj.rows() is also a input note that when j >= Nnode, the value
             * returned is not filled
             * */
            template <class TPoint, class TDiFj> // TPoint remains general now
            void GetDiNj(const TPoint &p, TDiFj &DiNj)
            {
                int diffOrder = getDiffOrderFromDiffSize(DiNj.rows());
                assert(DiNj.cols() >= Nnode);
                auto x = p[0], y = p[1]; //, z = p[2]; // param space
                switch (elemType)
                {
                case ElemType::Line2:
                    switch (diffOrder)
                    {
                    case 3:
                        for (int i = 0; i < 10; i++)
                            DiNj(10 + i, 0) = DiNj(10 + i, 1) = 0.0;
                    case 2:
                        for (int i = 0; i < 6; i++)
                            DiNj(4 + i, 0) = DiNj(4 + i, 1) = 0.0;
                    case 1:
                        DiNj(1, 0) = (-1) * 0.5;
                        DiNj(1, 1) = (1) * 0.5;
                        DiNj(2, 0) = 0;
                        DiNj(2, 1) = 0;
                        DiNj(3, 0) = 0;
                        DiNj(3, 1) = 0;
                    case 0:
                        DiNj(0, 0) = (1 - x) * 0.5;
                        DiNj(0, 1) = (1 + x) * 0.5;
                        break;
                    default:
                        assert(false);
                    }
                    break;

                case ElemType::Tri3:
                    switch (diffOrder)
                    {
                    case 3:
                        for (int i = 0; i < 10; i++)
                            DiNj(10 + i, 0) = DiNj(10 + i, 1) = DiNj(10 + i, 2) = 0.0;
                    case 2:
                        for (int i = 0; i < 6; i++)
                            DiNj(4 + i, 0) = DiNj(4 + i, 1) = DiNj(4 + i, 2) = 0.0;
                    case 1:
                        DiNj(1, 0) = -1;
                        DiNj(1, 1) = 1;
                        DiNj(1, 2) = 0;
                        DiNj(2, 0) = -1;
                        DiNj(2, 1) = 0;
                        DiNj(2, 2) = 1;
                        DiNj(3, 0) = 0.;
                        DiNj(3, 1) = 0.;
                        DiNj(3, 2) = 0.;
                    case 0:
                        DiNj(0, 0) = 1 - x - y;
                        DiNj(0, 1) = x;
                        DiNj(0, 2) = y;
                        break;
                    default:
                        assert(false);
                    }
                    break;

                case ElemType::Quad4:
                    switch (diffOrder)
                    {
                    case 3:
                        for (int i = 0; i < 10; i++)
                            DiNj(10 + i, 0) = DiNj(10 + i, 1) = DiNj(10 + i, 2) = DiNj(10 + i, 3) = 0.0;
                    case 2:
                        for (int i = 0; i < 6; i++)
                            DiNj(4 + i, 0) = DiNj(4 + i, 1) = DiNj(4 + i, 2) = DiNj(4 + i, 3) = 0.0;
                        DiNj(5, 0) = (-1) * (-1) * 0.25;
                        DiNj(5, 1) = (1) * (-1) * 0.25;
                        DiNj(5, 2) = (1) * (1) * 0.25;
                        DiNj(5, 3) = (-1) * (1) * 0.25;
                    case 1:
                        DiNj(1, 0) = (-1) * (1 - y) * 0.25;
                        DiNj(1, 1) = (1) * (1 - y) * 0.25;
                        DiNj(1, 2) = (1) * (1 + y) * 0.25;
                        DiNj(1, 3) = (-1) * (1 + y) * 0.25;
                        DiNj(2, 0) = (1 - x) * (-1) * 0.25;
                        DiNj(2, 1) = (1 + x) * (-1) * 0.25;
                        DiNj(2, 2) = (1 + x) * (1) * 0.25;
                        DiNj(2, 3) = (1 - x) * (1) * 0.25;
                        DiNj(3, 0) = 0;
                        DiNj(3, 1) = 0;
                        DiNj(3, 2) = 0;
                        DiNj(3, 3) = 0;
                    case 0:
                        DiNj(0, 0) = (1 - x) * (1 - y) * 0.25;
                        DiNj(0, 1) = (1 + x) * (1 - y) * 0.25;
                        DiNj(0, 2) = (1 + x) * (1 + y) * 0.25;
                        DiNj(0, 3) = (1 - x) * (1 + y) * 0.25;
                        break;
                    default:
                        assert(false);
                    }
                    break;

                case ElemType::Line3:
                    switch (diffOrder)
                    {
                    case 3:
                        for (int i = 0; i < 10; i++)
                            DiNj(10 + i, 0) = DiNj(10 + i, 1) = DiNj(10 + i, 2) = 0.0;
                    case 2:
                        for (int i = 0; i < 6; i++)
                            DiNj(4 + i, 0) = DiNj(4 + i, 1) = DiNj(4 + i, 2) = 0.0;
                        DiNj(4, 0) = ((-1) * (-1) + (-1) * (-1)) * 0.5;
                        DiNj(4, 1) = ((1) * (1) + (1) * (1)) * 0.5;
                        DiNj(4, 2) = ((-1) * (1) + (-1) * (1)) * 1.0;
                    case 1:
                        DiNj(1, 0) = ((-1) * (0 - x) + (1 - x) * (-1)) * 0.5;
                        DiNj(1, 1) = ((1) * (0 + x) + (1 + x) * (1)) * 0.5;
                        DiNj(1, 2) = ((-1) * (1 + x) + (1 - x) * (1)) * 1.0;
                        DiNj(2, 0) = 0;
                        DiNj(2, 1) = 0;
                        DiNj(2, 2) = 0;
                        DiNj(3, 0) = 0;
                        DiNj(3, 1) = 0;
                        DiNj(3, 2) = 0;
                    case 0:
                        DiNj(0, 0) = (1 - x) * (0 - x) * 0.5;
                        DiNj(0, 1) = (1 + x) * (0 + x) * 0.5;
                        DiNj(0, 2) = (1 - x) * (1 + x) * 1.0;
                        break;
                    default:
                        assert(false);
                    }
                    break;

                case ElemType::Tri6:
                    switch (diffOrder)
                    {
                    case 3:
                        for (int i = 0; i < 10; i++)
                            DiNj(10 + i, 0) = DiNj(10 + i, 1) = DiNj(10 + i, 2) = DiNj(10 + i, 3) = DiNj(10 + i, 4) = DiNj(10 + i, 5) = 0.0;
                    case 2:
                        for (int i = 0; i < 6; i++)
                            DiNj(4 + i, 0) = DiNj(4 + i, 1) = DiNj(4 + i, 2) = DiNj(4 + i, 3) = DiNj(4 + i, 4) = DiNj(4 + i, 5) = 0.0;
                        DiNj(4, 0) = 2 * 1 + 2 * 1;
                        DiNj(4, 1) = 1 * 2 + 1 * 2;
                        DiNj(4, 2) = 0;
                        DiNj(4, 3) = -4 * 1 - 4 * 1;
                        DiNj(4, 4) = 0;
                        DiNj(4, 5) = 0;
                        DiNj(5, 0) = 2 * 1 + 2 * 1;
                        DiNj(5, 1) = 0;
                        DiNj(5, 2) = 0;
                        DiNj(5, 3) = -4 * 1;
                        DiNj(5, 4) = 4;
                        DiNj(5, 5) = -4;
                        DiNj(7, 0) = 2 * 1 + 2 * 1;
                        DiNj(7, 1) = 0;
                        DiNj(7, 2) = 1 * 2 + 1 * 2;
                        DiNj(7, 3) = 0;
                        DiNj(7, 4) = 0;
                        DiNj(7, 5) = -4 * 1 - 4 * 1;
                    case 1:
                        DiNj(1, 0) = 2 * (-1 + y + x) + (-1 + 2 * y + 2 * x) * 1;
                        DiNj(1, 1) = 1 * (-1 + 2 * x) + x * 2;
                        DiNj(1, 2) = 0;
                        DiNj(1, 3) = -4 * (-1 + y + x) - 4 * x * 1;
                        DiNj(1, 4) = 4 * y;
                        DiNj(1, 5) = -4 * y * 1;
                        DiNj(2, 0) = 2 * (-1 + y + x) + (-1 + 2 * y + 2 * x) * 1;
                        DiNj(2, 1) = 0;
                        DiNj(2, 2) = 1 * (-1 + 2 * y) + y * 2;
                        DiNj(2, 3) = -4 * x * 1;
                        DiNj(2, 4) = 4 * x;
                        DiNj(2, 5) = -4 * 1 * (-1 + y + x) - 4 * y * 1;
                        DiNj(3, 0) = 0;
                        DiNj(3, 1) = 0;
                        DiNj(3, 2) = 0;
                        DiNj(3, 3) = 0;
                        DiNj(3, 4) = 0;
                        DiNj(3, 5) = 0;
                    case 0:
                        DiNj(0, 0) = (-1 + 2 * y + 2 * x) * (-1 + y + x);
                        DiNj(0, 1) = x * (-1 + 2 * x);
                        DiNj(0, 2) = y * (-1 + 2 * y);
                        DiNj(0, 3) = -4 * x * (-1 + y + x);
                        DiNj(0, 4) = 4 * x * y;
                        DiNj(0, 5) = -4 * y * (-1 + y + x);
                        break;
                    default:
                        assert(false);
                    }
                    break;

                case ElemType::Quad9:
                    switch (diffOrder)
                    {
                    case 3:
                        for (int i = 0; i < 10; i++)
                            DiNj(10 + i, 0) = DiNj(10 + i, 1) = DiNj(10 + i, 2) =
                                DiNj(10 + i, 3) = DiNj(10 + i, 4) = DiNj(10 + i, 5) =
                                    DiNj(10 + i, 6) = DiNj(10 + i, 7) = DiNj(10 + i, 8) = 0.0;
                        DiNj(11, 0) = ((-1) * (-1) + (-1) * (-1)) * ((-1) * (1 - y) + (0 - y) * (-1)) * 0.25;
                        DiNj(11, 1) = ((1) * (1) + (1) * (1)) * ((-1) * (1 - y) + (0 - y) * (-1)) * 0.25;
                        DiNj(11, 2) = ((1) * (1) + (1) * (1)) * ((1) * (1 + y) + (0 + y) * (1)) * 0.25;
                        DiNj(11, 3) = ((-1) * (-1) + (-1) * (-1)) * ((1) * (1)) * 0.25;
                        DiNj(11, 4) = ((-1) * (1) + (-1) * (1)) * ((-1) * (1 - y) + (0 - y) * (-1)) * 0.50;
                        DiNj(11, 5) = ((1) * (1) + (1) * (1)) * ((-1) * (1 + y) + (1 - y) * (1)) * 0.50;
                        DiNj(11, 6) = ((-1) * (1) + (-1) * (1)) * ((1) * (1 + y) + (0 + y) * (1)) * 0.50;
                        DiNj(11, 7) = ((-1) * (-1) + (-1) * (-1)) * ((-1) * (1 + y) + (1 - y) * (1)) * 0.50;
                        DiNj(11, 8) = ((-1) * (1) + (-1) * (1)) * ((-1) * (1 + y) + (1 - y) * (1)) * 1.00;

                        DiNj(13, 0) = ((-1) * (1 - x) + (0 - x) * (-1)) * ((-1) * (-1) + (-1) * (-1)) * 0.25;
                        DiNj(13, 1) = ((1) * (1 + x) + (0 + x) * (1)) * ((-1) * (-1) + (-1) * (-1)) * 0.25;
                        DiNj(13, 2) = ((1) * (1 + x) + (0 + x) * (1)) * ((1) * (1) + (1) * (1)) * 0.25;
                        DiNj(13, 3) = ((-1) * (1 - x) + (0 - x) * (-1)) * ((1) * (1) + (1) * (1)) * 0.25;
                        DiNj(13, 4) = ((-1) * (1 + x) + (1 - x) * (1)) * ((-1) * (-1) + (-1) * (-1)) * 0.50;
                        DiNj(13, 5) = ((1) * (1 + x) + (0 + x) * (1)) * ((-1) * (1) + (-1) * (1)) * 0.50;
                        DiNj(13, 6) = ((-1) * (1 + x) + (1 - x) * (1)) * ((1) * (1) + (1) * (1)) * 0.50;
                        DiNj(13, 7) = ((-1) * (1 - x) + (0 - x) * (-1)) * ((-1) * (1) + (-1) * (1)) * 0.50;
                        DiNj(13, 8) = ((-1) * (1 + x) + (1 - x) * (1)) * ((-1) * (1) + (-1) * (1)) * 1.00;
                    case 2:
                        for (int i = 0; i < 6; i++)
                            DiNj(4 + i, 0) = DiNj(4 + i, 1) = DiNj(4 + i, 2) =
                                DiNj(4 + i, 3) = DiNj(4 + i, 4) = DiNj(4 + i, 5) =
                                    DiNj(4 + i, 6) = DiNj(4 + i, 7) = DiNj(4 + i, 8) = 0.0;
                        DiNj(4, 0) = ((-1) * (-1) + (-1) * (-1)) * ((0 - y) * (1 - y)) * 0.25;
                        DiNj(4, 1) = ((1) * (1) + (1) * (1)) * ((0 - y) * (1 - y)) * 0.25;
                        DiNj(4, 2) = ((1) * (1) + (1) * (1)) * ((0 + y) * (1 + y)) * 0.25;
                        DiNj(4, 3) = ((-1) * (-1) + (-1) * (-1)) * ((0 + y) * (1 + y)) * 0.25;
                        DiNj(4, 4) = ((-1) * (1) + (-1) * (1)) * ((0 - y) * (1 - y)) * 0.50;
                        DiNj(4, 5) = ((1) * (1) + (1) * (1)) * ((1 - y) * (1 + y)) * 0.50;
                        DiNj(4, 6) = ((-1) * (1) + (-1) * (1)) * ((0 + y) * (1 + y)) * 0.50;
                        DiNj(4, 7) = ((-1) * (-1) + (-1) * (-1)) * ((1 - y) * (1 + y)) * 0.50;
                        DiNj(4, 8) = ((-1) * (1) + (-1) * (1)) * ((1 - y) * (1 + y)) * 1.00;
                        DiNj(5, 0) = ((-1) * (1 - x) + (0 - x) * (-1)) * ((-1) * (1 - y) + (0 - y) * (-1)) * 0.25;
                        DiNj(5, 1) = ((1) * (1 + x) + (0 + x) * (1)) * ((-1) * (1 - y) + (0 - y) * (-1)) * 0.25;
                        DiNj(5, 2) = ((1) * (1 + x) + (0 + x) * (1)) * ((1) * (1 + y) + (0 + y) * (1)) * 0.25;
                        DiNj(5, 3) = ((-1) * (1 - x) + (0 - x) * (-1)) * ((1) * (1 + y) + (0 + y) * (1)) * 0.25;
                        DiNj(5, 4) = ((-1) * (1 + x) + (1 - x) * (1)) * ((-1) * (1 - y) + (0 - y) * (-1)) * 0.50;
                        DiNj(5, 5) = ((1) * (1 + x) + (0 + x) * (1)) * ((-1) * (1 + y) + (1 - y) * (1)) * 0.50;
                        DiNj(5, 6) = ((-1) * (1 + x) + (1 - x) * (1)) * ((1) * (1 + y) + (0 + y) * (1)) * 0.50;
                        DiNj(5, 7) = ((-1) * (1 - x) + (0 - x) * (-1)) * ((-1) * (1 + y) + (1 - y) * (1)) * 0.50;
                        DiNj(5, 8) = ((-1) * (1 + x) + (1 - x) * (1)) * ((-1) * (1 + y) + (1 - y) * (1)) * 1.00;
                        DiNj(7, 0) = ((0 - x) * (1 - x)) * ((-1) * (-1) + (-1) * (-1)) * 0.25;
                        DiNj(7, 1) = ((0 + x) * (1 + x)) * ((-1) * (-1) + (-1) * (-1)) * 0.25;
                        DiNj(7, 2) = ((0 + x) * (1 + x)) * ((1) * (1) + (1) * (1)) * 0.25;
                        DiNj(7, 3) = ((0 - x) * (1 - x)) * ((1) * (1) + (1) * (1)) * 0.25;
                        DiNj(7, 4) = ((1 - x) * (1 + x)) * ((-1) * (-1) + (-1) * (-1)) * 0.50;
                        DiNj(7, 5) = ((0 + x) * (1 + x)) * ((-1) * (1) + (-1) * (1)) * 0.50;
                        DiNj(7, 6) = ((1 - x) * (1 + x)) * ((1) * (1) + (1) * (1)) * 0.50;
                        DiNj(7, 7) = ((0 - x) * (1 - x)) * ((-1) * (1) + (-1) * (1)) * 0.50;
                        DiNj(7, 8) = ((1 - x) * (1 + x)) * ((-1) * (1) + (-1) * (1)) * 1.00;

                    case 1:
                        DiNj(1, 0) = ((-1) * (1 - x) + (0 - x) * (-1)) * ((0 - y) * (1 - y)) * 0.25;
                        DiNj(1, 1) = ((1) * (1 + x) + (0 + x) * (1)) * ((0 - y) * (1 - y)) * 0.25;
                        DiNj(1, 2) = ((1) * (1 + x) + (0 + x) * (1)) * ((0 + y) * (1 + y)) * 0.25;
                        DiNj(1, 3) = ((-1) * (1 - x) + (0 - x) * (-1)) * ((0 + y) * (1 + y)) * 0.25;
                        DiNj(1, 4) = ((-1) * (1 + x) + (1 - x) * (1)) * ((0 - y) * (1 - y)) * 0.50;
                        DiNj(1, 5) = ((1) * (1 + x) + (0 + x) * (1)) * ((1 - y) * (1 + y)) * 0.50;
                        DiNj(1, 6) = ((-1) * (1 + x) + (1 - x) * (1)) * ((0 + y) * (1 + y)) * 0.50;
                        DiNj(1, 7) = ((-1) * (1 - x) + (0 - x) * (-1)) * ((1 - y) * (1 + y)) * 0.50;
                        DiNj(1, 8) = ((-1) * (1 + x) + (1 - x) * (1)) * ((1 - y) * (1 + y)) * 1.00;
                        DiNj(2, 0) = ((0 - x) * (1 - x)) * ((-1) * (1 - y) + (0 - y) * (-1)) * 0.25;
                        DiNj(2, 1) = ((0 + x) * (1 + x)) * ((-1) * (1 - y) + (0 - y) * (-1)) * 0.25;
                        DiNj(2, 2) = ((0 + x) * (1 + x)) * ((1) * (1 + y) + (0 + y) * (1)) * 0.25;
                        DiNj(2, 3) = ((0 - x) * (1 - x)) * ((1) * (1 + y) + (0 + y) * (1)) * 0.25;
                        DiNj(2, 4) = ((1 - x) * (1 + x)) * ((-1) * (1 - y) + (0 - y) * (-1)) * 0.50;
                        DiNj(2, 5) = ((0 + x) * (1 + x)) * ((-1) * (1 + y) + (1 - y) * (1)) * 0.50;
                        DiNj(2, 6) = ((1 - x) * (1 + x)) * ((1) * (1 + y) + (0 + y) * (1)) * 0.50;
                        DiNj(2, 7) = ((0 - x) * (1 - x)) * ((-1) * (1 + y) + (1 - y) * (1)) * 0.50;
                        DiNj(2, 8) = ((1 - x) * (1 + x)) * ((-1) * (1 + y) + (1 - y) * (1)) * 1.00;
                        DiNj(3, 0) = 0;
                        DiNj(3, 1) = 0;
                        DiNj(3, 2) = 0;
                        DiNj(3, 3) = 0;
                        DiNj(3, 4) = 0;
                        DiNj(3, 5) = 0;
                        DiNj(3, 6) = 0;
                        DiNj(3, 7) = 0;
                    case 0:
                        DiNj(0, 0) = ((0 - x) * (1 - x)) * ((0 - y) * (1 - y)) * 0.25;
                        DiNj(0, 1) = ((0 + x) * (1 + x)) * ((0 - y) * (1 - y)) * 0.25;
                        DiNj(0, 2) = ((0 + x) * (1 + x)) * ((0 + y) * (1 + y)) * 0.25;
                        DiNj(0, 3) = ((0 - x) * (1 - x)) * ((0 + y) * (1 + y)) * 0.25;
                        DiNj(0, 4) = ((1 - x) * (1 + x)) * ((0 - y) * (1 - y)) * 0.50;
                        DiNj(0, 5) = ((0 + x) * (1 + x)) * ((1 - y) * (1 + y)) * 0.50;
                        DiNj(0, 6) = ((1 - x) * (1 + x)) * ((0 + y) * (1 + y)) * 0.50;
                        DiNj(0, 7) = ((0 - x) * (1 - x)) * ((1 - y) * (1 + y)) * 0.50;
                        DiNj(0, 8) = ((1 - x) * (1 + x)) * ((1 - y) * (1 + y)) * 1.00;
                        break;
                    default:
                        assert(false);
                    }
                    break;

                default:
                    assert(false);
                    break;
                }
            }

            // // TODO: Integration method ("Eigen-Flavored Template")

            /**
             * @brief
             * sum(f(iintpoint, intpointPparam, D_iN_j) * wi), is integration of param space, responsible of feeding it shape function, coords
             * int point location, intpoint index.
             *
             * e.g.
             * if doing volume int for source term, before sent here, f should be aware of rec coeffs;
             * if doing facial int for Riemann, f should be aware of rec values of both-side (rec value buffer - shared)
             * if doing facial int for Bij, f should be aware of D_iPhi_j for both sides
             *
             *
             * the values of N, dNdxii... on gauss points should be stored, they are accessed by Intergration to calculate J (or J could be cached???),
             * and N, dNdxii stored on face gauss points should be accessed by face's Integration's f
             *
             * \param f likely std::function<void(Tacc & inc, int ig, Tpoint & pparam, tDiFj & DiFj)>,
             * accepts: (returned value to accumulate) (integration index in volume) (point in volume as param space) (DiNj for this volume at this point)
             * f may need info from father volumes (eg: DiNj) or some coords, so do it by hand with lambda or functors
             */
            template <class Tacc, class Tf>
            void Integration(Tacc &buf, Tf &&f)
            {
                for (int i = 0; i < nIntPoint; i++)
                {
                    Tacc acc;
                    tPoint p;
                    int d;
                    for (d = 0; d < dim; d++)
                        p(d) = bufInt[d * nIntPoint + i];
                    for (; d < 3; d++)
                        p(d) = 0.;

                    f(acc, i, p, NBuffer[elemType][0][iInt][i]);
                    // std::cout << acc << " " << bufInt[dim * nIntPoint + i] << std::endl;
                    // std::cout << HammerTri7 << " " << bufInt << " " << dim << std::endl;
                    // std::cout << "BUF " << buf << std::endl;
                    buf += acc * bufInt[dim * nIntPoint + i];
                }
                // switch (paramSpace)
                // {
                // case LineSpace:
                //     switch (iInt)
                //     {
                //     case INT_SCHEME_LINE_1:
                //     case INT_SCHEME_LINE_2:
                //     case INT_SCHEME_LINE_3:
                //         break;
                //     case INT_SCHEME_LINE_4:
                //         break;
                //     case INT_SCHEME_LINE_5:
                //         break;
                //     default:
                //         log() << "Line int scheme not implemented\n";
                //         break;
                //     }
                //     break;

                // case TriSpace:
                //     switch (iInt)
                //     {
                //     case INT_SCHEME_TRI_1:
                //         break;
                //     case INT_SCHEME_TRI_3:
                //         break;
                //     case INT_SCHEME_TRI_4:
                //         break;
                //     case INT_SCHEME_TRI_7:
                //         break;
                //     default:
                //         log() << "Tri int scheme not implemented\n";
                //         break;
                //     }
                //     break;

                // case QuadSpace:
                //     switch (iInt)
                //     {
                //     case INT_SCHEME_QUAD_1:
                //         break;
                //     case INT_SCHEME_QUAD_4:
                //         break;
                //     case INT_SCHEME_QUAD_9:
                //         break;
                //     case INT_SCHEME_QUAD_16:
                //         break;
                //     default:
                //         log() << "Quad int scheme not implemented\n";
                //         break;
                //     }
                //     break;
                // default:
                //     log() << "ParamSpace in Int not implemented\n";
                //     assert(false);
                //     break;
                // }
            }
        };
    }
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
    namespace Elem
    {
        // Some utility functions

        /**
         * @brief obtain jacobi (dx_j/dxi_i)
         * \param coords coords(all, i) = Vector({x,y,z}) for node i or for DiNj's N_i
         *
         * col major matrices preferred
         */
        template <class TMat>
        tJacobi DiNj2Jacobi(const tDiFj &DiNj, const TMat &coords)
        {
            return DiNj({1, 2, 3}, Eigen::all) * coords.transpose();
        }

        inline tPoint Jacobi2LineNorm2D(const tJacobi &Jacobi)
        {
            assert(Jacobi(0, 2) == 0.0);
            return tPoint{Jacobi(0, 1), -Jacobi(0, 0), 0.0};
        }

        template <class TMat>
        inline void Convert2dDiffsLinMap(TMat &&mat, const Eigen::Matrix2d &dXijdXi)
        {
            switch (mat.rows())
            {
            case 10:
                for (int iB = 0; iB < mat.cols(); iB++)
                {
                    Eigen::ETensorR3<real, 2, 2, 2> dPhidxiidxijdxik;
                    dPhidxiidxijdxik(0, 0, 0) = mat(6, iB);
                    dPhidxiidxijdxik(0, 0, 1) = dPhidxiidxijdxik(0, 1, 0) = dPhidxiidxijdxik(1, 0, 0) = mat(7, iB);
                    dPhidxiidxijdxik(0, 1, 1) = dPhidxiidxijdxik(1, 0, 1) = dPhidxiidxijdxik(1, 1, 0) = mat(8, iB);
                    dPhidxiidxijdxik(1, 1, 1) = mat(9, iB);
                    dPhidxiidxijdxik.MatTransform0(dXijdXi.transpose());
                    dPhidxiidxijdxik.MatTransform1(dXijdXi.transpose());
                    dPhidxiidxijdxik.MatTransform2(dXijdXi.transpose());
                    mat(6, iB) = dPhidxiidxijdxik(0, 0, 0);
                    mat(7, iB) = dPhidxiidxijdxik(0, 0, 1);
                    mat(8, iB) = dPhidxiidxijdxik(0, 1, 1);
                    mat(9, iB) = dPhidxiidxijdxik(1, 1, 1);
                }
            case 6:
                for (int iB = 0; iB < mat.cols(); iB++)
                {
                    Eigen::Matrix2d dPhidxiidxij{{mat(3, iB), mat(4, iB)},
                                                 {mat(4, iB), mat(5, iB)}};
                    dPhidxiidxij = dXijdXi * dPhidxiidxij * dXijdXi.transpose();
                    mat(3, iB) = dPhidxiidxij(0, 0), mat(4, iB) = dPhidxiidxij(0, 1), mat(5, iB) = dPhidxiidxij(1, 1);
                }
            case 3:
                mat({1, 2}, Eigen::all) = dXijdXi * mat({1, 2}, Eigen::all);
            case 1:
                break;

            default:
                assert(false);
                break;
            }
        }
    }

    inline bool CompareIndexVectors(std::vector<index> l, std::vector<index> r)
    {
        if (l.size() != r.size())
            return false;
        std::sort(l.begin(), l.end());
        std::sort(r.begin(), r.end());
        for (index i = 0; i < l.size(); i++)
            if (l[i] != r[i])
                return false;
        return true;
    }

}
