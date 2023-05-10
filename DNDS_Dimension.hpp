#pragma once

#include "DNDS_Defines.h"
#include <array>
#include <iostream>
#include "Eigen/Dense"

namespace DNDS
{
    namespace Dim
    {
        // enum BasicDim
        // {
        //     L = 0, // length
        //     M = 1, // mass
        //     T = 2, // time
        //     K = 3, // temp
        //     n = 4, // substance
        //     I = 5, // current
        //     IL = 6  // Luminous
        // };

        struct DimVal
        {
            std::array<int16_t, 7> v;

            DimVal(int16_t L,
                   int16_t M,
                   int16_t T,
                   int16_t K,
                   int16_t n,
                   int16_t I,
                   int16_t IL)
                : v{L,
                    M,
                    T,
                    K,
                    n,
                    I,
                    IL} {};

            DimVal(){};

            int16_t &operator[](const decltype(v)::size_type i)
            {
                return v[i];
            }
            int16_t operator[](const decltype(v)::size_type i) const
            {
                return v[i];
            }

            const DimVal operator*(const DimVal &r) const
            {
                DimVal ret;
                for (int i = 0; i < 7; i++)
                    ret[i] = (*this)[i] + r[i];
            }

            const DimVal operator/(const DimVal &r) const
            {
                DimVal ret;
                for (int i = 0; i < 7; i++)
                    ret[i] = (*this)[i] - r[i];
            }

            friend std::ostream &operator<<(std::ostream &o, const DimVal &d)
            {
                o << "L" << d[0] << ", ";
                o << "M" << d[1] << ", ";
                o << "T" << d[2] << ", ";
                o << "K" << d[3] << ", ";
                o << "n" << d[4] << ", ";
                o << "I" << d[5] << ", ";
                o << "IL" << d[6] << ", ";
            }
        };

        static const DimVal NonDim = {0, 0, 0, 0, 0, 0, 0};
        static const DimVal L = {1, 0, 0, 0, 0, 0, 0};
        static const DimVal M = {0, 1, 0, 0, 0, 0, 0};
        static const DimVal T = {0, 0, 1, 0, 0, 0, 0};
        static const DimVal K = {0, 0, 0, 1, 0, 0, 0};
        static const DimVal n = {0, 0, 0, 0, 1, 0, 0};
        static const DimVal I = {0, 0, 0, 0, 0, 1, 0};
        static const DimVal IL = {0, 0, 0, 0, 0, 0, 1};

        static const DimVal Length = L;
        static const DimVal Mass = M;
        static const DimVal Time = T;
        static const DimVal Temperature = K;
        static const DimVal Substance = n;
        static const DimVal ElectricCurrent = I;
        static const DimVal Luminous = IL;

        static const DimVal Speed = Length / Time;
        static const DimVal Acceleration = Speed / Time;
        static const DimVal Force = Mass * Acceleration;
        static const DimVal Energy = Force * Length;

        static const DimVal Momentum = Force * Time;

        static const DimVal SpecificHeat = Energy / Temperature;

        static const DimVal Area3D = Length * Length;
        static const DimVal Volume3D = Area3D * Length;
        static const DimVal MassDensity3D = Mass / Volume3D;
        static const DimVal MomentumDensity3D = Momentum / Volume3D;
        static const DimVal EnergyDensity3D = Energy / Volume3D;
        static const DimVal Pressure3D = Energy / Volume3D;

        static const DimVal DynamicViscosity = Pressure3D / (Speed / Length);

        class DimensionManager
        {
            int dim = 3;
            std::vector<std::pair<DimVal, real>> references;
            Eigen::Vector<real, 7> basic_ref;

        public:
            DimensionManager(int ndim) : dim(ndim) { basic_ref.setConstant(1); }
            void ClearRef() { references.clear(), basic_ref.setConstant(1); }
            void AddRef(const DimVal &dimnew, real valnew) { references.push_back(std::make_pair(dimnew, valnew)); }
            void SolveRef()
            {
                if (references.size() != 7)
                {
                    std::cout << "Need 7 Basics" << std::endl;
                    DNDS_assert(false);
                }

                Eigen::Vector<real, 7> rhs;
                Eigen::Matrix<real, 7, 7> mat;
                mat.setZero();
                for (int i = 0; i < 7; i++)
                    for (int j = 0; j < 7; j++)
                        mat(i, j) = references[i].first[j];

                for (int i = 0; i < 7; i++)
                    rhs(i) = std::log(references[i].second);

                auto LU = mat.fullPivLu();
                if (LU.rank() != 7)
                {
                    std::cout << "ill posed constraints for dim refs! " << std::endl;
                    DNDS_assert(false);
                }

                basic_ref = LU.solve(rhs);
                for (auto &i : basic_ref)
                    i = std::exp(i);
            }

            void Print()
            {
                std::cout << "Basic Reference Dims: [" << basic_ref.transpose() << "] " << std::endl;
            }

            real getRefValue(DimVal d)
            {
                real ret = 1;
                for (int i = 0; i < 7; i++)
                    ret *= std::pow(basic_ref(i), d[i]);
            }
        };
    }

}