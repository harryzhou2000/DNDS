#pragma once
#include <iostream>
#include <cassert>

#include <utility>
#include "unsupported/Eigen/CXX11/Tensor"

namespace SmallTensor
{
    typedef int index;
    // index

    // template <typename in0> // SFINAE recursion base
    // constexpr in0 __t_prod(in0 v0)
    // {
    //     return v0;
    // }

    // template <typename in0, typename... in>
    // constexpr in0 __t_prod(in0 v0, in... v)
    // {
    //     return v0 * __t_prod(v...);
    // }

    template <typename in0, typename... in>
    constexpr in0 __t_prod(in0 v0, in... v)
    {
        auto v00 = v0;
        in0 vs[sizeof...(v)] = {v...};
        for (index i = 0; i < sizeof...(v); i++)
            v00 *= vs[i];
        return v00;
    }

    template <typename in0, typename... in>
    constexpr bool __t_gt0(in0 v0, in... v)
    {
        auto v00 = v0 > 0;
        in0 vs[sizeof...(v)] = {v...};
        for (index i = 0; i < sizeof...(v); i++)
            v00 = vs[i] > 0 && v00;
        return v00;
    }

    template <typename T, index... dims>
    class Tensor
    {
        static_assert(__t_gt0(dims...), "all dims must be larger than 0");
        static const index _order = sizeof...(dims);
        static const index _dim[_order];
        static const index _size = __t_prod(dims...);

        T _data[_size];

    public:
        index order()
        {
            return _order;
        }

        index dim(index ind)
        {
            assert(ind >= 0 && ind < _order);
            return _dim[ind];
        }

        index size()
        {
            return _size;
        }

        Tensor()
        {
            for (index i = 0; i < _size; i++)
                _data[i] = 0.0;
        }

        Tensor(const T &v)
        {
            for (index i = 0; i < _size; i++)
                _data[i] = v;
        }

        Tensor(const Tensor<T, dims...> &R) // * copy ctor, trivial
        {
            for (index i = 0; i < _size; i++)
                _data[i] = R._data[i];
        }

        //! no move ctor defined!!

        void operator=(const Tensor<T, dims...> &R) // * copy =, trivial
        {
            for (index i = 0; i < _size; i++)
                _data[i] = R._data[i];
        }

        //! no move = defined!!

        template <typename ind0, typename... ind>
        static index __t_index(ind0 i0, ind... i)
        {
            ind0 icur = i0;
            ind0 is[sizeof...(i)] = {i...};
            for (index ii = 0; ii < sizeof...(i); ii++)
                icur = icur * _dim[ii + 1] + is[ii];
            return icur;
        }

        template <typename... ind>
        T &operator()(ind... dim_ind)
        {
            static_assert(sizeof...(dim_ind) == _order, "input num of dim_ind must be same as order");
            return _data[__tindex(dim_ind...)];
        }

        void operator*=(const T &r)
        {
            for (index i = 0; i < _size; i++)
                _data[i] *= r;
        }

        Tensor<T, dims...> operator*(const T &r)
        {
            Tensor<T, dims...> ret;
            for (index i = 0; i < _size; i++)
                ret._data[i] = _data[i] * r;
            return ret;
        }

        void operator+=(const Tensor<T, dims...> &R)
        {
            for (index i = 0; i < _size; i++)
                _data[i] += R._data[i];
        }

        void operator-=(const Tensor<T, dims...> &R)
        {
            for (index i = 0; i < _size; i++)
                _data[i] -= R._data[i];
        }

        Tensor<T, dims...> operator+(const Tensor<T, dims...> &R)
        {
            Tensor<T, dims...> ret;
            for (index i = 0; i < _size; i++)
                ret._data[i] = _data[i] + R._data[i];
            return ret;
        }

        Tensor<T, dims...> operator-(const Tensor<T, dims...> &R)
        {
            Tensor<T, dims...> ret;
            for (index i = 0; i < _size; i++)
                ret._data[i] = _data[i] - R._data[i];
            return ret;
        }

        Tensor<T, dims...> operator+()
        {
            Tensor<T, dims...> ret;
            for (index i = 0; i < _size; i++)
                ret._data[i] = _data[i];
            return ret;
        }

        Tensor<T, dims...> operator-()
        {
            Tensor<T, dims...> ret;
            for (index i = 0; i < _size; i++)
                ret._data[i] = -_data[i];
            return ret;
        }

        Tensor<T, dims...> Contraction(index i, index j)
        {
        }
    };

    template <typename T, index... dims>
    const index Tensor<T, dims...>::_dim[Tensor<T, dims...>::_order] = {dims...};

}