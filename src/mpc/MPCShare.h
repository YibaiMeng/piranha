// Base class of all MPC shares 
#pragma once
#include <cstddef>
#include <initializer_list>

#include <cutlass/conv/convolution.h>

#include "../gpu/DeviceData.h"
#include "../globals.h"

template <typename T, typename I>
class MPCShareBase
{

protected:
    MPCShareBase(DeviceData<T, I> *a);

public:
    static const int numParties;
    void set(DeviceData<T, I> *a);
    size_t size() const;
    void zero();
    void fill(T val);
    void setPublic(std::vector<double> &v);
    DeviceData<T, I> *getShare(int i);
    const DeviceData<T, I> *getShare(int i) const;
    /// Copy the Shares to dst. May be on another GPU.
    void copyTo(MPCShareBase<T, I> &dst) const;
    static int numShares();
    static int nextParty(int party);
    static int prevParty(int party);
    typedef T share_type;
    typedef I iterator_type;

    MPCShareBase<T, I> &operator+=(const T rhs);
    MPCShareBase<T, I> &operator-=(const T rhs);
    MPCShareBase<T, I> &operator*=(const T rhs);
    MPCShareBase<T, I> &operator>>=(const T rhs);

    template <typename I2>
    MPCShareBase<T, I> &operator+=(const DeviceData<T, I2> &rhs);
    template <typename I2>
    MPCShareBase<T, I> &operator-=(const DeviceData<T, I2> &rhs);
    template <typename I2>
    MPCShareBase<T, I> &operator*=(const DeviceData<T, I2> &rhs);
    template <typename I2>
    MPCShareBase<T, I> &operator^=(const DeviceData<T, I2> &rhs);
    template <typename I2>
    MPCShareBase<T, I> &operator&=(const DeviceData<T, I2> &rhs);
    template <typename I2>
    MPCShareBase<T, I> &operator>>=(const DeviceData<T, I2> &rhs);
    template <typename I2>
    MPCShareBase<T, I> &operator<<=(const DeviceData<T, I2> &rhs);
    template <typename I2>
    MPCShareBase<T, I> &operator+=(const MPCShareBase<T, I2> &rhs);
    template <typename I2>
    MPCShareBase<T, I> &operator-=(const MPCShareBase<T, I2> &rhs);
    template <typename I2>
    MPCShareBase<T, I> &operator*=(const MPCShareBase<T, I2> &rhs);
    template <typename I2>
    MPCShareBase<T, I> &operator^=(const MPCShareBase<T, I2> &rhs);
    template <typename I2>
    MPCShareBase<T, I> &operator&=(const MPCShareBase<T, I2> &rhs);
};

template <typename T, typename I = BufferIterator<T>>
class MPCShare : public MPCShareBase<T, I>
{

public:
    MPCShare(DeviceData<T, I> *a);
};

template <typename T>
class MPCShare<T, BufferIterator<T>> : public MPCShareBase<T, BufferIterator<T>>
{

public:
    MPCShare(DeviceData<T> *a);
    MPCShare(size_t n);
    MPCShare(std::initializer_list<double> il, bool convertToFixedPoint = true);

    void resize(size_t n);

private:
    DeviceData<T> _shareA;
};

// Functionality

template <typename T, typename I>
void dividePublic(MPCShare<T, I> &a, T denominator);

template <typename T, typename I, typename I2>
void dividePublic(MPCShare<T, I> &a, DeviceData<T, I2> &denominators);

template <typename T, typename I, typename I2>
void reconstruct(MPCShare<T, I> &in, DeviceData<T, I2> &out);

template <typename T>
void matmul(const MPCShare<T> &a, const MPCShare<T> &b, MPCShare<T> &c,
            int M, int N, int K,
            bool transpose_a, bool transpose_b, bool transpose_c, T truncation);

template <typename T, typename U, typename I, typename I2, typename I3, typename I4>
void selectShare(const MPCShare<T, I> &x, const MPCShare<T, I2> &y, const MPCShare<U, I3> &b, MPCShare<T, I4> &z);

template <typename T, typename I, typename I2>
void sqrt(const MPCShare<T, I> &in, MPCShare<T, I2> &out);

template <typename T, typename I, typename I2>
void inverse(const MPCShare<T, I> &in, MPCShare<T, I2> &out);

template <typename T, typename I, typename I2>
void sigmoid(const MPCShare<T, I> &in, MPCShare<T, I2> &out);

template <typename T>
void convolution(const MPCShare<T> &A, const MPCShare<T> &B, MPCShare<T> &C,
                 cutlass::conv::Operator op,
                 int batchSize, int imageHeight, int imageWidth, int filterSize,
                 int Din, int Dout, int stride, int padding, int truncation);

template <typename T, typename U, typename I, typename I2>
void dReLU(const MPCShare<T, I> &input, MPCShare<U, I2> &result);

template <typename T, typename U, typename I, typename I2, typename I3>
void ReLU(const MPCShare<T, I> &input, MPCShare<T, I2> &result, MPCShare<U, I3> &dresult);

template <typename T, typename U, typename I, typename I2, typename I3>
void maxpool(const MPCShare<T, I> &input, MPCShare<T, I2> &result, MPCShare<U, I3> &dresult, int k);