module Interop

import ..CUJulia: CuArray, CuSubArray, box

const array_h = """
template<typename T, int N>
struct Array {
    T *data;
    const int dims[N];
    const int strides[N];
public:
    __device__ int length() {
        int n = dims[0];
        for (int i = 1; i < N; i++) n *= dims[i];
        return n;
    }
    __device__ T& operator[](const int idx) { return data[idx]; }
    __device__ T& operator()(const int idx) {
        int i = idx0 * strides[0];
        return data[i];
    }
    __device__ T& operator()(const int idx0, const int idx1) {
        int i = idx0 * strides[0];
        i += idx1 * strides[1];
        return data[i];
    }
    __device__ T& operator()(const int idx0, const int idx1, const int idx2) {
        int i = idx0 * strides[0];
        i += idx1 * strides[1];
        i += idx2 * strides[2];
        return data[i];
    }
    __device__ T& operator()(const int idx0, const int idx1, const int idx2, const int idx3) {
        int i = idx0 * strides[0];
        i += idx1 * strides[1];
        i += idx2 * strides[2];
        i += idx3 * strides[3];
        return data[i];
    }
    __device__ T& operator()(const int idx0, const int idx1, const int idx2, const int idx3, const int idx4) {
        int i = idx0 * strides[0];
        i += idx1 * strides[1];
        i += idx2 * strides[2];
        i += idx3 * strides[3];
        i += idx4 * strides[4];
        return data[i];
    }
    __device__ T& operator()(const int *idxs) {
        int i = 0;
        for (int k = 0; k < N; k++) {
            i += idxs[k] * strides[k];
        }
        return data[i];
    }
    __device__ void subindex(int *out, const int idx) {
        int cumdims[N];
        cumdims[0] = 1;
        for (int i = 1; i < N; i++) cumdims[i] = cumdims[i-1] * dims[i-1];

        for (int i = N-1; i >= 1; i--) {
            int k = idx / cumdims[i];
            out[i] = k;
            idx -= k * cumdims[i];
        }
        out[0] = idx;
        return;
    }
};
"""

immutable _CuArray{T,N}
    ptr::Ptr{T}
    dims::Ntuple{N,Cint}
    strides::Ntuple{N,Cint}
end

box{T}(x::CuArray{T}) = _CuArray(Ptr{T}(x), map(Cint,size(x)), map(Cint,strides(x)))
box{T}(x::CuSubArray{T}) = _CuArray(Ptr{T}(x), map(Cint,size(x)), map(Cint,strides(x)))

end
