const array_h = """
template<typename T, int N>
struct Array {
    T *data;
    const int dims[N];
public:
    __device__ int length() {
        int n = dims[0];
        for (int i = 1; i < N; i++) n *= dims[i];
        return n;
    }
    __device__ T& operator[](const int idx) { return data[idx]; }
    __device__ T& operator()(int idx0, int idx1) {
        return data[idx0 + idx1*dims[0]];
    }
    __device__ T& operator()(int idx0, int idx1, int idx2) {
        int i = idx0;
        i += idx1 * dims[0];
        i += idx2 * dims[0] * dims[1];
        return data[i];
    }
    __device__ T& operator()(int idx0, int idx1, int idx2, int idx3) {
        int i = idx0;
        i += idx1 * dims[0];
        i += idx2 * dims[0] * dims[1];
        i += idx3 * dims[0] * dims[1] * dims[2];
        return data[i];
    }
    __device__ T& operator()(int idx0, int idx1, int idx2, int idx3, int idx4) {
        int i = idx0;
        i += idx1 * dims[0];
        i += idx2 * dims[0] * dims[1];
        i += idx3 * dims[0] * dims[1] * dims[2];
        i += idx4 * dims[0] * dims[1] * dims[2] * dims[3];
        return data[i];
    }
    __device__ void idx2sub(const int idx, int *subs) {
        int cumdims[N];
        cumdims[0] = 1;
        for (int i = 1; i < N; i++) cumdims[i] = cumdims[i-1] * dims[i-1];

        int temp = idx;
        for (int i = N-1; i >= 1; i--) {
            int k = temp / cumdims[i];
            subs[i] = k;
            temp -= k * cumdims[i];
        }
        subs[0] = temp;
        return;
    }
    __device__ T& operator()(int *subs) {
        int idx = 0;
        int stride = 1;
        for (int i = 0; i < N; i++) {
            if (dims[i] > 1) idx += subs[i] * stride;
            stride *= dims[i];
        }
        return data[idx];
    }
};
"""
