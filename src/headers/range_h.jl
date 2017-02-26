const range_h = """
struct Range {
    const int start;
    const int step;
public:
    __device__ int operator()(const int idx) { return start + step*idx; }
};

template<int N>
struct Ranges {
    Range data[N];
public:
    __device__ Range& operator[](const int idx) { return data[idx]; }

    __device__ void convert(int *subs) {
        for (int i = 0; i < N; i++) {
            subs[i] = data[i](subs[i]);
        }
    }
};
"""
