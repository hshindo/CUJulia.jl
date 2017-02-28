export
    CuArray, CuVector, CuMatrix, CuVecOrMat,
    device

type CuArray{T,N}
    ptr::CuPtr
    dims::NTuple{N,Int}
end

typealias CuVector{T} CuArray{T,1}
typealias CuMatrix{T} CuArray{T,2}
typealias CuVecOrMat{T} Union{CuVector{T},CuMatrix{T}}

(::Type{CuArray{T}}){T,N}(dims::NTuple{N,Int}) = CuArray{T,N}(alloc(T,prod(dims)), dims)
(::Type{CuArray{T}}){T}(dims::Int...) = CuArray{T}(dims)
@generated function (::Type{CuArray{T}}){T,U}(x::CuArray{U})
    f = CuFunction("""
    __global__ void f($T *y, $U *x, int length) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) {
            y[idx] = x[idx];
        }
    }""")
    quote
        y = CuArray{T}(size(x))
        $f(y.ptr, x.ptr, length(y), dx=length(y))
        y
    end
end

CuArray{T,N}(x::Array{T,N}) = copy!(CuArray{T}(size(x)), x)
Base.Array{T,N}(x::CuArray{T,N}) = copy!(Array{T}(size(x)), x)

device(x::CuArray) = x.ptr.dev
Base.pointer{T}(x::CuArray{T}, index::Int=1) = Ptr{T}(x.ptr) + sizeof(T) * (index-1)

Base.length(x::CuArray) = prod(x.dims)
Base.size(x::CuArray) = x.dims
Base.size(x::CuArray, dim::Int) = x.dims[dim]
Base.ndims{T,N}(x::CuArray{T,N}) = N
Base.eltype{T}(x::CuArray{T}) = T
Base.isempty(x::CuArray) = length(x) == 0

Base.strides(x::CuVector) = (1,)
Base.strides(x::CuMatrix) = (1,size(x,1))
function Base.strides{T}(x::CuArray{T,3})
    s2 = size(x,1)
    s3 = s2 * size(x,2)
    (1,s2,s3)
end
function Base.strides{T}(x::CuArray{T,4})
    s2 = size(x,1)
    s3 = s2 * size(x,2)
    s4 = s3 * size(x,3)
    (1,s2,s3,s4)
end
function Base.stride{T,N}(x::CuArray{T,N}, dim::Int)
    d = 1
    for i = 1:dim-1
        d *= size(x,i)
    end
    d
end

Base.similar{T,N}(x::CuArray{T}, dims::NTuple{N,Int}) = CuArray{T}(dims)
Base.similar(x::CuArray) = similar(x, size(x))
Base.similar(x::CuArray, dims::Int...) = similar(x, dims)

Base.convert{T}(::Type{Ptr{T}}, x::CuArray{T}) = Ptr{T}(x.ptr)
Base.convert(::Type{CUdeviceptr}, x::CuArray) = CUdeviceptr(x.ptr)
Base.convert{T,N}(::Type{CuArray{T,N}}, x::Array{T,N}) = CuArray(x)
Base.convert{T,N}(::Type{Array{T,N}}, x::CuArray{T,N}) = Array(x)
Base.unsafe_convert{T}(::Type{Ptr{T}}, x::CuArray) = Ptr{T}(x.ptr)
Base.unsafe_convert(::Type{CUdeviceptr}, x::CuArray) = CUdeviceptr(x.ptr)

Base.zeros{T,N}(::Type{CuArray{T}}, dims::NTuple{N,Int}) = fill(CuArray, T(0), dims)
Base.zeros{T,N}(x::CuArray{T,N}) = zeros(CuArray{T}, size(x))
Base.zeros{T}(::Type{CuArray{T}}, dims::Int...) = zeros(CuArray{T}, dims)
Base.ones{T}(x::CuArray{T}) = ones(CuArray{T}, x.dims)
Base.ones{T}(::Type{CuArray{T}}, dims::Int...) = ones(CuArray{T}, dims)
Base.ones{T}(::Type{CuArray{T}}, dims) = fill(CuArray, T(1), dims)

function Base.copy!{T}(dest::Array{T}, src::CuArray{T}; stream=C_NULL)
    cuMemcpyDtoHAsync(dest, src, length(src)*sizeof(T), stream)
    dest
end
function Base.copy!{T}(dest::CuArray{T}, src::Array{T}; stream=C_NULL)
    cuMemcpyHtoDAsync(dest, src, length(src)*sizeof(T), stream)
    dest
end
function Base.copy!{T}(dest::CuArray{T}, src::CuArray{T}; stream=C_NULL)
    cuMemcpyDtoDAsync(dest, src, length(src)*sizeof(T), stream)
    dest
end
Base.copy(x::CuArray) = copy!(similar(x),x)

@generated function Base.fill!{T}(x::CuArray{T}, value)
    f = CuFunction("""
    __global__ void f($T *x, int length, $T value) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) {
            x[idx] = value;
        }
    }""")
    quote
        $f(x.ptr, length(x), T(value), dx=length(x))
        x
    end
end
Base.fill{T}(::Type{CuArray}, value::T, dims::NTuple) = fill!(CuArray{T}(dims), value)

Base.reshape{T,N}(x::CuArray{T}, dims::NTuple{N,Int}) = CuArray{T,N}(x.ptr, dims)
Base.reshape{T}(x::CuArray{T}, dims::Int...) = reshape(x, dims)

function redim{T,N}(x::CuArray{T,N}, n::Int; pad=0)
    dims = ntuple(n) do i
        1 <= i-pad <= N ? size(x,i-pad) : 1
    end
    reshape(x, dims)
end

function sub2dims(indexes::Tuple)
    ranges = Int[]
    dims = Int[]
    for i = 1:length(indexes)
        index = indexes[i]
        if isa(index, Range)
            push!(ranges, start(index)-1, step(index))
            push!(dims, length(index))
        elseif isa(index, Colon)
            push!(ranges, 0, 1)
            push!(dims, size(x,i))
        elseif isa(index, Int)
            push!(ranges, index-1, 0)
            push!(dims, 1)
        else
            throw("Invalid index.")
        end
    end
    ranges, ntuple(i -> dims[i], length(dims))
end

@generated function Base.getindex{T,N}(x::CuArray{T,N}, indexes...)
    f = CuFunction("""
    __global__ void f(Array<$T,$N> y, Array<$T,$N> x, Ranges<$N> ranges) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < y.length()) {
            int subs[$N];
            y.idx2sub(idx, subs);
            ranges.convert(subs);
            y[idx] = x(subs);
        }
    }
    """)
    quote
        ranges, dims = sub2dims(indexes)
        y = similar(x, dims)
        $f(y, x, ranges, dx=length(y))
        y
    end
end

@generated function Base.setindex!{T,N}(y::CuArray{T,N}, x::CuArray{T,N}, indexes...)
    f = CuFunction("""
    __global__ void f(Array<$T,$N> y, Array<$T,$N> x, Ranges<$N> ranges) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < x.length()) {
            int subs[$N];
            x.idx2sub(idx, subs);
            ranges.convert(subs);
            y(subs) = x[idx];
        }
    }
    """)
    quote
        ranges, dims = sub2dims(indexes)
        dims == size(x) || throw("Size mismatch.")
        $f(y, x, ranges, dx=length(x))
        y
    end
end

@generated function Base.cat(dim::Int, xs::CuArray...)
    f = CuFunction("""
    __global__ void f(void *p) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < 10) {

        }
    }
    """)
    quote
        p = Ptr{Void}[Ptr{T}(xs[i].ptr) for i=1:length(xs)]
        $f(p, dx=10)
    end
end

function cat2(dim::Int, A::CuArray...)
    cumdim = 0
    for x in A
        cumdim += size(x, dim)
    end
    outsize = [size(A[1])...]
    while length(outsize) < dim
        push!(outsize, 1)
    end
    outsize[dim] = cumdim
    y = similar(A[1], outsize...)
    range = map(s -> 1:s, outsize)
    offset = 1
    for x in A
        s = size(x, dim)
        range[dim] = offset:(offset+s-1)
        y[range...] = x
        offset += s
    end
    y
end

box{T}(a::CuArray{T}) = CUArray(Ptr{T}(a.ptr), box(size(a)))
