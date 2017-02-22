import Base: exp, log
import Base: .+, +, .-, -, .*, *

for op in (:exp, :log)
    @eval begin
        @generated function $op{T}(x::CuArray{T})
            op = $op
            f = CuFunction("""
            __global__ void f($T *x, $T *y, int length) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < length) {
                    y[idx] = $op(x[idx]);
                }
            }""")
            quote
                y = similar(x)
                $f(x.ptr, y.ptr, length(x), dx=length(x))
                y
            end
        end
    end
end

for op in (:+, :-)
    @eval begin
        @generated function $op{T}(x1::CuArray{T}, x2::CuArray{T})
            op = $op
            f = CuFunction("""
            __global__ void f($T *x1, $T *x2, $T *y, int length) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < length) {
                    y[idx] = x1[idx] $op x2[idx];
                }
            }""")
            quote
                size(x1) == size(x2) || throw(DimensionMismatch())
                y = similar(x1)
                $f(x1.ptr, x2.ptr, y.ptr, length(y), dx=length(y))
                y
            end
        end
    end
end

for op in (:.+, :.-, :.*)
    @eval begin
        function $op{T,N}(x1::CuArray{T,N}, x2::CuArray{T,N})
            dims = ntuple(N) do i
                size(x1,i) == size(x2,i) && return size(x1,i)
                size(x1,i) == 1 && return size(x2,i)
                size(x2,i) == 1 && return size(x1,i)
                throw("Cannot be broadcasted.")
            end
            y = CuArray{T}(dims)
            broadcast!($op, y, x1, x2)
        end
    end
end

function *(x1::CuArray, x2::CuArray)
    BLAS.gemm(x1, x2)
end
