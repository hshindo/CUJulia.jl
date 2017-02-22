import Base: broadcast, broadcast!

for (op1,op2) in ((:.+,:+), (:.-,:-), (:.*,:*))
    @eval begin
        @generated function broadcast!{T,N}(::typeof($op1), y::CuArray{T,N}, x1::CuArray{T,N}, x2::CuArray{T,N})
            op = $op2
            f = CuFunction("""
            __global__ void f(Array<$T,$N> y, Array<$T,$N> x1, Array<$T,$N> x2) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < y.length()) {
                    int subs[$N];
                    y.idx2sub(idx, subs);
                    y[idx] = x1(subs) $op x2(subs);
                }
            }""")
            quote
                $f(y, x1, x2, dx=length(y))
                y
            end
        end
    end
end
