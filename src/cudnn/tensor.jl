type TensorDesc
    ptr::Ptr{Void}

    function TensorDesc(x; dims)
        T = eltype(x)
        N = ndims(x)
        @assert N <= 4
        csize = Cint[1, 1, 1, 1]
        cstrides = Cint[1, 1, 1, 1]
        st = strides(x)
        for i = 1:N
            csize[4-i-pad+1] = size(x,i)
            cstrides[4-i-pad+1] = st[i]
        end
        p = Ptr{Void}[0]
        cudnnCreateTensorDescriptor(p)
        cudnnSetTensorNdDescriptor(p[1], datatype(T), 4, csize, cstrides)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyTensorDescriptor)
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::TensorDesc) = desc.ptr
