type FilterDesc
    ptr::Ptr{Void}

    function FilterDesc(x; format=CUDNN_TENSOR_NCHW)
        p = Ptr{Void}[0]
        cudnnCreateFilterDescriptor(p)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyFilterDescriptor)

        T = eltype(x)
        N = ndims(x)
        c_size = Cint[size(x,i) for i=N:-1:1]
        cudnnSetFilterNdDescriptor(desc, datatype(T), format, N, c_size)
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::FilterDesc) = desc.ptr
