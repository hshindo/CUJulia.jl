type TensorDesc
    ptr::Ptr{Void}

    function TensorDesc{T,N}(x::CuArray{T,N})
        csize = Cint[size(x,i) for i=N:-1:1]
        cstrides = Cint[stride(x,i) for i=N:-1:1]
        p = Ptr{Void}[0]
        cudnnCreateTensorDescriptor(p)
        cudnnSetTensorNdDescriptor(p[1], datatype(T), N, csize, cstrides)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyTensorDescriptor)
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::TensorDesc) = desc.ptr
