type FilterDesc
    ptr::Ptr{Void}

    function FilterDesc{T,N}(x::CuArray{T,N})
        csize = Cint[size(x,i) for i=N:-1:1]
        p = Ptr{Void}[0]
        cudnnCreateFilterDescriptor(p)
        cudnnSetFilterNdDescriptor(p[1], datatype(T), format, N, csize)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyFilterDescriptor)
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::FilterDesc) = desc.ptr
