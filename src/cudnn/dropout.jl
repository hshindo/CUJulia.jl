export
    cudnnDropoutGetStatesSize,
    cudnnDropoutGetReserveSpaceSize,
    cudnnSetDropoutDescriptor,
    cudnnDropoutForward,
    cudnnDropoutBackward

type DropoutDesc
    ptr::Ptr{Void}

    function DropoutDesc()
        p = Ptr{Void}[0]
        cudnnCreateDropoutDescriptor(p)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyDropoutDescriptor)
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::DropoutDesc) = desc.ptr
