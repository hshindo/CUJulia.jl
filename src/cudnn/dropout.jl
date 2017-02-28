export
    cudnnDropoutForward,
    cudnnDropoutBackward

type DropoutDesc
    ptr::Ptr{Void}
    reservespace
    reservesize

    function DropoutDesc(h, xdesc)
        p = Ptr{Void}[0]
        cudnnCreateDropoutDescriptor(p)
        ptr = p[1]

        p = Cint[0]
        cudnnDropoutGetStatesSize(h, p)
        statessize = p[1]
        states = CuArray{Int8}(Int(statessize))

        p = Cint[0]
        cudnnDropoutGetReserveSpaceSize(xdesc, p)
        reservesize = p[1]
        reservespace = CuArray{Int8}(Int(reservesize))
        cudnnSetDropoutDescriptor(ptr, h, rate, states, statessize, 0)

        desc = new(ptr, reservespace, reservesize)
        finalizer(desc, cudnnDestroyDropoutDescriptor)
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::DropoutDesc) = desc.ptr
