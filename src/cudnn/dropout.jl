type DropoutDesc
    ptr::Ptr{Void}

    function DropoutDesc(h, droprate::Float64; seed=0)
        p = Ptr{Void}[0]
        cudnnCreateDropoutDescriptor(p)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyDropoutDescriptor)

        p = Cint[0]
        cudnnDropoutGetStatesSize(h, p)
        statessize = p[1]
        states = CuArray{Int8}(Int(statessize))
        cudnnSetDropoutDescriptor(desc, h, droprate, states, statessize, seed)
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::DropoutDesc) = desc.ptr

function dropout{T}(x::CuArray{T}, droprate::Float64)
    h = handle(x)
    dropdesc = DropoutDesc(h, droprate)

    xdesc = TensorDesc(x)
    p = Cint[0]
    cudnnDropoutGetReserveSpaceSize(xdesc, p)
    reservesize = p[1]
    reservespace = CuArray{Int8}(Int(reservesize))

    y = similar(x)
    cudnnDropoutForward(h, dropdesc, xdesc, x, xdesc, y, reservespace, reservesize)

    function backward!(gy, gx)
        isvoid(gx) && return
        cudnnDropoutBackward(h, dropdesc, xdesc, dy, xdesc, dx, reservespace, reservesize)
    end
    y, backward!
end
