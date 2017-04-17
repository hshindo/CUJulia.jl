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

function dropout(x, droprate::Float64)
    dropdesc = DropoutDesc()
    h = handle(x)
    p = Cint[0]
    cudnnDropoutGetStatesSize(h, p)
    statessize = p[1]
    states = CuArray{Int8}(Int(statessize))
    cudnnSetDropoutDescriptor(dropdesc, h, droprate, states, statessize, 0)

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
