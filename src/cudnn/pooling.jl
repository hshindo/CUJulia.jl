export
    # cudnnPoolingMode_t
    CUDNN_POOLING_MAX,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,

    # cudnnNanPropagation_t
    CUDNN_NOT_PROPAGATE_NAN,
    CUDNN_PROPAGATE_NAN

type PoolingDesc
    ptr::Ptr{Void}

    function PoolingDesc()
        p = Ptr{Void}[0]
        cudnnCreatePoolingDescriptor(p)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyPoolingDescriptor)
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::PoolingDesc) = desc.ptr

function pooling{N}(mode, window::NTuple{N,Int}, padding::NTuple{N,Int}, stride::NTuple{N,Int}, x;
    maxpoolingNanOpt=CUDNN_NOT_PROPAGATE_NAN)

    pooldesc = PoolingDesc()
    c_window = Cint[window[i] for i=N:-1:1]
    c_padding = Cint[padding[i] for i=N:-1:1]
    c_stride = Cint[stride[i] for i=N:-1:1]
    cudnnSetPoolingNdDescriptor(pooldesc, mode, maxpoolingNanOpt, N, c_window, c_padding, c_stride)

    h = handle(x)
    T = eltype(x)
    outdims = ntuple(N) do i
        (size(x,i) + 2padding[i] - window[i]) ÷ stride[i] + 1
    end
    y = similar(x, outdims..., size(x,N+1), size(x,N+2))
    xdesc = TensorDesc(x)
    ydesc = TensorDesc(y)
    cudnnPoolingForward(h, pooldesc, T[1], xdesc, x, T[0], ydesc, y)

    function backward!(gy, gx)
        isvoid(gx) && return
        cudnnPoolingBackward(h, pooldesc, T[1], ydesc, y, ydesc, dy, xdesc, x, T[1], xdesc, dx)
    end
    y, backward!
end
