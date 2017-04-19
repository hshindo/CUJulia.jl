export
    # cudnnConvolutionMode_t
    CUDNN_CONVOLUTION,
    CUDNN_CROSS_CORRELATION,

    # cudnnConvolutionFwdPreference_t
    CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT

type ConvolutionDesc
    ptr::Ptr{Void}

    function ConvolutionDesc()
        p = Ptr{Void}[0]
        cudnnCreateConvolutionDescriptor(p)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyConvolutionDescriptor)
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::ConvolutionDesc) = desc.ptr

function convolution{T,N}(x::CuArray{T}, w::CuArray{T}, b::CuArray{T},
    pads::NTuple{N,Int}, strides::NTuple{N,Int}; mode=CUDNN_CROSS_CORRELATION)

    convdesc = ConvolutionDesc()
    c_pads = Cint[pads[i] for i=N:-1:1]
    c_strides = Cint[strides[i] for i=N:-1:1]
    c_upscale = fill(Cint(1), N)
    cudnnSetConvolutionNdDescriptor(convdesc, N, c_pads, c_strides, c_upscale, mode, datatype(T))

    outdims = ntuple(N) do i
        (size(x,i) + 2pads[i] - size(w,i)) ÷ strides[i] + 1
    end
    y = similar(x, outdims..., size(w,N+2), size(x,N+2))
    xdesc = TensorDesc(x)
    wdesc = FilterDesc(w)
    ydesc = TensorDesc(y)

    h = handle(x)
    p = cudnnConvolutionFwdAlgo_t[0]
    cudnnGetConvolutionForwardAlgorithm(h, xdesc, wdesc, convdesc, ydesc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, p)
    algo = p[1]

    p = Cint[0]
    cudnnGetConvolutionForwardWorkspaceSize(h, xdesc, wdesc, convdesc, ydesc, algo, p)
    worksize = p[1]
    workspace = CuArray{Int8}(Int(worksize))

    cudnnConvolutionForward(h, T[1], xdesc, x, wdesc, w, convdesc,
        algo, workspace, worksize, T[0], ydesc, y)
    dims1 = ntuple(_ -> 1, ndims(y)-ndims(b))
    b = reshape(b, size(b)..., dims1...)
    addtensor!(b, y)

    function backward!(gy, gx, gw, gb)
        isvoid(gw) || ∇convolution_filter!(xdesc, x, ydesc, gy, convdesc, wdesc, gw)
        isvoid(gx) || ∇convolution_data!(wdesc, w, ydesc, gy, convdesc, xdesc, gx)
        isvoid(gb) || ∇convolution_bias!(ydesc, gy, bdesc, gb)
    end
    y, backward!
end

function ∇convolution_filter!(xdesc, x, dydesc, dy, convdesc, dwdesc, dw)
    h = handle(x)
    p = cudnnConvolutionBwdFilterAlgo_t[0]
    cudnnGetConvolutionBackwardFilterAlgorithm(h, xdesc, dydesc, convdesc, dwdesc,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, p)
    algo = p[1]

    p = Cint[0]
    cudnnGetConvolutionBackwardFilterWorkspaceSize(h, xdesc, dydesc, convdesc, dwdesc, algo, p)
    worksize = p[1]
    workspace = CuArray{Int8}(Int(p[1]))

    T = eltype(x)
    cudnnConvolutionBackwardFilter(h, T[1], xdesc, x, dydesc, dy, convdesc,
        algo, workspace, worksize, T[1], dwdesc, dw)
end

function ∇convolution_data!(wdesc, w, dydesc, dy, convdesc, dxdesc, dx)
    h = handle(dy)
    p = cudnnConvolutionBwdDataAlgo_t[0]
    cudnnGetConvolutionBackwardDataAlgorithm(h, wdesc, dydesc, convdesc, dxdesc,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, p)
    algo = p[1]

    p = Cint[0]
    cudnnGetConvolutionBackwardDataWorkspaceSize(h, wdesc, dydesc, convdesc,
        dxdesc, algo, p)
    worksize = p[1]
    workspace = CuArray{Int8}(Int(worksize))

    T = eltype(dy)
    cudnnConvolutionBackwardData(h, T[1], wdesc, w, dydesc, dy, convdesc,
        algo, workspace, worksize, T[1], dxdesc, dx)
end

function ∇convolution_bias!(dydesc, dy, dbdesc, db)
    h = handle(dy)
    T = eltype(dy)
    cudnnConvolutionBackwardBias(h, T[1], dydesc, dy, T[1], dbdesc, db)
end
