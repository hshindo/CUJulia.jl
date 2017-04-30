export
    # cudnnRNNInputMode_t
    CUDNN_LINEAR_INPUT,
    CUDNN_SKIP_INPUT,

    # cudnnDirectionMode_t
    CUDNN_UNIDIRECTIONAL,
    CUDNN_BIDIRECTIONAL,

    # cudnnRNNMode_t
    CUDNN_RNN_RELU,
    CUDNN_RNN_TANH,
    CUDNN_LSTM,
    CUDNN_GRU

type RNNDesc
    ptr::Ptr{Void}

    function RNNDesc()
        p = Ptr{Void}[0]
        cudnnCreateRNNDescriptor(p)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyRNNDescriptor)
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::RNNDesc) = desc.ptr

type RNN
    desc::RNNDesc
    wdesc
    w
    hxdesc
    hx
    cxdesc
    cx
end

"""
* w
* hx: (hsize, batchsize, layersize)
* cx: same as hx
"""
function RNN{T}(w::CuArray{T}, hx::CuArray{T,3}, cx::CuArray{T,3},
    droprate::Float64, dir, mode; inputmode=CUDNN_LINEAR_INPUT)

    h = handle(w)
    rnndesc = RNNDesc()
    dropdesc = DropoutDesc(h, droprate)
    hsize = size(hx, 1)
    nlayers = size(hx, 3)
    cudnnSetRNNDescriptor(rnndesc, hsize, nlayers, dropdesc, inputmode, dir, mode, datatype(T))

    xdesc = TensorDesc(CuArray{T}(1,size(hx,1),size(hx,2)))
    p = Csize_t[0]
    cudnnGetRNNParamsSize(h, rnndesc, xdesc, p, datatype(T))
    Int(p[1]) == length(w) || throw("The number of parameters is wrong: $(p[1]), $(length(w)).")
    wdesc = FilterDesc(reshape(w,1,1,length(w)))

    #=
    if mode == CUDNN_RNN_RELU || mode == CUDNN_RNN_TANH
        nids = 2
    elseif mode == CUDNN_LSTM
        nids = 8
    elseif mode == CUDNN_GRU
        nids = 6
    end
    for l = 1:(dir == CUDNN_BIDIRECTIONAL ? nlayers*2 : nlayers)
        for id = 1:nids
            p = Ptr{Void}[0]
            cudnnGetRNNLinLayerMatrixParams(h, rnndesc, l-1, xdesc, wdesc, w, id-1, C_NULL, p)
            println(UInt64(p[1]))

            p = Ptr{Void}[0]
            cudnnGetRNNLinLayerBiasParams(h, rnndesc, l-1, xdesc, wdesc, w, id-1, C_NULL, p)
            println(UInt64(p[1]))
        end
    end
    =#
    RNN(rnndesc, wdesc, w, TensorDesc(hx), hx, TensorDesc(cx), cx)
end

function (rnn::RNN){T}(xs::Vector{CuMatrix{T}})
    h = handle(xs[1])
    xdescs = map(x -> TensorDesc(reshape(x,1,size(x)...)), xs)
    xdesc = ydesc = map(x -> x.ptr, xdescs)
    y = similar(x)
    hydesc = rnn.hxdesc
    hy = similar(hx)
    cydesc = rnn.cxdesc
    cy = similar(cx)

    p = Csize_t[0]
    cudnnGetRNNWorkspaceSize(h, rnn.desc, length(xs), xdesc, p)
    workspace = CuArray{Int8}(Int(p[1]))

    p = Csize_t[0]
    cudnnGetRNNTrainingReserveSize(h, rnn.desc, length(xs), xdesc, p)
    reservespace = CuArray{Int8}(Int(p[1]))

    x = map(x -> Ptr{Void}(x), xs)
    cudnnRNNForwardTraining(h, rnn.desc, length(xs), xdesc, x, rnn.hxdesc, rnn.hx, rnn.cxdesc, rnn.cx,
        rnn.wdesc, rnn.w, ydesc, y, hydesc, hy, cydesc, cy,
        workspace, length(workspace), reservespace, length(reservespace))
end

function rnn{T}(hsize::Int, nlayers::Int, droprate::Float64, dir, mode, w::CuMatrix{T}, b::CuMatrix{T},
    xs::Vector{CuMatrix{T}}, hx::CuArray{T}, cx::CuArray{T};
    inputmode=CUDNN_LINEAR_INPUT)

    h = handle(xs[1])
    rnndesc = RNNDesc()
    dropdesc = DropoutDesc(h, droprate)
    cudnnSetRNNDescriptor(rnndesc, hsize, nlayers, dropdesc, inputmode, dir, mode, datatype(T))

    x = map(x -> Ptr{Void}(x), xs)
    xdescs = map(x -> TensorDesc(reshape(x,1,size(x)...)), xs)
    xdesc = map(x -> x.ptr, xdescs)
    hxdesc = cxdesc = hydesc = cydesc = TensorDesc(hx)
    wdesc = FilterDesc(reshape(w,1,1,length(w)+length(b)))

    y = similar(x)
    hy = similar(hx)
    cy = similar(cx)

    p = Csize_t[0]
    cudnnGetRNNWorkspaceSize(h, rnndesc, length(xs), xdesc, p)
    workspace = CuArray{Int8}(Int(p[1]))

    p = Csize_t[0]
    cudnnGetRNNTrainingReserveSize(h, rnndesc, length(xs), xdesc, p)
    reservespace = CuArray{Int8}(Int(p[1]))

    cudnnRNNForwardTraining(h, rnndesc, length(xs), xdesc, x, hxdesc, hx, cxdesc, cx,
        wdesc, w, ydesc, y, hydesc, hy, cydesc, cy,
        workspace, length(workspace), reservespace, length(reservespace))
end

function getnlinlayers(mode)
    mode == CUDNN_RNN_RELU && return 2
    mode == CUDNN_RNN_TANH && return 2
    mode == CUDNN_RNN_LSTM && return 8
    mode == CUDNN_GRU && return 6
    throw("Invalid mode: $mode.")
end
