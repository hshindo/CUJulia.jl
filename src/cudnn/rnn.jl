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

function rnn(hiddensize::Int, numlayers::Int, droprate::Float64, direction, mode, seqlength::Int,
    xs::Vector;
    inputmode=CUDNN_LINEAR_INPUT)
    xdims, x::CuArray{T}, hx::CuArray, cx::CuArray)

    rnndesc = RNNDesc()
    dropdesc = DropoutDesc()
    cudnnSetRNNDescriptor(rnndesc, hiddensize, numlayers, dropdesc, inputmode,
        direction, mode, datatype(T))

    h = handle(xs[1])
    xdesc = [TensorDesc(xs[i]) for i=1:length(xs)]
    p = Csize_t[0]
    cudnnGetRNNWorkspaceSize(h, rnndesc, seqlength, xdesc, p)
    workspace = CuArray{Int8}(Int(p[1]))

    p = Csize_t[0]
    cudnnGetRNNTrainingReserveSize(h, rnndesc, seqlength, xdesc, p)
    reservesize = CuArray{Int8}(Int(p[1]))

    p = Csize_t[0]
    cudnnGetRNNParamsSize(h, rnndesc, xdesc, p, datatype(T))
    paramsize = CuArray{Int8}(Int(p[1]))

    linmatdesc = FilterDesc() # ?
    p = Ptr{Void}[0]
    cudnnGetRNNLinLayerMatrixParams(h, rnndesc, 0, xdesc, wdesc, w,
        0, linmatdesc, p)
    linmat = p[1]

    bdesc = FilterDesc()
    p = Ptr{Void}[0]
    cudnnGetRNNLinLayerBiasParams(h, rnndesc, 0, xdesc[1], wdesc, w,
        0, bdesc, b_p)
    b = p[1]



    hxdesc = TensorDesc(hx)
    cxdesc = TensorDesc(cx)

    y = similar(x)
    hy = similar(hx)
    cy = similar(cx)
    ydescs = similar(xdescs)
    for i=1:length(xdims)
        ydescs[i] = TensorDesc(CuArray(T,xdims[i]))
    end
    hydesc = TensorDesc(hy)
    cydesc = TensorDesc(cy)

    h = handle(x)
    rnndesc, dropdesc, dropstate = rnn_desc(x, size(hx,2), size(hx,4), input_t,
        dir_t, net_t, droprate, seed)
    wsize_p = Cint[0]
    cudnnGetRNNParamsSize(h, rnndesc, xdesc, wsize_p, datatype(T))
    wsize = wsize_p[1]
    w = curand(T, 1, 1, 1, Int(wsize/(T.size)))
    wdesc = FilterDesc(w)

    worksize_p = Cint[0]
    cudnnGetRNNWorkspaceSize(h, rnndesc, Cint(length(xdescs)), xdescs, worksize_p)
    worksize = worksize_p[1]
    workspace = CuArray(Int8, Int(worksize))

    trainsize_p = Cint[0]
    cudnnGetRNNTrainingReserveSize(h, rnndesc, Cint(length(xdescs)), xdescs, trainsize_p)
    trainsize = trainsize_p[1]
    trainspace = CuArray(Int8, Int(trainsize))

    mdesc_p = Ptr{Void}[0]
    cudnnCreateFilterDescriptor(mdesc_p)
    mdesc = mdesc_p[1]
    m_p = Ptr{Void}[0]
    cudnnGetRNNLinLayerMatrixParams(h, rnndesc, Cint(0), xdesc, wdesc, w,
        Cint(0), mdesc, m_p)
    m = m_p[1]

    bdesc_p = Ptr{Void}[0]
    cudnnCreateFilterDescriptor(bdesc_p)
    bdesc = bdesc_p[1]
    b_p = Ptr{Void}[0]
    cudnnGetRNNLinLayerBiasParams(h, rnndesc, Cint(0), xdesc, wdesc, w,
        Cint(0), bdesc, b_p)
    b = b_p[1]

    cudnnRNNForwardTraining(h, rnndesc, Cint(length(xdescs)), xdescs, x, hxdesc,
        hx, cxdesc, cx, wdesc, w, ydescs, y, hydesc, hy, cydesc, cy, workspace,
        worksize, trainspace, trainsize)

    function backward!()

    end
    w, y, hy, cy, dropdesc, dropstate
end
