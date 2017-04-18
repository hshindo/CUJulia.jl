export
    # cudnnSoftmaxMode_t
    CUDNN_SOFTMAX_MODE_INSTANCE,
    CUDNN_SOFTMAX_MODE_CHANNEL,

    # cudnnSoftmaxAlgorithm_t
    CUDNN_SOFTMAX_FAST,
    CUDNN_SOFTMAX_ACCURATE,
    CUDNN_SOFTMAX_LOG

function softmax(x; algo=CUDNN_SOFTMAX_ACCURATE, mode=CUDNN_SOFTMAX_MODE_CHANNEL)
    h = handle(x)
    dims1 = ntuple(_ -> 1, 4-ndims(x))
    x4d = reshape(x, dims1..., size(x)...)
    xdesc = TensorDesc(x4d)
    y = similar(x)
    cudnnSoftmaxForward(h, algo, mode, T[1], xdesc, x, T[0], xdesc, y)

    function backward!(dy, dx)
        isvoid(dx) && return
        cudnnSoftmaxBackward(h, algo, mode, T[1], xdesc, y, xdesc, dy, T[1], xdesc, dx)
    end
    y, backward!
end

logsoftmax(x) = softmax(x, algo=CUDNN_SOFTMAX_LOG)
