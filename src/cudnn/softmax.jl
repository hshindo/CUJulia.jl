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
    reshape(x)

    pad = 4 - N
    xdesc = TensorDesc(x)
    y = similar(x)
    cudnnSoftmaxForward(h, algo, mode, T[1], xdesc, x, T[0], xdesc, y)

    function backward!(dy, dx)
        isvoid(dx) && return
        cudnnSoftmaxBackward(h, algo, mode, T[1], xdesc, y, xdesc, dy, T[1], xdesc, dx)
    end
    y, backward!
end

logsoftmax(x) = softmax(x, algo=CUDNN_SOFTMAX_LOG)

N = ndims(x)
@assert N <= 4
csize = Cint[1, 1, 1, 1]
cstrides = Cint[1, 1, 1, 1]
st = strides(x)
for i = 1:N
    csize[4-i-pad+1] = size(x,i)
    cstrides[4-i-pad+1] = st[i]
end
