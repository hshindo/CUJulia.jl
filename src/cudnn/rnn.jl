export
    # cudnnRNNMode_t
    CUDNN_RNN_RELU,
    CUDNN_RNN_TANH,
    CUDNN_LSTM, CUDNN_GRU,

    # cudnnRNNInputMode_t
    CUDNN_LINEAR_INPUT,
    CUDNN_SKIP_INPUT,

    # cudnnDirectionMode_t
    CUDNN_UNIDIRECTIONAL,
    CUDNN_BIDIRECTIONAL

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
