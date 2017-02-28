export CUDNN_RNN_RELU, CUDNN_RNN_TANH, CUDNN_LSTM, CUDNN_GRU
export CUDNN_LINEAR_INPUT, CUDNN_SKIP_INPUT # cudnnRNNInputMode_t


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
