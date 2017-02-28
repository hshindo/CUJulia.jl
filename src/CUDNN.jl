module CUDNN

using ..CUJulia

if is_windows()
    const libcudnn = Libdl.find_library(["cudnn64_5"])
else
    const libcudnn = Libdl.find_library(["libcudnn"])
end
isempty(libcudnn) && throw("CUDNN library cannot be found.")

const version = ccall((:cudnnGetVersion,libcudnn),Cint,())
const major = div(version, 1000)
const minor = div(version - major*1000, 100)

info("CUDNN version: $(version)")
include("../lib/$(CUJulia.major).$(CUJulia.minor)/libcudnn$(major)$(minor).jl")
include("../lib/$(CUJulia.major).$(CUJulia.minor)/libcudnn$(major)$(minor)_types.jl")

function checkstatus(status)
    status == CUDNN_STATUS_SUCCESS && return
    throw(bytestring(cudnnGetErrorString(status)))
end

datatype(::Type{Float32}) = CUDNN_DATA_FLOAT
datatype(::Type{Float64}) = CUDNN_DATA_DOUBLE
datatype(::Type{Float16}) = CUDNN_DATA_HALF

const handles = Ptr{Void}[]
function handle(x)
    dev = device(x) + 1
    while dev > length(handles)
        p = Ptr{Void}[0]
        cudnnCreate(p)
        push!(handles, p[1])
    end
    handles[dev]
end
atexit(() -> foreach(cudnnDestroy, handles))

export
    cudnnActivationForward, cudnnActivationBackward,
    cudnnSoftmaxForward, cudnnSoftmaxBackward

export CUDNN_ACTIVATION_SIGMOID, CUDNN_ACTIVATION_RELU, CUDNN_ACTIVATION_TANH, CUDNN_ACTIVATION_CLIPPED_RELU
export CUDNN_BATCHNORM_PER_ACTIVATION, CUDNN_BATCHNORM_SPATIAL
export CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION
export CUDNN_SOFTMAX_MODE_INSTANCE, CUDNN_SOFTMAX_MODE_CHANNEL # mode
export CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_LOG # algorithm

type TensorDesc
    ptr::Ptr{Void}

    function TensorDesc{T,N}(x::CuArray{T,N}; pad=0)
        @assert N <= 4
        csize = Cint[1, 1, 1, 1]
        cstrides = Cint[1, 1, 1, 1]
        st = strides(x)
        for i = 1:N
            csize[4-i-pad+1] = size(x,i)
            cstrides[4-i-pad+1] = st[i]
        end
        p = Ptr{Void}[0]
        cudnnCreateTensorDescriptor(p)
        cudnnSetTensorNdDescriptor(p[1], datatype(T), 4, csize, cstrides)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyTensorDescriptor)
        desc
    end
end

type ActivationDesc
    ptr::Ptr{Void}

    function ActivationDesc(mode::UInt32; relu_nanopt=CUDNN_NOT_PROPAGATE_NAN, relu_ceiling=1.0)
        p = Ptr{Void}[0]
        cudnnCreateActivationDescriptor(p)
        cudnnSetActivationDescriptor(p[1], mode, relu_nanopt, relu_ceiling)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyActivationDescriptor)
        desc
    end
end

type ConvDesc
    ptr::Ptr{Void}

    function ConvDesc{T,N}(::Type{T}, pads::NTuple{N,Int}, strides; mode=CUDNN_CROSS_CORRELATION)
        p = Ptr{Void}[0]
        cudnnCreateConvolutionDescriptor(p)
        c_pads = Cint[pads[i] for i=N:-1:1]
        c_strides = Cint[strides[i] for i=N:-1:1]
        c_upscale = fill(Cint(1), N)
        cudnnSetConvolutionNdDescriptor(p[1], N, c_pads, c_strides, c_upscale, mode, datatype(T))
        desc = new(p[1])
        finalizer(desc, cudnnDestroyConvolutionDescriptor)
        desc
    end
end

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

type FilterDesc
    ptr::Ptr{Void}

    function FilterDesc{T,N}(x::CuArray{T,N})
        csize = Cint[size(x,i) for i=N:-1:1]
        p = Ptr{Void}[0]
        cudnnCreateFilterDescriptor(p)
        cudnnSetFilterNdDescriptor(p[1], datatype(T), format, N, csize)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyFilterDescriptor)
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::TensorDesc) = desc.ptr
Base.unsafe_convert(::Type{Ptr{Void}}, desc::ActivationDesc) = desc.ptr
Base.unsafe_convert(::Type{Ptr{Void}}, desc::ConvDesc) = desc.ptr
Base.unsafe_convert(::Type{Ptr{Void}}, desc::DropoutDesc) = desc.ptr
Base.unsafe_convert(::Type{Ptr{Void}}, desc::FilterDesc) = desc.ptr

end
