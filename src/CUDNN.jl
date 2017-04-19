module CUDNN

using ..CUJulia

if is_windows()
    const libcudnn = Libdl.find_library(["cudnn64_6","cudnn64_5"])
else
    const libcudnn = Libdl.find_library(["libcudnn"])
end
isempty(libcudnn) && throw("CUDNN library cannot be found.")

const version = ccall((:cudnnGetVersion,libcudnn),Cint,())
const major = div(version, 1000)
const minor = div(version - major*1000, 100)

info("CUDNN version: $(version)")
include("../lib/$(CUJulia.major)$(CUJulia.minor)/libcudnn$(major)$(minor).jl")
include("../lib/$(CUJulia.major)$(CUJulia.minor)/libcudnn$(major)$(minor)_types.jl")

function checkstatus(status)
    status == CUDNN_STATUS_SUCCESS && return
    throw(unsafe_string(cudnnGetErrorString(status)))
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

include("cudnn/activation.jl")
#include("batchnorm.jl")
include("cudnn/convolution.jl")
include("cudnn/dropout.jl")
include("cudnn/filter.jl")
include("cudnn/math.jl")
include("cudnn/pooling.jl")
#include("cudnn/rnn.jl")
include("cudnn/softmax.jl")
include("cudnn/tensor.jl")

end
