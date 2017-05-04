module CUJulia

using Base.LinAlg.BLAS

if is_windows()
    const libcuda = Libdl.find_library(["nvcuda"])
else
    const libcuda = Libdl.find_library(["libcuda"])
end
isempty(libcuda) && throw("CUDA driver library cannot be found.")

function check_curesult(status)
    status == CUDA_SUCCESS && return
    p = Ptr{UInt8}[0]
    cuGetErrorString(status, p)
    throw(unsafe_string(p[1]))
end

p = Cint[0]
ccall((:cuDriverGetVersion,libcuda), UInt32, (Ptr{Cint},), p)
const driver_version = Int(p[1])
info("CUDA driver version: $driver_version")

const major = div(driver_version, 1000)
const minor = div(driver_version - major*1000, 10)
include("../lib/$(major)$(minor)/libcuda.jl")
include("../lib/$(major)$(minor)/libcuda_types.jl")

include("device.jl")
include("function.jl")
initctx()
infodevices()

ctype(::Type{Int64}) = :int
ctype(::Type{Float32}) = :float
ctype(::Type{Float64}) = :double

include("pointer.jl")
include("array.jl")
include("subarray.jl")
#include("math.jl")
#include("broadcast.jl")
#include("reducedim.jl")

include("Interop.jl")
include("NVRTC.jl")
include("CUBLAS.jl")
include("CUDNN.jl")

end
