using Base.Test
using CUJulia

path = joinpath(dirname(@__FILE__), "cudnn.jl")
include(path)
