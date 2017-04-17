function argmax(x::Array, dim::Int)
    _, index = findmax(x, dim)
    ind2sub(size(x), vec(index))[dim]
end

T = Float32
x = rand(T, 100, 100)
cux = CuArray{T}(x)
@test argmax(x) == Array(argmax(cux))
