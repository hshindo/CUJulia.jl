function addtensor!(x, y)
    h = handle(x)
    T = eltype(x)
    xdesc = TensorDesc(x)
    cudnnAddTensor(h, T[1], xdesc, x, T[0], xdesc, y)
    y
end
