function addtensor!{T,N}(A::CuArray{T,N}, C::CuArray{T,N})
    h = handle(A)
    adesc = TensorDesc(A)
    cudnnAddTensor(h, T[1], adesc, A, T[1], adesc, C)
    C
end
