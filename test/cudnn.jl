using CUJulia.CUDNN

@testset "functions" for i = 1:5

T = Float32

# activation
x = CuArray(rand(T,10,5,4,3))
for mode in (CUDNN_ACTIVATION_SIGMOID,CUDNN_ACTIVATION_RELU)
    y, df = activation(mode, x)
    @test true
end

# convolution
x = CuArray(rand(T,5,4,3,2))
w = CuArray(rand(T,2,2,3,4))
b = CuArray(rand(T,5,4))
convolution(x, w, b, (0,0), (1,1))
@test true

# dropout
x = CuArray(rand(T,10,5,4,3))
y, df = dropout(x, 0.5)
@test true

# pooling
x = CuArray(rand(T,5,4,3,10))
y, df = pooling(CUDNN_POOLING_MAX, (2,2), (1,1), (2,2), x)
@test true

# softmax
x = CuArray(rand(T,5,4,3,10))
y, df = softmax(x)
y, df = logsoftmax(x)

end
