using LinearAlgebra

function append_ones(mat::Matrix{Float64})
    """
    Function to append a row of ones at the end of a matrix. This is for the biases in a
    neuron.
    
    Input:
        mat(Matrix{Float64}): matrix
    
    Output:
        Matrix with row of ones at the end.
    """
    return vcat(mat, ones(1, size(mat, 2)))
end;


abstract type ActivationFunction end;

##### LINEAR ACTIVATION FUNCTION #####
function linear_activation(params::Matrix{Float64}, input::Matrix{Float64})
    return params * append_ones(input)
end;

struct LinearActivation <: ActivationFunction
    func::Function
    derivative::Function
    function LinearActivation()
        new(linear_activation, x -> 1)
    end
end;

##### SIGMOID ACTIVATION FUNCTION #####
function sigmoid_activation(params::Matrix{Float64}, input::Matrix{Float64})
    return 1.0 ./ (1 .+ exp.(-params * append_ones(input)))
end;

function sigmoid_derivative(input::Matrix{Float64})
    return input .* (-input .+ 1)
end;

struct Sigmoid_Activation <: ActivationFunction
    func::Function
    derivative::Function
    function Sigmoid_Activation()
        new(sigmoid_activation, sigmoid_derivative)
    end
end;

##### RELU ACTIVATION FUNCTION #####
function relu_activation(params::Matrix{Float64}, input::Matrix{Float64})
    z = params * append_ones(input)
    return z .* (z .> 0)
end;

function relu_derivative(input::Matrix{Float64})
    return Int.(input.>0)
end;

struct ReLU_Activation <: ActivationFunction
    func::Function
    derivative::Function
    function ReLU_Activation()
        new(relu_activation, relu_derivative)
    end
end;

##### TANH ACTIVATION FUNCTION #####
function tanh_activation(params::Matrix{Float64}, input::Matrix{Float64})
    return tanh.(params*append_ones(input))
end;

function tanh_derivative(input::Matrix{Float64})
    return sech.(input).^2
end;

struct Tanh_Activation <: ActivationFunction
    func::Function
    derivative::Function
    function Tanh_Activation()
        new(tanh_activation, tanh_derivative)
    end
end;

##### SOFTMAX ACTIVATION FUNCTION #####
function softmax_activation(params::Matrix{Float64}, input::Matrix{Float64})
    """
    Stable form of softmax function. Other form has numerical stability issues so
    this form is used to fix that.
    """
    exps = input.-maximum(input, dims=1)
    return exp.(exps)./sum(exp.(exps), dims=1)
end;

struct Softmax_Activation <: ActivationFunction
    func::Function
    function Softmax_Activation()
        new(softmax_activation)
    end
end;