using DelimitedFiles
using Statistics
using LinearAlgebra
using Plots
using RDatasets: dataset
using CategoricalArrays

iris = dataset("datasets", "iris");

X = Matrix(iris[:,1:end-1]);
y = iris[:,end];

logistic(z) = 1/(1+exp(-z));

function normalization(x::Vector{Float64})
    return (x.-mean(x))/std(x)
end;

mutable struct LR_Problem
    X::Matrix{Float64}       # Normalized data
    y::Vector{Int}       # Normalized vector
    X_mean::Matrix{Float64}  # Means of X matrix per column
    X_std::Matrix{Float64}   # STD of X matrix per colum
    theta::Matrix{Float64}       # Weights and biases
    alpha::Float64               # Learning rate
    tol::Float64             # Tolerance of error for convergence
    
    function LR_Problem(X::Matrix{Float64}, y::CategoricalArray; alpha = 0.01::Float64, tol = 1e-2::Float64)
        X_mean = mapslices(mean, X; dims = 1)
        X_std = mapslices(std, X; dims = 1)
        X = mapslices(normalization, X; dims = 1)
        X = [ones(size(X)[1]) X];
        y = levelcode.(y)
        num_cat = size(levels(y))[1]
        if num_cat == 2
            num_cat = 1
        end
        
        theta = rand(size(X)[2],num_cat)
        new(X, y, X_mean, X_std, theta, alpha, tol)
    end
end;  

function compute_cost(prob::LR_Problem)
    m = size(prob.X)[1]
    h = logistic.(prob.X * prob.theta)
    y_h = -(prob.y'.==unique(prob.y))
    return diag(y_h * log.(h) - (y_h.+1) * log.(-h.+1))./m
end;

function compute_gradients(prob::LR_Problem)
    m = size(prob.X)[1]
    h = logistic.(prob.X * prob.theta)
    y_h = prob.y.==unique(prob.y)'
    return transpose(prob.X) * (h-(prob.y.==unique(prob.y)'))/m
end;

function train!(prob::LR_Problem; max_iter  = 10000::Int64)
    iter = 0
    # Train while it has not converged or reach 10000 iterations
    gradients = compute_gradients(prob)
    while norm(gradients,2)>prob.tol && iter<=max_iter
        prob.theta -= prob.alpha * gradients
        gradients = compute_gradients(prob)
    end
    return err_hist
end;

function predict(prob::LR_Problem, x::Matrix{Float64})
    x = (x-prob.X_mean)./prob.X_std # normalize the vector
    x = hcat(1, x) # add first column of 1
    y = logistic.(x * prob.theta) # predict
    return argmax(y)[2]
end;

prob = LR_Problem(X,y);

train!(prob);

x = [6 3 4.8 1.8];
predict(prob, x)
