using DelimitedFiles
using Statistics
using LinearAlgebra
using Plots

"""
Script of Linear Regression model. Reads data inputs and predicts a new value.
This script is just an example of the functionality of the LR model. In the attatched PDF file
there is comparissons between different learning rates and the plots of how the error and the functions change.
"""

function data_read(str::String)
    """
    Function to read de data from a CSV type file.
    The file has to hav only numbers with no letters and the delimiter has to be a ",".
    
    Input:
        str (String):           String with the name of the file to read.
                                Path has to be the same as this script.
    
    Returns:
        data (Matrix{Float64}): Matrix with data with floats.
    """
    data = readdlm(str, ',', Float64, '\n');
    return data
end


function normalization(x::Vector{Float64})
    """
    Normalize de values of a vector between de values [-1,1]. This is done by resting the mean and
    dividing by the standard deviation.
    
    Inputs:
        x (Vector{Float64}): vector
    """
    return (x.-mean(x))/std(x)
end


mutable struct LR_Problem
    X::Matrix{Float64}       # Normalized data
    y::Vector{Float64}       # Normalized vector
    X_mean::Vector{Float64}  # Means of X matrix per column
    X_std::Vector{Float64}   # STD of X matrix per colum
    y_mean::Float64          # Mean of y
    y_std::Float64           # STD of y
    θ::Matrix{Float64}       # Weights and biases
    θ_hist::Matrix{Float64}
    α::Float64               # Learning rate
    tol::Float64             # Tolerance of error for convergence
    
    function LR_Problem(data::Matrix{Float64}; θ = Matrix{Float64}(undef, 0,0)::Matrix{Float64}, α = 0.01::Float64, tol = 1e-8::Float64)
        """
        Constructor of a Linear Regression problem. It takes the data from the problem, the learning
        rate and the tolerance for convergence to get all the information of the problem:
            1. It normalizes the data, and saves the means and std for further predictions.
            2. It splits X and y sets of the data.
            3. Adds column of 1 to the beginning of X for the bias.
            4. Initializes random array of weights and biases.
        
        Input:
            data (Matrix{Float64}): Matrix of data with X and y as the last column.
            θ (Matrix{Float64}): starting weights and biases. If the size is not correct, it will automatically make a new one.
            α (Float64, optional): Learning rate of the problem. Default of 0.01.
            tol (Float64, optional): Tolerance for convergence. Default of 1e-8.
        """
        means = mapslices(mean, data; dims = 1)
        stds = mapslices(std, data; dims = 1)
        normalized_data = mapslices(normalization, data; dims = 1)
        X = [ones(size(data)[1]) normalized_data[:,1:end-1]]
        y = normalized_data[:,end]
        if size(θ)[2] > 1
            θ = transpose(θ)
        end
        if size(θ)[1] != size(X)[2]
            θ = rand(size(X)[2],1)
        end
        θ_hist = transpose(θ)
        new(X, y, means[1:end-1], stds[1:end-1], means[end], stds[end], θ, θ_hist, α, tol)
    end
end

function compute_cost(prob::LR_Problem)
    """
    Computes the cost of the aproximation of Xθ to y using the Least Mean Squares algorithm.
    
    Inputs:
        prob (LR_Problem): Struct with all of the information of the problem.
    
    Outputs:
        returns (Float64): LMS cost of the aproximation.
    """
    m = size(prob.X)[1]
    return norm(prob.X*prob.θ - prob.y, 2)^2/2m
end

function compute_gradients(prob::LR_Problem)
    """
    Computes the gradients of θ for an aproximation of y using the data X, assuming the LMS 
    algorithm is the cost function.
    
    Inputs:
        prob (LR_Problem): Struct with all of the information of the problem.
    
    Outputs:
        returns (Matrix{Float64}) = nx1 matrix with the gradients.
    """
    m = size(prob.X)[1]
    return 1/m * transpose(prob.X) * (prob.X * prob.θ - prob.y)
end

function train!(prob::LR_Problem; max_iter  = 100000::Int64)
    """
    Train a linear regression problem. It changes overwrites the weights and biases of the problem.
    
    Inputs:
        prob (LR_Problem): Linear regression problem.
        max_iiter (Int64): Maximum number of iterations to train.
    """
    iter = 0
    # Train while it has not converged or reach 10000 iterations
    while norm(compute_gradients(prob),2)>prob.tol && iter<=max_iter
        prob.θ -= prob.α * compute_gradients(prob)
        prob.θ_hist = vcat(prob.θ_hist, transpose(prob.θ))
    end
end

function predict(prob::LR_Problem, x::Vector{Float64})
    """
    Function to predict based on a trained LR model and values x.
    
    Inputs:
        prob (LR_Problem): trained model.
        x (Vector{Float64}): values to predict. Given in vector form, i.e: [1,2,3].
    
    Outputs:
        y (Float64): predicted value.
    """
    x = (x-prob.X_mean)./prob.X_std # normalize the vector
    x = transpose(vcat(1, x)) # add first column of 1 and transpose
    y = (x * prob.θ)[1] # predict
    y = y * prob.y_std + prob.y_mean # de-normalize
    return y
end

function main()
    println("Reading data...")
    data = data_read("../data/data2.txt")
    prob = LR_Problem(data)
    println("Initializing training...")
    train!(prob)
    error = compute_cost(prob)
    println("Error = $error")
    println("Sample predictions of values [2200.0, 0]")
    predicted = predict(prob, [2200.0, 2.0])
    println("Predicted value = $predicted")
end

main()
