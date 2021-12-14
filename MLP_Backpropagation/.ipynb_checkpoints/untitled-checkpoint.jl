using DelimitedFiles
using Statistics
using LinearAlgebra
using Plots

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

