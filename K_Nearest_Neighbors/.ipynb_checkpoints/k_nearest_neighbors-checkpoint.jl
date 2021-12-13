using DelimitedFiles
using Statistics
using LinearAlgebra
using Plots
using Distances
using RDatasets: dataset
using CategoricalArrays
using StatsBase
using JSON


function data_read(str::String)
    data = readdlm(str, ',', Float64, '\n');
    return data
end;

function k_neighbors_regressor(X,y, x_hat, k)
    dist = colwise(Euclidean(), transpose(X),x_hat)
    k_nearest = y[sortperm(dist)[1:k]]
    return mean(k_nearest) 
end;

function weighted_k_neighbors_regressor(X,y, x_hat, k)
    dist = colwise(SqEuclidean(), transpose(X),x_hat)
    k_nearest = y[sortperm(dist)[1:k]]
    weights = reverse(dist[sortperm(dist)[1:k]]/sum(dist[sortperm(dist)[1:k]]))
    return transpose(k_nearest) * weights
end;


function k_neighbors_classifier(X,y, x_hat, k)
    dist = colwise(Euclidean(), transpose(X),x_hat)
    k_nearest = y[sortperm(dist)[1:k]]
    d = countmap(k_nearest)
    return Dict(i=>j/k for (i,j) in d)
end;


println("Regression")
println("Reading data...")
data = data_read("../data/data2.txt");

X = data[:,1:end-1];
y = data[:,end];

println("Predicting with new values [2200, 5] and 5 nearest neighbors...")
predicted = k_neighbors_regressor(X,y,[2200, 5], 5)
predicted_weighted = weighted_k_neighbors_regressor(X,y,[2200, 5], 5)

println("Predicted value = $predicted")
println("Predicted value with weights = $predicted_weighted")



println("\n\nClassification")
println("Reading data...")
iris = dataset("datasets", "iris");

X = Matrix(iris[:,1:end-1]);
y = iris[:,end];


println("Predicting with new values [1,2,3,4] and 5 nearest neighbors...")
predicted = json(k_neighbors_classifier(X,y,[1,2,3,4], 5),4)
println("Predicted values = $predicted")