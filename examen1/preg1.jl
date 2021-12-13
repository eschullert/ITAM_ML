using Statistics
using LinearAlgebra


# making the data
f(x) = 3+x^3;

y = range(-2, stop=2, length = 200);
X = [f(x) for x in y_test];
y = [a for a in y];

#adding col of ones
X = [ones(size(X)[1]) X];

#error function and gradient
m = size(X)[1]
E(t, g) = (norm(X*t - y, 2)^2 + g*norm(t)^2) /2m;
grad_E(t, g) = (t * g + transpose(X) * (X*t-y))/m;


alpha = 0.1
for gamma = 1:10
    #weights and biases
    theta = rand(size(X)[2],1);
    while norm(grad_E(theta, gamma))>10^(-5)
        theta -= alpha*grad_E(theta, gamma)
    end
    println("gamma = ", gamma, ", error = ", E(theta, gamma))
end



