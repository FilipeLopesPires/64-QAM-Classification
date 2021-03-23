function g = sigmoidGradient(z)
% SIGMOIDGRADIENT returns the gradient of the sigmoid function 
% evaluated at z.
%
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, the function returns
%   the gradient for each element.
%
    g = zeros(size(z));
    h = 1 ./ (1 + exp(-z));
    g = h .* (1-h);
end
