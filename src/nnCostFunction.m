function [J, grad] = nnCostFunction(nn_params,input_layer_size, ...
                                   hidden_layer_size,num_labels, X, y, lambda)
                                   
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Number of examples
m = size(X, 1);
         
% Inicialize the variables that the function will output
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


ymulti=zeros(num_labels,m);
for i=1:m
  ymulti(y(i),i)=1;
end

% Part 1: Feedforward the neural network and return the cost in the variable J. 

for t=1:m

  a1 = [1; X(t,:)'];
  z2=Theta1*a1;
  a2 = 1.0 ./ (1.0 + exp(-z2));
  a2= [1; a2];

  z3=Theta2*a2;
  a3 = 1.0 ./ (1.0 + exp(-z3));

  % Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad.  

  delta3=a3-ymulti(:,t);
  delta2=Theta2'*delta3.*[1; sigmoidGradient(z2)];
  
  Theta1_grad=Theta1_grad+delta2(2:end)*a1';
  Theta2_grad=Theta2_grad +delta3*(a2)';
  
%Computing the cost for each example
h=a3;
y=ymulti(:,t);

%No regularization
JJ(t,:)=-y.*log(h)-(1-y).*log(1-h);

end

J=sum(JJ(:))/m;

Theta1_grad=Theta1_grad./m;
Theta2_grad=Theta2_grad./m;

%Regularization
% Part 3: Implement regularization with the cost function and gradients.
%The gradients for  the regularization are computed separately and then 
%added Theta1_grad  and Theta2_grad from Part 2.

theta1=Theta1(:,2:end);
theta2=Theta2(:,2:end);
reg=(lambda/(2*m))*(sum(theta1(:).^2)+sum(theta2(:).^2));
J=J+reg;

Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+(lambda/m).*theta1;
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+(lambda/m).*theta2;

% =========================================================================
% Unroll gradients
%   The returned variable grad is an "unrolled" vector of the
%   partial derivatives of the neural network.

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
