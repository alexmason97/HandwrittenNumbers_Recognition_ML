function [J, grad] = lrCostFunction(theta, X, y, lambda)
%   Linear Regression COST FUNCTION to compute cost and gradient 
%   for logistic regression with regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

% initialize necessary variables for cost equation and gradient descent 
J = 0;
grad = zeros(size(theta)); % Zeros matrix of size theta

% Goals: To compute the cost of a particular choice of theta.
%       Set J to the cost function.
%       Compute the partial derivatives and set grad to the partial
%       derivatives of the cost w.r.t. each parameter in theta.
%
%      Note: we can vectorize these equations and make use of our
%      Sigmoid function as follows: sigmoid(X * theta) from sigmoid.m
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. 
%
%

%using vectorization approach we can compute this in a few simple lines to 
%represent similar math equations for our linear regression cost function
shift_theta = theta(2:size(theta)); % row vector of size theta 
theta_reg = [0;shift_theta]; % matrix of all theta values
J = (1/m)*(-y'*log(sigmoid(X*theta))-(1-y')*log(1-sigmoid(X*theta)))+...
(lambda/(2*m))*theta_reg'*theta_reg; #cost function
grad = (1/m)*(X'*(sigmoid(X*theta)-y)+lambda*theta_reg);
% gradient descent equation 

% =============================================================

grad = grad(:); # run gradient descent for each column

end
