function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

 
all_theta = zeros(num_labels, n + 1); % zeros matrix size from 10 to n+1 

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Goal:  train num_labels anf logistic regression classifiers with 
%        regularization parameter lambda. 


%        theta(:) will return a column vector.
%
% Note: we will use y == i (where i=1 in our for loop)
%       to obtain a vector of 1's and 0's that tell us
%       whether the ground truth is true/false for this class.
%
% Note: We will be utilizing the program fmincg.m to optimize the cost
%       function. 
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%

% Set initial theta
initial_theta = zeros(n+1, 1);
% utilize optimset and set our options for fminunc 
options = optimset('GradObj', 'on', 'MaxIter', 50);
% For loop for oneVsAll  classification
for i=1:num_labels
  [theta] = fmincg (@(t) (lrCostFunction(t, X, (y==i), lambda)),...
  initial_theta, options); 
  all_theta(i,:) = theta;
  
end










% =========================================================================


end
