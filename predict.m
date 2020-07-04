function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% initialize variables needed for prediction
p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];

% Goal: Make predictions using learned neural network. We will set p to a 
%       vector containing labels between 1 to num_labels.
%
% forward propogation setup algorithm

a1 = X;

z2 = a1 * Theta1';
a2 = [ones(m,1), sigmoid(z2)];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

[~, indices] = max(a3, [], 2);
p = indices;

% =========================================================================


end
