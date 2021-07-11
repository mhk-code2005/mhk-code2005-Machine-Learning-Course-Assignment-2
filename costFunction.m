function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
a=1/m;
d=0;

for I=1:m;
d+=-(y(I)*log(sigmoid(X(I,:)*theta)))-(1-y(I))*log(1-sigmoid(X(I,:)*theta));

end
J=a*d;

for I=1:size(theta);
grad(I)= (1/m)*(X(:,I)')*(sigmoid(X*theta)-y);
end
% =============================================================

end
