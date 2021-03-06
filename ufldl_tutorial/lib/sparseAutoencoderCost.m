function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
    lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64)
% hiddenSize: the number of hidden units (probably 25)
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example.

% The input theta is a vector (because minFunc expects the parameters to be a vector).
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
% follows the notation convention of the lecture notes.

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values).
% Here, we initialize them to zeros.
cost = 0;
W1grad = zeros(size(W1));
W2grad = zeros(size(W2));
b1grad = zeros(size(b1));
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b)
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
%
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2.
%

%
% Feedforward Pass ---
%

% Computing the activations of each unit in each layer
A1 = data;
A2 = sigmoid(bsxfun(@plus, W1*data, b1));
A3 = sigmoid(bsxfun(@plus, W2*A2, b2));

% Squared Error
errorMatrix = A3 - data;
sumOfSquares = 0.5*sum(abs(errorMatrix).^2);

% Weight Decay
weightDecay = sum([W1(:).^2; W2(:).^2]);

% Sparsity Parameter
rhoCap = mean(A2,2);
klDiver = sparsityParam*log(sparsityParam./rhoCap) + (1-sparsityParam)*log((1-sparsityParam)./(1-rhoCap));
sparsityTerm = sum(klDiver);

% Cost Function
cost = mean(sumOfSquares) + (lambda/2)*weightDecay + beta*sparsityTerm;

%
% Backpropagation ---
%

% For the output layer
Del3 = -(data - A3).*A3.*(1-A3);

% For the other layers
sparseDerivative = -(sparsityParam./rhoCap) + ((1-sparsityParam)./(1-rhoCap));
sparseDerivative = repmat(sparseDerivative,1,size(data,2));
Del2 = (W2'*Del3 + beta*sparseDerivative).*A2.*(1-A2);
Del1 = (W1'*Del2).*A1.*(1-A1);

% Desired Partial Derivatives
W1grad = Del2*A1';
W2grad = Del3*A2';
b1grad = sum(Del2,2);
b2grad = sum(Del3,2);

% Taking the average
W1grad = W1grad./size(data,2) + lambda*W1;
W2grad = W2grad./size(data,2) + lambda*W2;
b1grad = b1grad./size(data,2);
b2grad = b2grad./size(data,2);

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end