function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)

% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.

% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example.

% Your code should produce the prediction matrix
% pred, where pred(i) is argmax_c P(y(c) | x(i)).

%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start
%                from 1.

% Input Layer 1
A1 = data;

% Hidden Layer 2 Activation
W1 = stack{1}.w;
b1 = stack{1}.b;
A2 = sigmoid(bsxfun(@plus, W1*A1, b1));

% Hidden Layer 3 Activation
W2 = stack{2}.w;
b2 = stack{2}.b;
A3 = sigmoid(bsxfun(@plus, W2*A2, b2));

% Softmax Layer 4 Activation
Mn = softmaxTheta*A3;
Mn = bsxfun(@minus, Mn, max(Mn));      % Prevent Overflow
probability = exp(Mn);
normalizationFactor = sum(probability);
normalizedProbability = probability./repmat(normalizationFactor,numClasses,1);

% Prediction
[val pred] = max(normalizedProbability);

% -----------------------------------------------------------

end