function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));

%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

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

% Calculating the Softmax Gradient
errorTerm = groundTruth - normalizedProbability;
softmaxThetaGrad = ((-1/M)*(errorTerm*A3')) + lambda*softmaxTheta;

% For all the layers
Del3 = -(softmaxTheta'*(groundTruth - normalizedProbability)).*A3.*(1-A3);
Del2 = (W2'*Del3).*A2.*(1-A2);

% Desired Partial Derivatives
W1grad = Del2*A1';
W2grad = Del3*A2';
b1grad = sum(Del2,2);
b2grad = sum(Del3,2);

% Taking the average
W1grad = W1grad./size(data,2);
W2grad = W2grad./size(data,2);
b1grad = b1grad./size(data,2);
b2grad = b2grad./size(data,2);

% Calculating the Cost
logProbability = log(normalizedProbability);
withIndicator = groundTruth.*logProbability;
squaredTheta = softmaxTheta.^2;            % Weight Decay Term
cost = (-1/M)*sum(withIndicator(:)) + (lambda/2)*sum(squaredTheta(:));

% Storing the Hidden Layer Gradients
stackgrad{1}.w = W1grad;
stackgrad{2}.w = W2grad;
stackgrad{1}.b = b1grad;
stackgrad{2}.b = b2grad;

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end