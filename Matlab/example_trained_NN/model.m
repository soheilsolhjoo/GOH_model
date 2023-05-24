function W = model(parameters,I0)

% I0 = [I1; I41; I42];
numLayers = numel(fieldnames(parameters))/2;

% First fully connect operation.
weights = parameters.fc1_Weights;
bias = parameters.fc1_Bias;
W = fully_connect(I0,weights,bias);

% tanh and fully connect operations for remaining layers.
for i=2:numLayers
    name = "fc" + i;

    W = tanh(W);

    weights = parameters.(name + "_Weights");
    bias = parameters.(name + "_Bias");
    W = fully_connect(W, weights, bias);
end

end

function FC = fully_connect(X,W,b)
FC = W*X + b;
end