clear; clc; close all;
%% Load Data
global sb_11 sb_22 sx_11 sx_22 sy_11 sy_22
% global lam_sb lam_sx lam_sy
global ss_11 ss_22
load_data
ss_data = [ss_11 ss_22];
lambda  = [ss_data(:,1) ss_data(:,3)];
stress  = [ss_data(:,2) ss_data(:,4)];
lambda  = F_construct(lambda); %now Lambda = F = diag(l11,l22,l33)
%% Energy
W_b = exp_strain_energy([sb_11 sb_22]);
W_b = W_b - W_b(1);
W_1 = exp_strain_energy([sx_11 sx_22]);
W_1 = W_1 - W_1(1);
W_2 = exp_strain_energy([sy_11 sy_22]);
W_2 = W_2 - W_2(1);

F0 = lambda;           % X
W0 = [W_b;W_1;W_2]';    % Y
%% Deep NN
numLayers   = 5;
numNeurons  = 5;
numOrient   = 2; % to be used later by adding it to numOutput
numInput    = 1+numOrient; % I1, I4(n)
numOutput   = 2+numOrient; % W, W'_1, W'_4(n), TO BE ADDED : alpha

layers = featureInputLayer(numInput);

for i = 1:numLayers-1
    layers = [
        layers
        fullyConnectedLayer(numNeurons)
        tanhLayer];
end

layers = [
    layers
    fullyConnectedLayer(numOutput)];

net = dlnetwork(layers);
%% Train NN
numEpochs = 1500;
solverState = lbfgsState;

orient  = randi(180,1,numOrient);
% F0 = dlarray(F0,"CB");
g  = calc_g(orient);     % direction(s)
I0 = calc_l2i(g, F0);  % invariants
I0 = dlarray(I0,"BC");
W0 = dlarray(W0,"CB");
S0 = dlarray(stress,"CB");

lossFcn = @(net) dlfeval(@modelLoss,net,I0,W0,S0,g,F0);

monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info="Epoch", ...
    XLabel="Epoch");

for i = 1:numEpochs
    [net, solverState] = lbfgsupdate(net,lossFcn,solverState);

    updateInfo(monitor,Epoch=i);
    recordMetrics(monitor,i,TrainingLoss=solverState.Loss);
end
%% evaluate
% orient  = [0,90];
g  = calc_g(orient);     % direction(s)
I0 = calc_l2i(g, F0);  % invariants
I0 = dlarray(I0,"BC");
W_pred = model(parameters,I0);

figure 
hold on
plot(W_pred(1,:) - W_pred(1,1),W0,'b*');
pline = [min(W0) ceil(max(W0)*1000)/1000];
plot(pline,pline,'r-');
hold off

S_pred = calc_sig(g, extractdata(W_pred(2:end,:))', F0);

figure; hold on
plot(S0(:,1))
plot(S_pred(:,1))
hold off
figure; hold on
plot(S0(:,end))
plot(S_pred(:,end))
hold off
%% Supporting Functions
function [loss,gradients] = modelLoss(net,I0,W0,S0,g,F0)

% Make predictions with the initial conditions.
% g  = calc_g(orient);     % direction(s)
% I0 = calc_l2i(g, F0);  % invariants
% I0 = dlarray(I0,"BC");

model_pred = forward(net,I0);
W0Pred = model_pred(1,:) - model_pred(1,1);
mseW = l2loss(W0Pred, W0);

% Calculate derivatives with respect to I0.
gradientsW = dlgradient(sum(W0Pred,"all"),{I0},EnableHigherDerivatives=true);
dW_auto = gradientsW{1}(:,:);
msedWI = l2loss(model_pred(2:end,:),dW_auto);

% Calculate stresses corresponding to the identified W and dWI
S_pred = dlarray(calc_sig(g, extractdata(model_pred(2:end,:))', F0),"CB");
mseS = sqrt(l2loss(S_pred*1000,S0*1000));

% % Calculate second-order derivatives with respect to X.
% Uxx = dlgradient(sum(Ux,"all"),X,EnableHigherDerivatives=true);
% 
% % Calculate mseF. Enforce Burger's equation.
% f = Ut + U.*Ux - (0.01./pi).*Uxx;
% zeroTarget = zeros(size(f),"like",f);
% mseF = l2loss(f,zeroTarget);
% 
% % Calculate mseU. Enforce initial and boundary conditions.
% XT0 = cat(1,X0,T0);
% U0Pred = forward(net,XT0);
% mseU = l2loss(U0Pred,U0);

% Calculated loss to be minimized by combining errors.
L1 = mseW - msedWI;
L2 = mseS;
loss = (L1^2 + L2^2) / (L1 + L2);

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end