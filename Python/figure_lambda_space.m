clear; clc; close all
data = readmatrix('lambda_space.csv');
experiment = readmatrix('concatenated.csv');

% NN response
L1 = data(:,1);
n = numel(L1);
m = sqrt(n);
L1 = reshape(L1, m, m);
L2 = reshape(data(:,2) , m, m);
W = reshape(data(:,7) , m, m);
S1 = reshape(data(:,8) , m, m);
S2 = reshape(data(:,9) , m, m);

% experiment
L1_e = experiment(:,1);
L2_e = experiment(:,2);
W_e = experiment(:,5);
S1_e = experiment(:,3);
S2_e = experiment(:,4);


% figure;hold on
% surf(L1,L2,W);
% alpha 0.8
% scatter3(L1_e,L2_e,W_e,20,W_e,'filled','o');
% hold off;shading interp
% 
% figure;hold on
% surf(L1,L2,S1);
% scatter3(L1_e,L2_e,S1_e,20,S1_e,'filled','o','MarkerEdgeColor','k');
% hold off;shading interp
% 
% figure;hold on
% surf(L1,L2,S2);
% scatter3(L1_e,L2_e,S2_e,20,S2_e,'filled','o','MarkerEdgeColor','k');
% hold off;shading interp

%%
[Fx,Fy] = gradient(W);
grad = sqrt(Fx.^2 + Fy.^2);
surf(grad)
contourf(L1,L2,grad)
c = colorbar;
c.Label.String = '\nabla W';
set(gca,'TickLabelInterpreter','latex','FontSize',12)
xlabel('$\lambda_1$','Interpreter','latex','FontSize',12)
ylabel('$\lambda_2$','Interpreter','latex','FontSize',12)

% any(Fy<0,'all')