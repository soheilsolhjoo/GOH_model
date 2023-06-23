clear; clc; close all
%% load files
exp_x   = readmatrix('data_x.csv');
exp_y   = readmatrix('data_y.csv');
exp_eq  = readmatrix('data_eq.csv');

GOH_x   = readmatrix('GOH_x_S');
GOH_y   = readmatrix('GOH_y_S');
GOH_eq  = readmatrix('GOH_eq_S');

NN_x   = readmatrix('NN_x_S');
NN_y   = readmatrix('NN_y_S');
NN_eq  = readmatrix('NN_eq_S');
%% data collection
% _x
lambdas = exp_x(:,1:2);

fig_gen(lambdas, exp_x, GOH_x, NN_x)
fig_gen(lambdas, exp_y, GOH_y, NN_y)
fig_gen(lambdas, exp_eq,GOH_eq,NN_eq)

%% functions
function fig_gen(lambdas, data_exp,data_goh,data_NN)
l1 = lambdas(:,1);
l2 = lambdas(:,2);

s1_exp = data_exp(:,3);
s2_exp = data_exp(:,4);
s1_goh = data_goh(:,5);
s2_goh = data_goh(:,6);
s1_NN  = data_NN(:,5);
s2_NN  = data_NN(:,6);

figure; hold on
scatter(l1,s1_exp,'bo','MarkerEdgeAlpha',0.4,'DisplayName','$\sigma_x$')
scatter(l2,s2_exp,'ro','MarkerEdgeAlpha',0.4,'DisplayName','$\sigma_y$')

plot(l1,s1_NN, 'k-','DisplayName','NN')
plot(l1,s1_goh,'k--','DisplayName','GOH')
legend( ...
    'Interpreter','latex', ...
    'Location','northwest', ...
    'AutoUpdate','off', ...
    'FontSize',12)

plot(l2,s2_NN, 'k-')
plot(l2,s2_goh,'k--')
xlabel('Stretch','Interpreter','latex','FontSize',12)
ylabel('Stress (KPa)','Interpreter','latex','FontSize',12)
set(gca,'TickLabelInterpreter','latex')
hold off

end