clear; clc; close all;
%% Load Data
global sb_11 sb_22 sx_11 sx_22 sy_11 sy_22
global lam_sb lam_sx lam_sy
global ss_11 ss_22
load_data
ss_data = [ss_11 ss_22];
lambda  = [ss_data(:,1) ss_data(:,3)];
stress  = [ss_data(:,2) ss_data(:,4)];
lambda  = F_construct(lambda); %now Lambda = F = diag(l11,l22,l33)

%% Optimization
% c: C10, k1, k2, kappa, [theta(s)(in degrees)]
% NOTE: the number of directions should be known.
% The direction can changes between 0 and 180 degrees
% c0  = [0, 0, 0, 0, [95,85]]; %the last [] is to list all directions

file_name = "optC\orient_2.mat";
train = 0;

if train
    c0 =[0, 0, 0, 0, [1,2,3]]; %#ok
    [optC_GS, fval, W_func]  = W_calibrator(lambda, stress, c0);
    save(file_name, "optC_GS","fval", "W_func", '-mat');
else
    load(file_name) 
end
% g   = calc_g(c0(5:end));
% inv = calc_l2i(g, lambda);
% psi = calc_i2p(c0, inv);
% der = calc_der(c0, inv);
% sig = calc_sig(c0, lambda);
%% Visualize the results
sb_temp = F_construct([ linspace(lam_sb(1,1),lam_sb(end,1))', ...
            linspace(lam_sb(1,2),lam_sb(end,2))']);
sx_temp = F_construct([ linspace(lam_sx(1,1),lam_sx(end,1))', ...
            linspace(lam_sx(1,2),lam_sx(end,2))']);
sy_temp = F_construct([ linspace(lam_sy(1,1),lam_sy(end,1))', ...
            linspace(lam_sy(1,2),lam_sy(end,2))']);

subplot(3,1,1)
sb_GOH_GS = W_func(optC_GS, sb_temp);
ss_plot([sb_11 sb_22],"Equibiaxial")
hold on;
plot(sb_temp(:,1),sb_GOH_GS(:,1),'b-')
plot(sb_temp(:,2),sb_GOH_GS(:,2),'r-')
hold off

subplot(3,1,2)
sx_GOH_GS = W_func(optC_GS, sx_temp);
ss_plot([sx_11 sx_22],"Off-biaxial X")
hold on;
plot(sx_temp(:,1),sx_GOH_GS(:,1),'b-')
plot(sx_temp(:,2),sx_GOH_GS(:,2),'r-')
hold off

subplot(3,1,3)
sy_GOH_GS = W_func(optC_GS, sy_temp);
ss_plot([sy_11 sy_22],"Off-biaxial Y")
hold on;
plot(sy_temp(:,1),sy_GOH_GS(:,1),'b-')
plot(sy_temp(:,2),sy_GOH_GS(:,2),'r-')
hold off
%% Compare energies
cal_e = zeros(size(sb_temp,1),1);
for i = 1:2
    cal_e(:,1) = cal_e(:,1) + cumtrapz(log(sb_temp(:,i)), sb_GOH_GS(:,i));
end

g   = calc_g(optC_GS(5:end));
inv = calc_l2i(g, sb_temp);
energy = calc_i2p(optC_GS, inv);
% energy = GOH_energy(optC_GS,invariants);
e_temp = energy - energy(1);
% close all
figure 
hold on
plot(e_temp,cal_e,'b*');
pline = [min(e_temp) ceil(max(e_temp)*1000)/1000];
plot(pline,pline,'r-');
hold off