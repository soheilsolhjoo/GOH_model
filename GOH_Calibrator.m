clear; clc; close all;
%% Data
% Load
% The followings are assumed for the data:
% Data is stored in 3 CSV files, *_equi.csv, *_offX.csv, and *_offY.csv.
% Each file has 4 columns called:
% "Lambda11, Lambda22, Sigma11MPa, Sigma22MPa".
% Moreover, the measured data starts from the second row.

data_dir    = 'dataset';
csvfiles    = dir(fullfile(data_dir, '*.csv'));
s_b     = importfile(fullfile(data_dir, csvfiles(1).name)); % Equibiaxial
s_x     = importfile(fullfile(data_dir, csvfiles(2).name)); % Off-biaxial X
s_y     = importfile(fullfile(data_dir, csvfiles(3).name)); % Off-biaxial Y

% Capture data for all sets
sb_11  = table2array(s_b(:,["Lambda11","Sigma11MPa"]));
sb_22  = table2array(s_b(:,["Lambda22","Sigma22MPa"]));
sx_11  = table2array(s_x(:,["Lambda11","Sigma11MPa"]));
sx_22  = table2array(s_x(:,["Lambda22","Sigma22MPa"]));
sy_11  = table2array(s_y(:,["Lambda11","Sigma11MPa"]));
sy_22  = table2array(s_y(:,["Lambda22","Sigma22MPa"]));

% data_corr = correct_data([sb_11,sb_22,sx_11,sx_22,sy_11,sy_22]);
% 
% sb_11 = data_corr(:,1:2);
% sb_22 = data_corr(:,3:4);
% sx_11 = data_corr(:,5:6);
% sx_22 = data_corr(:,7:8);
% sy_11 = data_corr(:,9:10);
% sy_22 = data_corr(:,11:12);

sb_lam    = [sb_11(:,1) sb_22(:,1)];
sx_lam    = [sx_11(:,1) sx_22(:,1)];
sy_lam    = [sy_11(:,1) sy_22(:,1)];

ss_11 = [sb_11; sx_11; sy_11];
ss_22 = [sb_22; sx_22; sy_22];
%%
% c: C10, k1, k2, kappa, [theta(s)(in degrees)]
% NOTE: the number of directions should be known.
% The direction can changes between 0 and 180 degrees
% c0  = [0, 0, 0, 0, [95,85]]; %#ok<> : the last [] is to list all directions
c0  = [0, 0, 0, 0, [85,95]]; %#ok<> : the last [] is to list all directions
CF = 1;     % COST FUNCTION
dirC = 1;    % calibrate direction: 0 (False) or 1 (True)
[optC_GS, fval, W_func]  = W_calibrator([ss_11 ss_22], c0, CF, dirC);

%%
% file_name = "optC\CF1_dirC1_dir1.mat";
% save(file_name, "optC_GS","fval", '-mat');
% clear
% load(file_name)
%% Visualize the results
[sb_GOH_GS, invariants, d_energy] = W_GOH_stress(sb_lam, optC_GS);
ss_plot([sb_11 sb_22],"Equibiaxial")
hold on;
plot(sb_lam(:,1),sb_GOH_GS(:,1),'b-')
plot(sb_lam(:,2),sb_GOH_GS(:,2),'r-')
hold off

% sx_GOH_GS = W_GOH_stress(sx_lam, optC_GS);
% ss_plot([sx_11 sx_22],"Off-biaxial X")
% hold on;
% plot(sx_lam(:,1),sx_GOH_GS(:,1),'b-')
% plot(sx_lam(:,2),sx_GOH_GS(:,2),'r-')
% hold off
% 
% sy_GOH_GS = W_GOH_stress(sy_lam, optC_GS);
% ss_plot([sy_11 sy_22],"Off-biaxial Y")
% hold on;
% plot(sy_lam(:,1),sy_GOH_GS(:,1),'b-')
% plot(sy_lam(:,2),sy_GOH_GS(:,2),'r-')
% hold off
%%
cal_e = zeros(size(sb_lam,1),1);
for i = 1:2
    cal_e(:,i) = cumtrapz(sb_lam(:,i), sb_GOH_GS(:,i));
end
%%
energy = GOH_energy(optC_GS,invariants);
% e_temp = energy;
e_temp = energy - energy(1);
% close all
figure 
hold on
plot(sb_lam(:,1), e_temp,'b-')
plot(sb_lam(:,2), e_temp,'r-')

plot(sb_lam(:,1), cal_e(:,1),'b*')
plot(sb_lam(:,2), cal_e(:,2),'r*')