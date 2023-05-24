function load_data
% Load
% The followings are assumed for the data:
% Data is stored in 3 CSV files, *_equi.csv, *_offX.csv, and *_offY.csv.
% Each file has 4 columns called:
% "Lambda11, Lambda22, Sigma11MPa, Sigma22MPa".
% Moreover, the measured data starts from the second row.
global sb_11 sb_22 sx_11 sx_22 sy_11 sy_22
global lam_sb lam_sx lam_sy
global ss_11 ss_22

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

data_corr = correct_data([sb_11,sb_22,sx_11,sx_22,sy_11,sy_22]);

sb_11 = data_corr(:,1:2);
sb_22 = data_corr(:,3:4);
sx_11 = data_corr(:,5:6);
sx_22 = data_corr(:,7:8);
sy_11 = data_corr(:,9:10);
sy_22 = data_corr(:,11:12);

lam_sb    = [sb_11(:,1) sb_22(:,1)];
lam_sx    = [sx_11(:,1) sx_22(:,1)];
lam_sy    = [sy_11(:,1) sy_22(:,1)];

ss_11 = [sb_11; sx_11; sy_11];
ss_22 = [sb_22; sx_22; sy_22];
end

