function calc_strain_e = exp_strain_energy(ss_data)
lambda  = [ss_data(:,1) ss_data(:,3)];
stress  = [ss_data(:,2) ss_data(:,4)];
calc_strain_e = zeros(size(lambda,1),1);
for i = 1:2
    calc_strain_e(:,1) = calc_strain_e(:,1) + ...
        cumtrapz(log(lambda(:,i)), stress(:,i));
end
end