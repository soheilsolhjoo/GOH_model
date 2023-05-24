function dWI = calc_der(c, inv)

del_I   = 1e-6; % delta_I for calculating derivative of W wrt I
dWI = zeros(size(inv));

for i = 1:size(inv,1)
    for j = 1:size(inv,2)
        I_p         = inv(i,:);
        I_p(j)    = I_p(j) + del_I;
        I_n         = inv(i,:);
        I_n(j)    = I_n(j) - del_I;
        dWI(i,j)  = -diff(calc_i2p(c,[I_p;I_n])) / (2 * del_I);
    end
end
end