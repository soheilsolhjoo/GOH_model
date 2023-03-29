function W = calc_i2p(c,inv)
% c: C10, k1, k2, kappa
% I: I1, I2, [I4]
C1  = c(1);
k1  = c(2);
k2  = c(3);
kap = c(4);
n_I4= numel(inv(1,:)) - 1;

W_iso   = C1 * (inv(:,1)-3);
% E       = kap * (I(:,1)-3) + (1-3*kap) * (I(:,2:2+n_I4-1)-1);
E       = kap * inv(:,1) + (1-3*kap) * inv(:,2:2+n_I4-1) - 1;
E       = (abs(E) + E)./2;
W_aniso = k1 / (2 * k2) * sum(exp((k2 * E.^2)-1),2);
W       = W_iso + W_aniso;
end