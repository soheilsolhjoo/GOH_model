function sigma = calc_sig(g, dWI, lambda)

nDig    = 6; % number of digits used to round the calculated sigma

% g       = calc_g(c(5:end));     % direction(s)
% inv     = calc_l2i(g, lambda);  % invariants
% dWI     = calc_der(c, inv);     % derivative of energy wrt invariants

% pre-allocations
sigma   = zeros(size(lambda,1),2);

% Cauchy stress calculation
for i = 1:size(lambda,1)
    % Calculate S'
    S_PK2 = (dWI(i,1).*eye(3));
    for j = 2:size(dWI,2)
        S_PK2   = S_PK2 + dWI(i,j) .* (g(j-1,:)' * g(j-1,:));
    end
    S_PK2 = 2 * S_PK2;
    % Calculate pressure using BC: sigma_33 = 0
    p     = lambda(i,3)^2 * S_PK2(3,3);
    % Calculate sigma_11 & sigma_22
    sigma(i,1)  = lambda(i,1)^2 * S_PK2(1,1) - p;
    sigma(i,2)  = lambda(i,2)^2 * S_PK2(2,2) - p;
end
sigma = round(sigma,nDig);
end