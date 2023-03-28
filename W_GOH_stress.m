function sigma = W_GOH_stress(lambda, W_par)

% List of parameters
% lambda: stretch data in the format of [stretch_1|stretch_2]
% W     : strain energy function
% W_par : parameters of W

del_I   = 1e-6; % delta_I for calculating derivative of W wrt I

% Assigining W and W_par
c0      = W_par;
n_I4    = (numel(c0) - 4);
n_I4_ls = 1; % (1) I1, (2) I1 & I2, (3) I1, I2 & I3
n_I     = n_I4_ls + n_I4; % + I4(i)
I_list  = [1 2:n_I]; % list of invariants

W       = @(c,I) GOH_energy(c,I);

for i = 1:n_I4
    g(i,:) = [cosd(c0(4+i)), sind(c0(4+i)), 0];
    g(i,:) = g(i,:)./norm(g(i,:));
end

% Construct F
[F, data_size] = F_construct(lambda);

% pre-allocations
C       = zeros(3,3,data_size);
I       = zeros(data_size,n_I);
dWI     = zeros(data_size,n_I);
sigma   = zeros(data_size,2);
p       = zeros(data_size,1);

% Stress calculation
% Calculate C, dev(W,I), S', p, s1, s2
for i = 1:data_size
    %     Calculate C
    C(:,:,i) = F(:,:,i) .* transpose(F(:,:,i));
    % Calculate the invariants
    inv = 1;
    I(i,inv) = sum(diag(C(:,:,i)));
    for j = 1:n_I4
        inv = I_list(j+n_I4_ls);
        %           inv = inv + 1;
        I(i,inv) = g(j,:) * C(:,:,i) * g(j,:)';
    end
    % Calculate dW/dI
    for inv = I_list
        I_p         = I(i,:);
        I_p(inv)    = I_p(inv) + del_I;
        I_n         = I(i,:);
        I_n(inv)    = I_n(inv) - del_I;
        dWI(i,inv)  = -diff(W(c0,[I_p;I_n])) / (2 * del_I);
    end
    % Calculate S'
    S_PK2 = 2* (dWI(i,1)*eye(3));% + dWI(i,2)*(I(i,1)*eye(3) - C(:,:,i)));
    for j = 1:n_I4
            inv = I_list(j+n_I4_ls);
            S_PK2 = S_PK2 + 2 * dWI(i,inv) * (g(j,:)'*g(j,:));
    end
    % Calculate pressure, using BC: sigma_33 = 0
    p(i,1)      = F(3,3,i)^2 * S_PK2(3,3);
    % Calculate sigma_11 & sigma_22
    sigma(i,1)  = round( F(1,1,i)^2 * S_PK2(1,1) + p(i,1) , 4);
    sigma(i,2)  = round( F(2,2,i)^2 * S_PK2(2,2) + p(i,1) , 4);
end
end