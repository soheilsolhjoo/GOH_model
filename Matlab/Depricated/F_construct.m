function [F,data_size] = F_construct(lambda, GS)
data_size = size(lambda,1);

switch GS
    case 1
        % General Solution
        F       = zeros(3,3,data_size); % pre-allocation
        i       = 1;
        loop    = true;
        while loop
            F(:,:,i) = zeros(3);
            F(1,1,i) = lambda(i,1);
            F(2,2,i) = lambda(i,2);
            F(3,3,i) = 1/(lambda(i,1) * lambda(i,2));

            i = i + 1;
            if i > data_size
                loop = false;
            end
        end
    case 2
        % Special Solution
        F = [lambda, 1./(lambda(:,1) .* lambda(:,2))];
end
end