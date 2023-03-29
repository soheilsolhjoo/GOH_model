function F = F_construct(lambda)
F = [lambda, 1./(lambda(:,1) .* lambda(:,2))];
end