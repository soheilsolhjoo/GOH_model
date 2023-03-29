function inv = calc_l2i(g, lambda)

C           = lambda.*lambda;
inv         = [sum(C,2) (g(:,1:2).^2 * C(:,1:2)')'];

end

