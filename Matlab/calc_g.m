function g = calc_g(orient)

g = zeros(numel(orient),3);
g(:,1:2)    = [cosd(orient)' sind(orient)'];

end

