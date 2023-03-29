function data = correct_data(data)
%DATA_CORR updates data to move first point to stress = 0, stretch = 1
% The code assumes raw_data: data_1 | data_2 | ...
data(1,:) = [];
for i = 1 : size(data,2)
    data(:,i) = data(:,i) - min(data(:,i)) + mod(i,2);
end
end