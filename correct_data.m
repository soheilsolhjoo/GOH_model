function data = correct_data(data)
%DATA_CORR updates data to move first point to stress = 0, stretch = 1
% The code assumes raw_data: data_1 | data_2 | ...
data(1,:) = [];
for i = 1 : size(data,2)
    corr = mod(i,2);
    MPa2KPa = (1-corr) * 1000 + corr;
    if corr
        data(:,i:i+1) = sortrows(data(:,i:i+1),1);
    end
    % data(:,i) = (data(:,i) - min(data(:,i)) + corr) * MPa2KPa;
    data(:,i) = (data(:,i) - data(1,i) + corr) * MPa2KPa;

end
end