function totalElements = total_elements(myStruct)
    fields = fieldnames(myStruct);
    totalElements = 0;
    
    for i = 1:numel(fields)
        fieldValue = myStruct.(fields{i});
        
        if isstruct(fieldValue)
            totalElements = totalElements + countStructureElements(fieldValue);
        else
            totalElements = totalElements + numel(fieldValue);
        end
    end
end
