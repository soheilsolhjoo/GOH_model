function stress_stretch = importfile(filename, dataLines)
%IMPORTFILE Import data from a text file
%  STRESS_STRETCH = IMPORTFILE(FILENAME) reads data from text file FILENAME
%  for the default selection.  Returns the data as a table.
%
%  STRESS_STRETCH = IMPORTFILE(FILE, DATALINES)
%  reads data for the specified row interval(s) of text file FILENAME.
%  Specify DATALINES as a positive scalar integer or a N-by-2 array of
%  positive scalar integers for dis-contiguous row intervals.
% 
%  Assumed structure: Lambda11 | Lambda22 | Sigma11MPa | Sigma22MPa
%
%  Example:
%  stress_stretch = importfile("FILE.CSV", [2, Inf]);

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [2, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 4);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Lambda11", "Lambda22", "Sigma11MPa", "Sigma22MPa"];
opts.VariableTypes = ["double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
stress_stretch = readtable(filename, opts);

end