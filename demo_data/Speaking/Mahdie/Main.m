fileName = 'Writing.json'; % filename in JSON extension.
fid = fopen(fileName); % Opening the file.
raw = fread(fid,inf); % Reading the contents.
str = char(raw'); % Transformation.
fclose(fid); % Closing the file.
data = jsondecode(str); % Using the jsondecode function to parse JSON from string.
%%
fileName = 'Writingdevice1.json'; % filename in JSON extension.
fid = fopen(fileName); % Opening the file.
raw = fread(fid,inf); % Reading the contents.
str = char(raw'); % Transformation.
fclose(fid); % Closing the file.
data1 = jsondecode(str); % Using the jsondecode function to parse JSON from string.
%%
fileName = 'Writingdevice2.json'; % filename in JSON extension.
fid = fopen(fileName); % Opening the file.
raw = fread(fid,inf); % Reading the contents.
str = char(raw'); % Transformation.
fclose(fid); % Closing the file.
data2 = jsondecode(str); % Using the jsondecode function to parse JSON from string.
%%
fileName = 'Writingdevice3.json'; % filename in JSON extension.
fid = fopen(fileName); % Opening the file.
raw = fread(fid,inf); % Reading the contents.
str = char(raw'); % Transformation.
fclose(fid); % Closing the file.
data3 = jsondecode(str); % Using the jsondecode function to parse JSON from string.
%%
fileName = 'Writingdevice4.json'; % filename in JSON extension.
fid = fopen(fileName); % Opening the file.
raw = fread(fid,inf); % Reading the contents.
str = char(raw'); % Transformation.
fclose(fid); % Closing the file.
data4 = jsondecode(str); % Using the jsondecode function to parse JSON from string.
%%
cosine_sim = dot(data1.record_data, data2.record_data) / (norm(data1.record_data) * norm(data2.record_data));
%%
jaccard_sim = length(intersect(data1.record_data, data2.record_data)) / length(union(data1.record_data, data2.record_data));
