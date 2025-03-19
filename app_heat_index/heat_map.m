% Load the heat index data
data = load('heat_index_labeled.mat');

% Extract variables
temperature = data.Temperature_Celsius;
humidity = data.Relative_Humidity_Percent;
heat_index = data.Heat_Index_Celsius;

% Flatten the heat index matrix for clustering
X = heat_index(:);

% Apply K-means clustering
numClusters = 4;
[idx, C] = kmeans(X, numClusters);

% Sort clusters by heat index center values
[~, order] = sort(C);
sorted_idx = zeros(size(idx));
for i = 1:numClusters
    sorted_idx(idx == order(i)) = i;
end

% Reshape back to the original heat index matrix shape
cluster_map = reshape(sorted_idx, size(heat_index));

% Define colormap: pale yellow, yellow, orange, red
custom_cmap = [
    1.0, 1.0, 0.8;  % Pale Yellow (Caution)
    1.0, 1.0, 0.0;  % Yellow (Extreme Caution)
    1.0, 0.6, 0.0;  % Orange (Danger)
    1.0, 0.0, 0.0   % Red (Extreme Danger)
];

% Plotting
figure;
imagesc(humidity, temperature, cluster_map);
colormap(custom_cmap);
colorbar('Ticks', 1:4, 'TickLabels', {'Caution', 'Extreme Caution', 'Danger', 'Extreme Danger'});
xlabel('Relative Humidity (%)');
ylabel('Temperature (°C)');
title('Heat Index Risk Categories via K-means Clustering');
set(gca, 'YDir', 'normal');  % So temperature increases upward
