function risk_output = getHeatIndexRisk(temp_C, humidity_percent)
%input temperature in degree C, humidity level in %
    % Load data
    data = load('heat_index_labeled.mat');  
    cluster_data = load('cluster_map.mat'); 

    % Ensure correct types and shapes
    heat_index_table = double(data.Heat_Index_Celsius);
    temperature_grid = reshape(double(data.Temperature_Celsius), [], 1);  % 16×1
    humidity_grid = reshape(double(data.Relative_Humidity_Percent), 1, []);  % 1×16

    cluster_centers_sorted = double(cluster_data.cluster_centers_sorted);

    % Handle special case: above 42°C = Extreme Danger
    if temp_C > 42
        HI_interp = NaN;
        risk_output = sprintf('[Override] Temp %.1f°C → Risk Level: Extreme Danger', temp_C);
        return;
    end

    % Use extrapolation
    HI_interp = interp2(humidity_grid, temperature_grid, heat_index_table, ...
                        humidity_percent, temp_C, 'linear', NaN);

    % Check if HI is invalid or temp too low
    if temp_C < 27 || isnan(HI_interp)
        risk_output = 'Temperature is too low or input is outside valid heat index range.';
        return;
    end

    % Find nearest cluster
    [~, cluster_label] = min(abs(HI_interp - cluster_centers_sorted));

    % Labels
    labels = {'Caution', 'Extreme Caution', 'Danger', 'Extreme Danger'};
    risk_output = sprintf('Heat Index: %.1f°C → Risk Level: %s', HI_interp, labels{cluster_label});
end
