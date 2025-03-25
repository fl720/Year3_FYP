
% ----------------- INPUTS ----------------------
t_input = 0 ; % Temperature input in Degree C
h_input = 0 ; % Humidity Level input in %

% ----------------- SETTINGS --------------------
low_temperature_warning = 12 ; 
high_temperature_warning = 27 ; 

% any temperature above 27 leave to the getHeatIndex. 
% any temperature below 10 leave to codnessIndex. 

if t_input >= high_temperature_warning 
    getHeatIndexRisk(t_input , h_input) ; 
elseif t_input <= low_temperature_warning 
    getCodnessIndex(t_input , h_input) ; 
else 
    sprintf('Normal Temperature, Humidity level');
end 