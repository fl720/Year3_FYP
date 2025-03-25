function risk_output = getCodnessIndex(temp_C, humidity_percent)
%input temperature in degree C, humidity level in %


    if humidity_percent <= 40 && temp_C < 12 
        risk_output = sprintf('[Override] Temp %.1f°C → Risk Level: risk of hypothermia, recommend using of mosit cream and putting the heating on', temp_C);
        return ; 
    end 

    if humidity_percent > 40 && temp_C < 12 
        risk_output = sprintf('[Override] Temp %.1f°C → Risk Level: risk of hypothermia, recommand putting the heating on.', temp_C);
        return ; 
    end 


    %
end