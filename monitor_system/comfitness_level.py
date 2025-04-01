
def comfort_level( temperature , humidity ) : 
    temperature_output = [ "Too cold, please turn the heating on. " , 
                          "Too cold, please wear more clothes and have a cup of hot drink! " , 
                          "Caution! High temperature. " , 
                          "Extreme caution! High temperature. " , 
                          "Danger! High temperature, please avoid outdoor activities. " , 
                          "Extreme danger! High temperature, please stay indoor until sunset! " 
                        ]
    humidity_output = [ "Too dry, suggest use of moist lotion. "]
    outputText = ""


    if( temperature <= 15 ) : 
        if( humidity < 40 ) : 
            outputText += humidity_output[0] 
        if( temperature <= 12 ) : 
            outputText += temperature_output[0] 
        else : 
            outputText += temperature_output[1] 
        
        return outputText 

    if( temperature >= 27 ) : 
        T2 = temperature * temperature 
        H2 = humidity * humidity 
        HI = -8.784695 + 1.61139411*temperature + 2.338549*humidity - 0.14611605*temperature*humidity - 0.012308094*T2 - 0.016424828*H2 + 0.002211732*T2*humidity + 0.00072546*temperature*H2 - 0.000003582*T2*H2
        if humidity <= 30 : 
            outputText += humidity_output[0]

        if HI >= 54 :
            outputText += temperature_output[5]
        elif HI >= 41 : 
            outputText += temperature_output[4]
        elif HI >= 32 : 
            outputText += temperature_output[3]
        elif HI >= 27 : 
            outputText += temperature_output[2] 
        else : 
            outputText = "Normal."
        
    return outputText 



    
