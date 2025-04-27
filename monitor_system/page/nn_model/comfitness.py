
# =================== TEMPERARY FILE ====================
def evaluate(temp: float, humidity: float) -> str:
    if temp > 30 and humidity > 70:
        return "Risk of heat exhaustion. Stay hydrated."
    elif temp < 5:
        return "Very cold. Dress warmly and avoid exposure."
    else:
        return "Conditions are normal. Stay active and healthy!"
