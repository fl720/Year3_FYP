import torch
import numpy as np
import joblib
from comfitness_level_NN import ComfortNN

# Load trained model and preprocessors
def load_model():
    output_size = torch.load("output_size.pth").item()
    model = ComfortNN(input_size=2, output_size=output_size)
    model.load_state_dict(torch.load("comfitness_model.pth"))
    model.eval()

    # Load scaler and label encoder using joblib instead of torch.load
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    return model, scaler, label_encoder

# Function to predict comfort level
def predict_comfort(temperature, humidity):
    model, scaler, label_encoder = load_model()
    
    X_new = np.array([[temperature, humidity]])
    X_new = scaler.transform(X_new)
    X_tensor = torch.tensor(X_new, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(X_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
    
    return label_encoder.inverse_transform([predicted_label])[0]

# Example Usage
if __name__ == "__main__":
    temp = 30  # Example temperature
    humidity = 70  # Example humidity level
    result = predict_comfort(temp, humidity)
    print(f"Predicted Comfort Level for {temp}°C and {humidity}% humidity: {result}")
