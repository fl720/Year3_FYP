import csv
from data.comfitness_level import comfort_level  # make sure both are imported
import torch
import numpy as np
import joblib
from comfitness_level_NN import ComfortNN
from constant import Base_path 

# Load trained model and preprocessors
def load_model():
    output_size = torch.load(f"./{Base_path}model/output_size.pth").item()
    model = ComfortNN(input_size=2, output_size=output_size)
    model.load_state_dict(torch.load(f"./{Base_path}model/comfitness_model.pth"))
    model.eval()

    # Load scaler and label encoder using joblib instead of torch.load
    scaler = joblib.load(f"./{Base_path}model/scaler.pkl")
    label_encoder = joblib.load(f"./{Base_path}model/label_encoder.pkl")

    return model, scaler, label_encoder
def predict_comfort(temperature, humidity):
    model, scaler, label_encoder = load_model()
    
    X_new = np.array([[temperature, humidity]])
    X_new = scaler.transform(X_new)
    X_tensor = torch.tensor(X_new, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(X_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
    
    return label_encoder.inverse_transform([predicted_label])[0]

# Define ranges
temperature_range = range(-10, 45)  # -10°C to 44°C
humidity_range = range(0, 101, 5)   # 0% to 100%, step 5%

# Open CSV to write comparison result
with open("comfort_comparison_matrix.csv", mode="w", newline='') as file:
    writer = csv.writer(file)

    # Write header: temperature values
    header = ["Humidity \\ Temp (°C)"] + [str(temp) for temp in temperature_range]
    writer.writerow(header)

    # Iterate through each humidity row
    for humidity in humidity_range:
        row = [f"{humidity}%"]
        for temp in temperature_range:
            predicted = predict_comfort(temp, humidity)
            expected = comfort_level(temp, humidity)
            match = "Correct" if predicted == expected else "Incorrect"
            row.append(match)
        writer.writerow(row)

print(" Comparison matrix saved to 'comfort_comparison_matrix.csv'.")
