import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Load dataset and preprocess data
def load_data(filename):
    df = pd.read_csv(filename)
    
    X = df[["Temperature", "Humidity"]].values
    y = df["Comfort Level"].values
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, label_encoder, scaler

# Define Neural Network Model
class ComfortNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ComfortNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 16)
        self.layer2 = nn.Linear(16, 16)
        self.output = nn.Linear(16, output_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.output(x)
        return x

# Train Model
def train_model(X, y, epochs=300, lr=0.01):
    output_size = len(np.unique(y))
    model = ComfortNN(input_size=2, output_size=output_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    torch.save(scaler, "scaler.pth")
    torch.save(label_encoder, "label_encoder.pth")
    torch.save(model.state_dict(), "comfitness_model.pth")
    torch.save(torch.tensor(output_size), "output_size.pth")

    # Save scaler and label encoder using joblib instead of torch.save
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    print("Model and preprocessors saved successfully.")

if __name__ == "__main__":
    filename = "comfitness_training.csv"
    X, y, label_encoder, scaler = load_data(filename)
    train_model(X, y)
