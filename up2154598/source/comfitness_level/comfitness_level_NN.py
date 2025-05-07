import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler

from constant import Base_path 


# Load dataset and preprocess data
def load_data(filename):
    df = pd.read_csv(filename)
    
    X = df[["Temperature", "Humidity"]].values
    y = df["Comfort Level"].values
    print( f"Training data size: {len(X)}")
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # ----- NORMALISATION -----
    scaler = StandardScaler()
    # scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    
    return X, y, label_encoder, scaler

# Define Neural Network Model
class ComfortNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ComfortNN, self).__init__()

        # Feedfoward Network
        self.layer1 = nn.Linear(input_size,16)
        self.layer2 = nn.Linear(16,output_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        # x = self.activation(x)
        return x

# Train Model
def train_model(X, y, epochs=2000, lr=0.01):
    output_size = len(np.unique(y))
    model = ComfortNN(input_size=2, output_size=output_size)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # for learning rate - most commonly used method 
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = loss_function(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    torch.save(scaler, f"./{Base_path}/model/scaler.pth")
    torch.save(label_encoder, f"./{Base_path}/model/label_encoder.pth")
    torch.save(model.state_dict(), f"./{Base_path}/model/comfitness_model.pth")
    torch.save(torch.tensor(output_size), f"./{Base_path}/model/output_size.pth")

    # Save scaler and label encoder using joblib instead of torch.save
    joblib.dump(scaler, f"./{Base_path}/model/scaler.pkl")
    joblib.dump(label_encoder, f"./{Base_path}/model/label_encoder.pkl")
    print("Model and preprocessors saved successfully.")

if __name__ == "__main__":
    filename = f"./{Base_path}/data/comfitness_training.csv"
    X, y, label_encoder, scaler = load_data(filename)
    train_model(X, y)
