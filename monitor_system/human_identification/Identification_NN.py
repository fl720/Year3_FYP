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
    # df = pd.read_csv(filename)
    df = load(filename)
    
    X = df[["Temperature", "Humidity"]].values
    y = df["Comfort Level"].values
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, label_encoder, scaler

# Define Neural Network Model
class ComfortNN(nn.Module):
    def __init__(self, Fig_size):
        super(ComfortNN, self).__init__()

        hidden_state_size = 16 * 16 * 3

        self.cnn = CNN(Fig_size, hidden_state_size)

        self.ffn = FFN(hidden_state_size, 8)

        self.output_layer = FFN(16, 2)


    def forward(self, fig1, fig2):

        x1 = self.cnn(fig1)
        x1 = self.ffn(x1)

        x2 = self.cnn(fig2)
        x2 = self.ffn(x2)

        x = [x1, x2]
        
        x = self.output_layer(x)
        return x

# Train Model
def train_model(X, y, epochs=300, lr=0.01):
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
    
    torch.save(scaler, "./comfitness_level/model/scaler.pth")
    torch.save(label_encoder, "./comfitness_level/model/label_encoder.pth")
    torch.save(model.state_dict(), "./comfitness_level/model/comfitness_model.pth")
    torch.save(torch.tensor(output_size), "./comfitness_level/model/output_size.pth")

    # Save scaler and label encoder using joblib instead of torch.save
    joblib.dump(scaler, "./comfitness_level/model/scaler.pkl")
    joblib.dump(label_encoder, "./comfitness_level/model/label_encoder.pkl")
    print("Model and preprocessors saved successfully.")

if __name__ == "__main__":
    filename = "./comfitness_level/data/comfitness_training.csv"
    X, y, label_encoder, scaler = load_data(filename)
    train_model(X, y)
