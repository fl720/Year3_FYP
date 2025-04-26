import torch 
import numpy as np 
import joblib 
import torch.nn as nn


class ComfortNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ComfortNN, self).__init__()

        # ---- FN ----
        self.layer1 = nn.Linear(input_size,16)
        self.layer2 = nn.Linear(16,output_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        # x = self.activation(x)
        return x

class Comfitness( ) :
    def __init__( self ): 
        self.Base_path     = "page"
        self._modelPackage = None 

    def load_model( self ):
        self._modelPackage
        if model is None : 
            output_size = torch.load(f"./{self.Base_path}/nn_model/output_size.pth").item()
            model = ComfortNN(input_size=2, output_size=output_size)
            model.load_state_dict(torch.load(f"./{self.Base_path}/nn_model/comfitness_model.pth"))
            model.eval()

            # Load scaler and label encoder
            scaler = joblib.load(f"./{self.Base_path}/nn_model/scaler.pkl")
            label_encoder = joblib.load(f"./{self.Base_path}/nn_model/label_encoder.pkl")
            self._modelPackage = [model, scaler, label_encoder]

        return self._modelPackage

    def getComfitness(self, temperature, humidity):
        [model, scaler, label_encoder] = self.load_model()
        
        X_new = np.array([[temperature, humidity]])
        X_new = scaler.transform(X_new)
        X_tensor = torch.tensor(X_new, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(X_tensor)
            predicted_label = torch.argmax(output, dim=1).item()
        
        return label_encoder.inverse_transform([predicted_label])[0]