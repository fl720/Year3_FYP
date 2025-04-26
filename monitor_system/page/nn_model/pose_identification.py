import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import pandas as pd


class CoevolutionNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # REPLACE THIS SECTION WITH FINIAL CNN STRUCTURE 

class Pose_identification() : 
    def __init__( self ) : 
        self.Base_path  = "page" 
        self.resize_len = 360 
        self.resize_hei = 360
        self._model     = None
        
    def getPose( self , image ) : 
        [ model , device ]  = self.load_model( )
        poses_name = ['Standing', 'Sitting', 'Lying', 'Bending', 'Crawling', 'Empty']


        for filename in sorted(os.listdir(image)):
            if not filename.endswith('.png'):
                continue

            image_id = filename.split('_')[-1].split('.')[0]
            rgb_path = os.path.join(image, filename)

            rgb = cv2.imread(rgb_path)
            rgb = cv2.resize(rgb, (self.resize_len, self.resize_hei))
            rgb = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            rgb = rgb.to(device)

            with torch.no_grad():
                output = model(rgb)
                # print(f"{filename} => Predicted: {output}")
                predicted_class = torch.argmax(output, dim=1).item()
                predicted_label = poses_name[predicted_class]
                
        return predicted_label

    def load_model( self ) : 
        if self._model is None : 
            model_path   = './{self.Base_path}/nn_modelcoevolution_model.pth' 
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = CoevolutionNet().to(device)
            
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            [ model , device ] = self._model

        return self._model 

