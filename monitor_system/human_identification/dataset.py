import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import os

class RGBDImageDataset(Dataset):
    def __init__(self, label_file, rgb_dir, depth_dir, transform=None):
        self.labels = pd.read_csv(label_file)
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]
        label = int(self.labels.iloc[idx, 1])

        rgb_path = os.path.join(self.rgb_dir, img_name)
        depth_path = os.path.join(self.depth_dir, img_name)

        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

        rgb = cv2.resize(rgb, (224, 224))
        depth = cv2.resize(depth, (224, 224))

        rgb = torch.tensor(rgb).permute(2, 0, 1).float() / 255.0
        depth = torch.tensor(depth).unsqueeze(0).float() / 255.0

        return rgb, depth, torch.tensor(label)


import torch.nn as nn

class CoevolutionNet(nn.Module):
    def __init__(self):
        super(CoevolutionNet, self).__init__()
        
        self.rgb_branch = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        
        self.depth_branch = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        
        self.fusion = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56 * 2, 256),  # assuming 224x224 input
            nn.ReLU(),
            nn.Linear(256, 6)  # 6 classes
        )
    
    def forward(self, rgb, depth):
        rgb_feat = self.rgb_branch(rgb)
        depth_feat = self.depth_branch(depth)
        concat = torch.cat((rgb_feat, depth_feat), dim=1)
        return self.fusion(concat)


from torch.utils.data import DataLoader
import torch.optim as optim

dataset = RGBDImageDataset('label.csv', 'rgb', 'depth')
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = CoevolutionNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    for rgb, depth, label in loader:
        rgb, depth, label = rgb.to(device), depth.to(device), label.to(device)
        output = model(rgb, depth)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
