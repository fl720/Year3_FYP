import os
import cv2
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim

# ===========================
# Dataset Class
# ===========================
class MultiFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        # Iterate through subdirectories (sources)
        for folder in os.listdir(root_dir):
            subdir = os.path.join(root_dir, folder)
            label_path = os.path.join(subdir, 'labels.csv')
            rgb_dir = os.path.join(subdir, 'rgb')
            depth_dir = os.path.join(subdir, 'depth')

            if not (os.path.exists(label_path) and os.path.exists(rgb_dir) and os.path.exists(depth_dir)):
                print(f"Skipped folder (missing one of labels.csv/rgb/depth): {subdir}")
                continue

            df = pd.read_csv(label_path)

            # Adding samples to the list
            for i in range(len(df)):
                image_index = int(df.loc[i, 'index'])
                label = int(df.loc[i, 'class'])

                # Format to 4-digit number (e.g. 1 -> 0001)
                image_id = f"{image_index:04d}"

                rgb_name = f"rgb_{image_id}.png"
                depth_name = f"depth_{image_id}.png"

                # rgb_path = os.path.join(rgb_dir, rgb_name)
                # depth_path = os.path.join(depth_dir, depth_name)
                rgb_path = os.path.normpath(os.path.join(rgb_dir, rgb_name))
                depth_path = os.path.normpath(os.path.join(depth_dir, depth_name))


                if os.path.exists(rgb_path) and os.path.exists(depth_path):
                    self.samples.append({
                        'rgb': rgb_path,
                        'depth': depth_path,
                        'label': label
                    })
                else:
                    print(f"Missing image: {rgb_path} or {depth_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        rgb = cv2.imread(sample['rgb'])
        depth = cv2.imread(sample['depth'], cv2.IMREAD_GRAYSCALE)

        rgb = cv2.resize(rgb, (224, 224))
        depth = cv2.resize(depth, (224, 224))

        rgb = torch.tensor(rgb).permute(2, 0, 1).float() / 255.0
        depth = torch.tensor(depth).unsqueeze(0).float() / 255.0

        label = torch.tensor(sample['label'])
        return rgb, depth, label


# ===========================
# Coevolutionary Network
# ===========================
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
            nn.Linear(32 * 56 * 56 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 6)  # 6 classes: 0 to 5
        )
    
    def forward(self, rgb, depth):
        rgb_feat = self.rgb_branch(rgb)
        depth_feat = self.depth_branch(depth)
        concat = torch.cat((rgb_feat, depth_feat), dim=1)
        return self.fusion(concat)


# ===========================
# Training Routine
# ===========================
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    dataset = MultiFolderDataset('./human_identification/data/')
    print(f"Total samples found: {len(dataset)}")

    # Split dataset into training and validation sets (80/20 or 70/30 ratio)
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader for both train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = CoevolutionNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Training phase
        for rgb, depth, labels in train_loader:
            rgb, depth, labels = rgb.to(device), depth.to(device), labels.to(device)

            outputs = model(rgb, depth)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for rgb, depth, labels in val_loader:
                rgb, depth, labels = rgb.to(device), depth.to(device), labels.to(device)

                outputs = model(rgb, depth)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "./human_identification/model/coevolution_model.pth")
    print("Model saved as coevolution_model.pth")


if __name__ == "__main__":
    train_model()
