import os
import cv2
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torchvision.transforms as T

# ===========================
# Dataset Class
# ===========================
class MultiFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for folder in os.listdir(root_dir):
            subdir = os.path.join(root_dir, folder)
            label_path = os.path.join(subdir, 'labels.csv')
            rgb_dir = os.path.join(subdir, 'rgb')
            depth_dir = os.path.join(subdir, 'depth')

            if not (os.path.exists(label_path) and os.path.exists(rgb_dir) and os.path.exists(depth_dir)):
                print(f"Skipped folder (missing one of labels.csv/rgb/depth): {subdir}")
                continue

            df = pd.read_csv(label_path)

            for i in range(len(df)):
                image_index = int(df.loc[i, 'index'])
                label = int(df.loc[i, 'class'])
                image_id = f"{image_index:04d}"

                rgb_path = os.path.normpath(os.path.join(rgb_dir, f"rgb_{image_id}.png"))
                depth_path = os.path.normpath(os.path.join(depth_dir, f"depth_{image_id}.png"))

                if os.path.exists(rgb_path) and os.path.exists(depth_path):
                    self.samples.append({
                        'rgb': rgb_path,
                        'depth': depth_path,
                        'label': label
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        rgb = cv2.imread(sample['rgb'])
        depth = cv2.imread(sample['depth'], cv2.IMREAD_GRAYSCALE)

        rgb = cv2.resize(rgb, (224, 224))
        depth = cv2.resize(depth, (224, 224))

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # Convert to RGB

        if self.transform:
            rgb = self.transform(rgb)
        else:
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
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Dropout(0.2), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Dropout(0.2), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Dropout(0.2), nn.MaxPool2d(2)
        )

        self.depth_branch = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Dropout(0.2), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Dropout(0.2), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Dropout(0.2), nn.MaxPool2d(2)
        )

        self.fusion = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 6)
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

    # Transforms (only for training)
    augment_rgb = T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    # Dataset & splits
    full_dataset = MultiFolderDataset('./human_identification/data/')
    print(f"Total samples found: {len(full_dataset)}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply transform only to training
    train_dataset.dataset.transform = augment_rgb
    val_dataset.dataset.transform = None

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Model, loss, optimizer, scheduler
    model = CoevolutionNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for rgb, depth, labels in train_loader:
            rgb, depth, labels = rgb.to(device), depth.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(rgb, depth)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")

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
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}")

        # Adjust LR
        scheduler.step(avg_val_loss)

    # Save model
    os.makedirs("./human_identification/model/", exist_ok=True)
    torch.save(model.state_dict(), "./human_identification/model/coevolution_model.pth")
    print("Model saved as coevolution_model.pth")


if __name__ == "__main__":
    train_model()
