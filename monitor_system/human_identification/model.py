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
