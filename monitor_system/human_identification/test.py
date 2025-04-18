import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from train import CoevolutionNet

label_map = {
    0: 'Standing',
    1: 'Sitting',
    2: 'Lying',
    3: 'Bending',
    4: 'Crawling',
    5: 'Empty'
}
# ===========================
# Test Function
# ===========================
def test_model():
    rgb_folder = './human_identification/test/rgb'
    depth_folder = './human_identification/test/depth'
    model_path = './human_identification/model/coevolution_model.pth'

    class_names = ['Standing', 'Sitting', 'Lying', 'Bending', 'Crawling', 'Empty']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CoevolutionNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = []

    for filename in sorted(os.listdir(rgb_folder)):
        if not filename.endswith('.png'):
            continue

        image_id = filename.split('_')[-1].split('.')[0]
        rgb_path = os.path.join(rgb_folder, filename)
        depth_path = os.path.join(depth_folder, f'depth_{image_id}.png')

        if not os.path.exists(depth_path):
            print(f"Missing depth image for {filename}")
            continue

        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

        rgb = cv2.resize(rgb, (224, 224))
        depth = cv2.resize(depth, (224, 224))

        rgb = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        depth = torch.tensor(depth).unsqueeze(0).unsqueeze(0).float() / 255.0

        rgb, depth = rgb.to(device), depth.to(device)

        with torch.no_grad():
            output = model(rgb, depth)
            predicted_class = torch.argmax(output, dim=1).item()
            predicted_label = class_names[predicted_class]
            # print(f"{filename} => Predicted: {predicted_label}")
            predictions.append((filename, predicted_class))

    # Save predictions to a text file
    # with open('./human_identification/predictions.txt', 'w') as f:
    #     for filename, label in predictions:
    #         f.write(f"{filename}: {label}\n")

    # Save predictions to a CSV file (optional)
    pd.DataFrame(predictions, columns=["filename", "prediction"]).to_csv('./human_identification/predictions.csv', index=False)
    # Save as original label instead of number


if __name__ == "__main__":
    test_model()
