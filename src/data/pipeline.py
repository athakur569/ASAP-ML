import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class HistopathologyDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row['image_path']).convert("RGB")
        label = torch.tensor(row['label'], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label