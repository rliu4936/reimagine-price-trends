import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
import os.path as op
import numpy as np
import os

from PIL import Image
from torch.utils.data import Dataset
from DataHandler import dgp_config as dcf
from DataHandler import equity_data as eqd
from Misc import utilities as ut
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms




class CurrentDataset(Dataset):
    def __init__(self, image_dir):
        """
        Args:
            image_dir: folder containing 60x64 grayscale PNGs
            use_cached_stats: if True, compute mean/std from first 50k images
        """
        self.image_dir = image_dir
        self.image_files = sorted([
            f for f in os.listdir(image_dir) if f.endswith(".png")
        ])
        self.images = self._load_images_as_numpy_array()
        
        mean, std = self._compute_mean_std(self.images)


        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to shape (1, 60, 64)
            transforms.Normalize(mean=[mean], std=[std])
        ])

    def _load_images_as_numpy_array(self):
        """Load all images into a numpy array of shape (N, 60, 64)"""
        arr = []
        for fname in self.image_files:
            path = os.path.join(self.image_dir, fname)
            img = Image.open(path).convert("L")  # Grayscale
            arr.append(np.array(img, dtype=np.uint8))
        return np.stack(arr)

    def _compute_mean_std(self, img_arr):
        """Compute mean and std from a numpy array (N, 60, 64)"""
        img_arr = img_arr.astype(np.float32) / 255.0
        mean = img_arr.mean()
        std = img_arr.std()
        return mean, std

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        path = os.path.join(self.image_dir, filename)
        image = Image.open(path).convert("L")

        stock_id = filename.replace(".png", "")

        return {
            "image": self.transform(image),      # Tensor: [1, 60, 64]
            "StockID": stock_id,                 # str: e.g., "000001.SZ"
            "ending_date": "2099-12-31",         # placeholder/future date
            "ret_val": torch.tensor(0.0),        # dummy value
            "MarketCap": torch.tensor(0.0),      # dummy value
        }