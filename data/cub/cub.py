import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CUB(Dataset):
    def __init__(self, root="CUB_200_2011", train=False, transform=None, target_transform=None, download=None):
        split = "train" if train else "test"
        dataset_path = "CUB_200_2011"

        # Load split CSV
        self.data = pd.read_csv(os.path.join(dataset_path, f"{split}_dataset.csv"))

        # Image directory
        self.images_dir = os.path.join(dataset_path, "images")

        # Transforms
        self.transform = transform
        self.target_transform = target_transform

        self.name = "cub"

    def __len__(self):
        return len(self.data)

    def format_class_name(self, class_name):
        # Converts "001.Black_footed_Albatross" to (0, "black footed albatross")
        name_parts = class_name.split(".")
        if len(name_parts) == 2:
            class_id = int(name_parts[0]) - 1  # zero-indexed
            formatted_name = name_parts[1].replace("_", " ").lower()
            return class_id, formatted_name
        return -1, class_name.lower()

    def __getitem__(self, idx):
        image_id, image_name, class_name = self.data.iloc[idx]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        class_id, _ = self.format_class_name(class_name)

        if self.target_transform:
            class_id = self.target_transform(class_id)

        return image, class_id
