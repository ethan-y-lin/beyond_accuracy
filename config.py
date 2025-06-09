import os
import torch

# Directories
CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
DATA_DIR = os.getenv("DATA_DIR", "./datasets")
RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")
METADATA_DIR = os.getenv("METADATA_DIR", "./metadata")
FORMATED_DESCRIPTOR_DIR = os.getenv("FORMATED_DESCRIPTOR_DIR", "./descriptors/no_class_names")
DESCRIPTOR_DIR = os.getenv("DESCRIPTOR_DIR", "./descriptors/class_names")

# Model Configs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = os.getenv("CLIP_MODEL", "clip-vit-large-patch14")
