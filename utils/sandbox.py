from utils.get_embeddings import get_image_embeddings
from utils.models import get_model, get_processor
import torch

models = ["dinov2-base", "clip-vit-large-patch14"]
datasets = ["nabirds", "cub", "cifar100"]
for model in models:
    for dataset in datasets:
        get_image_embeddings(dataset, model, split="test", save=True)