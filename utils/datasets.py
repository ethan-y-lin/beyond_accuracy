from data.nabirds.nabirds import NABirds
from data.cifar100.cifar100 import CIFAR100
from data.cub.cub import CUB
from config import DATA_DIR

def get_dataset(name, root=DATA_DIR, train=False, transform=None, target_transform=None, download=None):
    if name == "nabirds":
        return NABirds(root, train, transform, target_transform, download)
    elif name == "cifar100":
        return CIFAR100(root, train, transform, target_transform, download)
    elif name == "cub":
        return CUB(root, train, transform, target_transform, download)
    else:
        raise ValueError(f"Dataset {name} not found")