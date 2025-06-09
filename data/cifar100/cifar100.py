from torchvision.datasets import CIFAR100 as TorchCIFAR100
from torchvision import transforms

class CIFAR100(TorchCIFAR100):
    def __init__(self, root, train=False, transform=None, target_transform=None, download=False):
        super().__init__(root=root,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)