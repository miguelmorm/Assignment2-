import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loader(data_dir, batch_size=32, train=True):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader
