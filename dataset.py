import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms


def get_cifar10_dataloaders(data_dir="./data", batch_size=128, val_split=0.1, num_workers=2, seed=42):
    """
    Charge CIFAR-10 et retourne trois DataLoaders :
    - train_loader : 45 000 images (avec data augmentation)
    - val_loader   : 5 000 images (sans data augmentation)
    - test_loader  : 10 000 images (jeu de test officiel)
    """
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    full_val_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=eval_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=eval_transform
    )

    num_train = len(full_train_dataset)
    val_size = int(num_train * val_split)
    train_size = num_train - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_train, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader