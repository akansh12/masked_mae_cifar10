### Generated using claude.ai

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def get_cifar10_dataloaders(batch_size=256, num_workers=4, data_root='./data'):
    """
    Create data loaders for CIFAR-10 dataset.
    
    Args:
        batch_size: Batch size for training and evaluation
        num_workers: Number of workers for data loading
        data_root: Directory to store the dataset
        
    Returns:
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
    """
    # Mean and std for CIFAR-10 normalization
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    # Transformations for training
    # For MAE, we don't need heavy augmentations since reconstruction is the objective
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Transformations for testing
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load the training dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, 
        train=True,
        download=True, 
        transform=train_transform
    )
    
    # Load the test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, 
        train=False,
        download=True, 
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def visualize_cifar10_samples(dataloader, num_samples=16, classes=None):
    """
    Visualize some samples from the CIFAR-10 dataset.
    
    Args:
        dataloader: DataLoader containing CIFAR-10 images
        num_samples: Number of samples to visualize
        classes: List of class names for CIFAR-10
    """
    if classes is None:
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Get a batch of images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    # Convert images for visualization
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Un-normalize the images
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    
    # Un-normalize and convert to numpy for visualization
    images_np = images.clone()
    images_np = images_np * std + mean
    images_np = images_np.clamp(0, 1).numpy()
    
    # Plot the images
    fig = plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        ax = fig.add_subplot(num_samples//4, 4, i+1)
        ax.imshow(np.transpose(images_np[i], (1, 2, 0)))
        ax.set_title(f"Class: {classes[labels[i]]}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    train_loader, test_loader = get_cifar10_dataloaders()
    
    # Print dataset information
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Visualize some samples
    visualize_cifar10_samples(train_loader)