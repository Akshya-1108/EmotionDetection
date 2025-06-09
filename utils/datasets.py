from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=64):
    transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(44),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
