from library import *

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

cifar10 = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

train_loader = DataLoader(cifar10, batch_size=32, shuffle=True)

data_iter = iter(train_loader)
images, labels = next(data_iter)

print(f"Shape of batch images: {images.shape}")
print(f"Labels: {labels}")

# Redefine dataset 
dog_class = 5 
non_dog_classes = [0, 1, 2, 3, 4, 6, 7, 8, 9] 

binary_indices = [i for i, (_, label) in enumerate(cifar10) if label == dog_class or label in non_dog_classes]

binary_dataset = Subset(cifar10, binary_indices)
binary_dataset.dataset.targets = [1 if cifar10[i][1] == dog_class else 0 for i in binary_indices]

binary_loader = DataLoader(binary_dataset, batch_size=32, shuffle=True)

data_iter = iter(binary_loader)
images, labels = next(data_iter)

print(f"Shape of batch images: {images.shape}")
print(f"Binary labels: {labels}")


