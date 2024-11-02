from library import *

def get_data_augmentation():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(10),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Variazioni di colore
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  
    ])

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
transform = get_data_augmentation()
cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

selected_classes = [0, 1, 3, 5]  
filtered_data = [(img, label) for i, (img, label) in enumerate(cifar10) if label in selected_classes]

images = torch.stack([item[0] for item in filtered_data])
labels = torch.tensor([1 if item[1] == 5 else 0 for item in filtered_data])

binary_dataset = TensorDataset(images, labels)

dataset_size = len(binary_dataset)
train_size = int(0.8 * dataset_size)  
val_size = dataset_size - train_size  
train_dataset, val_dataset = random_split(binary_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")