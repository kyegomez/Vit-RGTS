import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from mega_vit.main import MegaVit

# 1. Setup and Imports
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Data Preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Using CIFAR-10 for demonstration purposes
cifar10 = datasets.CIFAR10(root="./data", download=True, transform=transform)
train_size = int(0.9 * len(cifar10))
val_size = len(cifar10) - train_size
train_dataset, val_dataset = random_split(cifar10, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 3. Model Initialization
model = MegaVit(
    image_size=224,
    patch_size=14,
    num_classes=10,  # CIFAR-10 has 10 classes
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0002)

# Warm-up + Cosine schedule for the learning rate
def lr_schedule(epoch):
    if epoch < 2500:
        return epoch / 2500
    return 0.5 * (1 + torch.cos((epoch - 2500) / (300000 - 2500) * 3.14159))

scheduler = LambdaLR(optimizer, lr_schedule)

# 4. Training Loop
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        clip_grad_norm_(model.parameters(), 0.05)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()

        print(f"Train Loss: {total_loss:.4f}, Train Acc: {correct / len(train_dataset):.4f}")
        
    return total_loss / len(loader), correct / len(train_dataset)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            
    return total_loss / len(loader), correct / len(val_dataset)

# Assuming we will train for a certain number of epochs (in this case, calculated to reach 300k steps)
num_epochs = (300000 * 64) // len(train_dataset)

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    print(train_loss, train_acc)

    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    print(val_loss, val_acc)
    
    print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# 5. Final Steps
torch.save(model.state_dict(), "mega_vit_model.pth")
print("Training finished.")
