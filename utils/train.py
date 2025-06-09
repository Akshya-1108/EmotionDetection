import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import get_dataloaders
from model.resNet18_emotion import ResNetEmotion
import torch
import torch.nn as nn
import torch.optim as optim

def evaluate(model, dataloader, device): 
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)  
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def train():
    train_loader, val_loader = get_dataloaders("archive", batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetEmotion().to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    for epoch in range(50):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)  
        print(f"Epoch {epoch+1}/50, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step(avg_loss)

        # current_lr = optimizer.param_groups[0]['lr']
        # print(f"Current Learning Rate: {current_lr:.6f}")

    torch.save(model.state_dict(), "saved_Model/emotion_model.pth")

if __name__ == "__main__":
    train()
