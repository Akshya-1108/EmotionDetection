from dataset import get_dataloaders
from models.emotion_cnn import EmotionCNN
import torch
import torch.nn as nn
import torch.optim as optim

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def train():
    train_loader, val_loader = get_dataloaders("data/fer2013", batch_size=64)

    model = EmotionCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(15):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/15, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), "saved_models/emotion_model.pth")

if __name__ == "__main__":
    train()
