import torch.nn as nn
import torchvision.models as models

class ResNetEmotion(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNetEmotion, self).__init__()
        # Load the pretrained ResNet18 model
        self.base_model = models.resnet18(pretrained=True)

        # Change the input layer to accept 1-channel grayscale images
        self.base_model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Freeze all layers if you want to train only classifier
        for param in self.base_model.parameters():
            param.requires_grad = True  

        # Unfreeze the last few layers for fine-tuning (optional)
        for param in list(self.base_model.layer4.parameters()):
            param.requires_grad = True

        # Replace the final classification layer
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)
