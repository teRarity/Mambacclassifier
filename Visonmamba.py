import torch.nn as nn
import torch.optim as optim
import multiprocessing
import os
import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Define the transformations for the dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5,0.5), (0.5, 0.5,0.5))
])
print(torch.cuda.is_available())
torch.cuda.empty_cache()
# Move the model to GPU
if torch.cuda.is_available():
    print("CUDA is available")
else:
    print("CUDA is not available")

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Split the training dataset into train and validation sets
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
train_dataset, val_dataset = random_split(trainset, [train_size, val_size])

# Create data loaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


class SSM(nn.Module):
    def __init__(self, channels):
        super(SSM, self).__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.linear = nn.Linear(channels, channels)

    def forward(self, x):
        out = self.bn(x)
        out = self.dwconv(out)
        out = self.gap(out)  # Apply Global Average Pooling
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = F.silu(out)
        out = out.view(out.size(0), self.bn.num_features, 1, 1)
        return out


class SimplifiedMamBaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
        super(SimplifiedMamBaBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False) if in_channels != out_channels else nn.Identity()
        self.dwconv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False)  # Depth-wise Convolution
        self.ssm = SSM(out_channels)  # Add the SSM block

    def forward(self, x):
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.dwconv(out)  # Apply Depth-wise Convolution
        out_ssm = self.ssm(out)  # Apply the SSM block
        out = out + out_ssm  # Add the output of the SSM block to the output of the depth-wise convolution
        return out

class MamBaClassifiernormal(nn.Module):
    def __init__(self, num_classes=10):
        super(MamBaClassifiernormal, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SimplifiedMamBaBlock(64, 64),  # Use SimplifiedMamBaBlock
            SimplifiedMamBaBlock(64, 64),  # Use SimplifiedMamBaBlock
            nn.MaxPool2d(kernel_size=2, stride=2),
            SimplifiedMamBaBlock(64, 128),  # Use SimplifiedMamBaBlock
            SimplifiedMamBaBlock(128, 128),  # Use SimplifiedMamBaBlock
            nn.MaxPool2d(kernel_size=2, stride=2),
            SimplifiedMamBaBlock(128, 256),  # Use SimplifiedMamBaBlock
            SimplifiedMamBaBlock(256, 256),  # Use SimplifiedMamBaBlock
            nn.MaxPool2d(kernel_size=2, stride=2),
            SimplifiedMamBaBlock(256, 512),  # Use SimplifiedMamBaBlock
            SimplifiedMamBaBlock(512, 512),  # Use SimplifiedMamBaBlock
            nn.MaxPool2d(kernel_size=2, stride=2),
            SimplifiedMamBaBlock(512, 512),  # Use SimplifiedMamBaBlock
            SimplifiedMamBaBlock(512, 512),  # Use SimplifiedMamBaBlock
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Move the flatten layer here
            nn.Linear(512 * 1 * 1, 4096),                #512 * 1 * 1, 4096
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model_input_shape(model):
    """
    Returns the expected input shape of the given model.
    """
    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 32, 32)

    # Pass the dummy input through the model to get the output shape
    with torch.no_grad():
        output = model(dummy_input)

    # Get the input shape from the first layer of the model
    input_shape = list(model.features[0].weight.shape[1:])
    input_shape.insert(0, 1)  # Add batch dimension

    return tuple(input_shape)
# Create an instance of the MamBaClassifier
model = MamBaClassifiernormal(num_classes=10)
# Get the expected input shape
input_shape = get_model_input_shape(model)
print(f"The MamBaClassifier expects input with shape: {input_shape}")

# Get the total number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())

# Print the number of parameters
print(f"Total number of parameters in the model: {total_params}")

def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def calculate_loss(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    average_loss = running_loss / len(dataloader)
    return average_loss

# Define the folder name
folder_name = "models"
# Check if the folder exists
if not os.path.exists(folder_name):
    # Create the folder
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created.")
else:
    print(f"Folder '{folder_name}' already exists.")

if __name__ == '__main__':
    multiprocessing.freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MamBaClassifiernormal(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for data in trainloader:
            inputs, labels = data[0].to(device), data[1].to(device)  # Corrected here

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_loss = calculate_loss(model, trainloader, criterion, device)
        test_loss = calculate_loss(model, testloader, criterion, device)
        train_accuracy = calculate_accuracy(model, trainloader, device)
        test_accuracy = calculate_accuracy(model, testloader, device)

        print(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

torch.save(model.state_dict(), 'D:\Coding\MaMbaClassification\models\MambaVison.pth')