import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
import torch.nn as nn
import torch.optim as optim
import torch
#-----------------------------------
#PART ONE: Practice: CNN Architecture
#-----------------------------------
#As part of this assignment you will implement a Convolutional Neural Network
#using PyTorch that matches the following architecture:
#Input: RGB image of size 64×64×3
#Conv2D with 16 filters, kernel size 3×3, stride 1, padding 1
#ReLU activation
#MaxPooling2D with kernel size 2×2, stride 2
#Conv2D with 32 filters, kernel size 3×3, stride 1, padding 1
#ReLU activation
#MaxPooling2D with kernel size 2×2, stride 2
#Flatten the output
#Fully connected layer with 100 units
#ReLU activation
#Fully connected layer with 10 units (assume 10 output classes
                                     
# ✅ Hyperparameters
batch_size = 64
epochs = 10
learning_rate = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ✅ CNN Model adjusted automatically
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # 🔹 Compute flatten size automatically using dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)  # match your image size
            x = self.pool(nn.ReLU()(self.conv1(dummy)))
            x = self.pool(nn.ReLU()(self.conv2(x)))
            self.flattened_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flattened_size, 100)
        self.fc2 = nn.Linear(100, 2)  # two classes only: Class0 and Class1

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = x.view(-1, self.flattened_size)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    print("🚀 Starting training pipeline...")

    # ✅ Load datasets
    print("📁 Loading training data...")
    train_loader = get_data_loader('data/train', batch_size=batch_size, train=True)
    print("📁 Loading validation data...")
    val_loader = get_data_loader('data/val', batch_size=batch_size, train=False)
    print("📁 Loading test data...")
    test_loader = get_data_loader('data/test', batch_size=batch_size, train=False)

    # ✅ Load model
    model = CNNModel()
    model.to(device)

    # ✅ Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ✅ Train model
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs, device=device)

    # ✅ Evaluate model
    test_loss, test_acc = evaluate_model(trained_model, test_loader, criterion, device=device)
    print(f"✅ Final Test Accuracy: {test_acc:.2f}%")


#-----------------------------------
#PART TWO: Save the trained model for API usage
#-----------------------------------
torch.save(trained_model.state_dict(), "cnn_model.pth")
print("📦 Model saved as cnn_model.pth for API usage.")

if __name__ == '__main__':
    main()
