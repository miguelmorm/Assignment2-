import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Output: (16, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                    # Output: (16, 16, 16)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # Output: (32, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                    # Output: (32, 8, 8)

            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 100),
            nn.ReLU(),
            nn.Linear(100, 10)  # 10 clases
        )

    def forward(self, x):
        return self.network(x)


def get_model(model_name):
    if model_name == "CNN":
        return CNNModel()
    else:
        raise ValueError(f"Model '{model_name}' not implemented")
