from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import io

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Define the same CNN model architecture used during training
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # 🔹 Automatically calculate the flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            x = self.pool(nn.ReLU()(self.conv1(dummy)))
            x = self.pool(nn.ReLU()(self.conv2(x)))
            self.flattened_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flattened_size, 100)
        self.fc2 = nn.Linear(100, 2)  # Output for 2 classes

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = x.view(-1, self.flattened_size)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

# ✅ Set device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ Load trained model
model = CNNModel().to(device)
checkpoint = torch.load("cnn_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ✅ Define preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# ✅ Optional: Map index to class name
class_names = {
    0: "Class A",
    1: "Class B"
}

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "🚀 CNN Image Classification API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            class_index = predicted.item()
            class_label = class_names.get(class_index, f"Class {class_index}")

        return jsonify({
            "predicted_class_index": class_index,
            "predicted_class_label": class_label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run with: uvicorn app:app --reload
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, host="0.0.0.0", port=port)
