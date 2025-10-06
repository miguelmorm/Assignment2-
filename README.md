# 🧠 Assignment 2 – Image Classification with CNNs

Welcome to my submission for Assignment 2 of the Applied Generative AI course at Columbia University. This project focuses on building, training, evaluating, and deploying a convolutional neural network (CNN) for binary image classification.

## 🚀 How to Run the Project

This project is built using Python, PyTorch, and FastAPI. Below are the instructions to set up and test the application end-to-end.

---

### 🔧 Step 1 – Install Requirements

Ensure you are inside your virtual environment and run:

pip install -r requirements.txt


---

### 🏋️‍♂️ Step 2 – Train the Model

To train the CNN model using the dataset located in data/train and data/val, run:

python main.py

This will:
- Load and preprocess the dataset
- Train a CNN model
- Save the model checkpoint to cnn_model.pth

---

### 📈 Step 3 – Evaluate the Model

You can evaluate the performance of the trained model by running:

python helper_lib/evaluator.py

This will output accuracy and loss metrics on the validation set.

---

### 🖼️ Step 4 – Start the API to Make Predictions

To launch the prediction API using FastAPI, run:

uvicorn app:app --reload

This will start a local server at http://127.0.0.1:8000.

---

### 📤 Step 5 – Make a Prediction (Using curl)

Once the API is running, you can use the curl command to test an image:

curl -X POST http://127.0.0.1:8000/predict   -F "image=@Golden_retriever_eating_pigs_foot.jpg"

You will receive a response like:

{
  "predicted_class_index": 1,
  "predicted_class_label": "Class B"
}

---

## ✅ Assignment Requirements Covered

The following three components were implemented:

1. Model Training (main.py):
   - Trains a CNN on a binary image classification task.
   - Saves model weights as cnn_model.pth.

2. Evaluation (evaluator.py):
   - Loads the saved model and evaluates it on the validation set.

3. API Endpoint (app.py):
   - Provides a REST API for making predictions on new images using FastAPI.

---

## 📁 Folder Structure

ASSIGNMENT2MAM2670/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── helper_lib/
│   ├── checkpoints.py
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── model.py
│   ├── resize_images.py
│   └── trainer.py
├── app.py
├── main.py
├── cnn_model.pth
├── requirements.txt
├── README.md

---

## 🧠 Notes

- The image Golden_retriever_eating_pigs_foot.jpg is used for demo prediction.
- All helper functions are modularized inside the helper_lib/ directory.
- The model checkpoint file may be large; consider .gitignore-ing it if necessary.

---

## 👨‍💻 Author

Miguel Angel Morales Muñoz  
Columbia University – Fall 2025  
Course: Applied Generative AI  
UNI: mam2670  
