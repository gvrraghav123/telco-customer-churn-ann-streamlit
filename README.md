# 📌 Telco Customer Churn Prediction (ANN + Streamlit)

## 📊 Project Overview
This project is an Artificial Neural Network (ANN) based machine learning application that predicts whether a telecom customer is likely to churn (leave the service) or stay.

The model is trained on the Telco Customer Churn dataset and deployed using Streamlit for interactive predictions.

---

## 🚀 Live Features
- Customer churn prediction using trained ANN model
- Interactive web UI built with Streamlit
- Real-time probability score
- Handles categorical encoding automatically
- Scaled input features for accurate prediction

---

## 🧠 Machine Learning Pipeline
1. Data Cleaning (missing value handling in TotalCharges)
2. Label Encoding for categorical features
3. Standard Scaling for numerical features
4. ANN Model Training using TensorFlow/Keras
5. Model & preprocessing artifacts saved using pickle
6. Streamlit deployment for inference

---

## 🏗️ Model Architecture
- Input Layer: 16 features
- Hidden Layer 1: 16 neurons (ReLU)
- Hidden Layer 2: 8 neurons (ReLU)
- Output Layer: 1 neuron (Sigmoid)

Loss Function: Binary Crossentropy  
Optimizer: Adam  
Metric: Accuracy  

---

## 📁 Project Structure
Telco-Churn-ANN/
│
├── app.py                      # Streamlit web application
├── churn_ann_model.keras      # Trained ANN model
├── scaler.pkl                 # StandardScaler object
├── encoders.pkl              # Label encoders
├── target_encoders.pkl       # Target label encoder
├── requirements.txt          # Dependencies
├── README.md                 # Documentation

---

## ⚙️ Installation & Setup

### 1. Clone repository
git clone https://github.com/your-username/telco-churn-ann.git
cd telco-churn-ann

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run app
streamlit run app.py

---

## 🎯 Output
- YES → Customer will churn
- NO → Customer will stay
- Shows churn probability (0–1)

---

## 📦 Requirements
streamlit
tensorflow
keras
numpy
pandas
scikit-learn

---

## ⚠️ Notes
- All model + pickle files must be in same folder as app.py
- Use TensorFlow 2.15 for best compatibility
- Handles unseen categories safely

---

## 👨‍💻 Author
ANN Classification Project (Telco Churn Prediction)
