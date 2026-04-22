# 🔐 Anomaly Detection using Quantized AutoEncoders

This project implements a deep learning–based anomaly detection system for intrusion detection using the **KDDCup99 dataset**. A fully connected AutoEncoder is trained to learn normal network behavior and detect anomalies based on reconstruction error. The model is further optimized using **TensorFlow Lite quantization** for efficient deployment in resource-constrained (embedded) environments.

---

## 🚀 Key Features

- Unsupervised anomaly detection using AutoEncoders  
- Detection based on reconstruction error thresholding  
- Training on normal samples only  
- Post-training quantization using TensorFlow Lite  
- Optimized for embedded and real-time deployment  

---

## 🧠 Model Architecture

- Symmetric Fully Connected AutoEncoder  
- Encoder: 128 → 64 → 32 → 16 (bottleneck)  
- Decoder: 16 → 32 → 64 → 128  
- Activation: ReLU (hidden layers), Sigmoid (output)  
- Loss Function: Mean Squared Error (MSE)  
- Optimizer: Adam  

---

## 📊 Dataset

- **KDDCup99 (UCI ML Repository)**
- ~494,000 samples  
- 41 features  
- Reformulated into binary classification:
  - `0` → Normal  
  - `1` → Attack  

---

## ⚙️ Methodology

- Data preprocessing:
  - One-hot encoding of categorical features  
  - Min-Max normalization  
- Train-test split: 80/20  
- Training only on normal data  
- Anomaly detection using:
  - Reconstruction error  
  - Threshold: μ + 3σ  

---

## 📈 Results

| Metric        | Value |
|--------------|------|
| Accuracy     | 0.9884 |
| Precision    | 0.9945 |
| Recall       | 0.9911 |
| F1-Score     | 0.9928 |
| ROC-AUC      | 0.9982 |

---

## ⚡ Model Optimization (Quantization)

- Applied **TensorFlow Lite dynamic range quantization**
- Achievements:
  - 🔽 ~90% model size reduction (665 KB → 66 KB)  
  - ⚡ ~19.9× faster inference  
  - ✅ No performance degradation  

---

# 🎯 Applications
- Intrusion Detection Systems (IDS)
- Cybersecurity anomaly detection
- Embedded AI systems
- Real-time network monitoring

---


# 🔮 Future Work
- Full integer quantization for further optimization
- Evaluation on modern datasets (CICIDS2017, UNSW-NB15)
- Lightweight/sparse AutoEncoder architectures

# 👨‍💻 Author

Abdul Basit
MS Artificial Intelligence | NUST CEME
AI Engineer | Embedded ML | Deep Learning
