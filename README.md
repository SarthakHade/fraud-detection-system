# 💳 AI-Powered Fraud Detection System

---

## 🚀 Overview
This project detects fraudulent financial transactions using **Machine Learning + Behavioral Analysis + Simulation Techniques**.

It not only predicts fraud but also provides:
- Explainable AI insights
- Risk-based decision support
- Future financial risk simulation
- Industry benchmarking (privacy-preserving)

---

## 🧠 Core Features

### 🔍 1. Fraud Detection
- Predicts whether a transaction is **Fraud / Safe**
- Outputs **risk score (0–100%)**

### 🧩 2. Behavioral Analysis
- Detects unusual patterns:
  - High transaction amount
  - Night transactions
  - Frequent transfers
  - New receiver risk

### 💡 3. Explainable AI
- Gives clear reasons:
  - "High amount"
  - "Unusual timing"
  - "New receiver"

### 📊 4. Monte Carlo Simulation (Brownie Point ⭐)
- Simulates thousands of financial scenarios
- Predicts **company survival probability**
- Adjusts based on fraud risk

### 📈 5. Privacy-Preserving Industry Benchmarking (Brownie Point ⭐)
- Compares company fraud rate with industry
- Uses simulated data (no real data exposure)
- Classifies:
  - 🟢 Normal
  - 🟡 Risky
  - 🔴 Critical

### 💬 6. Smart Chatbot Assistant
- Ask questions like:
  - Why is this fraud?
  - What is risk score?
  - What action should be taken?

---

## 🏗️ Complete Workflow

User Input (UI Form)
↓
Feature Engineering
↓
ML Model Prediction
↓
Fraud Risk Score + Decision
↓
Explainable AI (Reasons)
↓
Monte Carlo Simulation
↓
Industry Benchmark Comparison
↓
Chatbot Interaction


---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Python (Flask)  
- **ML:** Scikit-learn  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py

├── main.py                # Backend API
├── model files           # ML model
├── chatbot.py            # Chatbot logic
├── index.html            # UI
├── data-pipeline.js      # Feature processing
├── test_model.py         # Testing script
├── datasets (.csv)       # Training + testing data

