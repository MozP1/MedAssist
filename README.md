# 🏥 MedAssist: AI-Powered ICD-10 Code Predictor

MedAssist is a machine learning–based web application that predicts ICD-10 codes from user-entered clinical notes or symptoms. It uses a fine-tuned BERT model trained on clinical text data, making it useful for assisting in medical coding, especially in low-resource or rural healthcare setups.

---

## 🚀 Features

- 🔍 Predicts ICD-10 diagnosis codes from clinical notes
- 📖 Provides the full medical description along with the code
- 🧠 Fine-tuned BERT model for sequence classification
- 🌐 Simple and interactive Flask-based web UI
- 💾 Lightweight model trained on 200 cleaned real-like clinical samples

---

---

## 🔧 Installation

Before running the application, install the required dependencies:
pip install -r requirements.txt

## 🧠 Model Training

The model is a fine-tuned version of `bert-base-uncased` from Hugging Face.

### 📊 Data Format

Each row of the CSV dataset includes:
- `clinical_notes`: Text description of patient symptoms
- `icd10_code`: The corresponding ICD-10 code
- `description`: Explanation of the code


### 🏋️‍♀️ Training

Run this to train and save the model:

model.py

### Run the Flask app:

app.py

### Open your browser and go to:

http://127.0.0.1:5000

