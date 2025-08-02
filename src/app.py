import os
from flask import Flask, request, render_template
import torch
import pickle
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification


template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
app = Flask(__name__, template_folder=template_dir)


model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'pretrained_model'))
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()


with open(os.path.join(model_path, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)


data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical_notes.xlsx'))
df = pd.read_excel(data_path)
code_to_desc = dict(zip(df['icd10_code'], df['description']))


label_to_code = {i: code for i, code in enumerate(label_encoder.classes_)}
label_to_desc = {i: code_to_desc.get(code, "Unknown condition") for i, code in label_to_code.items()}


def predict_icd_code(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    icd_code = label_to_code[pred]
    desc = label_to_desc.get(pred, "Unknown condition")
    return icd_code, desc


@app.route("/", methods=["GET", "POST"])

def index():
    prediction = None  
    if request.method == "POST":
        user_input = request.form["note"]
        if user_input.strip():  
            code, desc = predict_icd_code(user_input)
            prediction = f"ICD-10 Code: {code} – {desc}"
    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
