import pandas as pd
import pickle
from transformers import BertTokenizer, BertForSequenceClassification
import torch


model_path = "./models/pretrained_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()


with open(f"{model_path}/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


df = pd.read_excel("data/testing_notes.xlsx")  


code_to_desc = dict(zip(df['icd10_code'], df['description']))
label_to_code = {i: code for i, code in enumerate(label_encoder.classes_)}
label_to_desc = {i: code_to_desc.get(code, "Unknown condition") for i, code in enumerate(label_encoder.classes_)}


def predict_icd_code(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    icd_code = label_to_code[pred]
    desc = label_to_desc.get(pred, "Unknown condition")
    return icd_code, desc


correct = 0
total = min(100, len(df))

for i in range(total):
    note = df.loc[i, 'clinical_notes']
    true_code = df.loc[i, 'icd10_code']
    
    predicted_code, predicted_desc = predict_icd_code(note)
    
    print(f"\nNote: {note}")
    print(f"Predicted ICD-10 Code: {predicted_code} ({predicted_desc})")
    print(f"Actual ICD-10 Code:    {true_code} ({df.loc[i, 'description']})")
    
    if predicted_code == true_code:
        correct += 1


accuracy = correct / total
print(f"\nAccuracy on test dataset: {accuracy*100:.2f}%")
