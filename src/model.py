import pandas as pd
import torch
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
import numpy as np



# Convert to tensor
# Import preprocessing function
from preprocess import preprocess_text

# Load dataset
df =  pd.read_excel('data/clinical_notes.xlsx')

df['processed_notes'] = df['clinical_notes'].apply(preprocess_text)

# Label encode ICD-10 codes
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['icd10_code'])

# Train-validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['processed_notes'], df['label'], test_size=0.2, random_state=42
)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_data(texts):
    return tokenizer(
        texts.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt'
    )

train_encodings = tokenize_data(train_texts)
val_encodings = tokenize_data(val_texts)

# Create PyTorch Dataset
class ClinicalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item

    def __len__(self):
        return len(self.labels)
    
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(df['label']),
    y=df['label']
)

class_weights = torch.tensor(class_weights_array, dtype=torch.float)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

train_dataset = ClinicalDataset(train_encodings, train_labels.values)
val_dataset = ClinicalDataset(val_encodings, val_labels.values)

# Load BERT model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_encoder.classes_)
)



# Training config
training_args = TrainingArguments(
    output_dir='./models/pretrained_model',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train model
trainer.train()

final_path = './models/pretrained_model'

# Save the trained model
model.save_pretrained(final_path)

# Save the tokenizer
tokenizer.save_pretrained(final_path)

# Save the label encoder
import pickle
with open(f"{final_path}/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)