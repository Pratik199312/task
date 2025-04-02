import os
import torch
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

# Start MLflow tracking 
mlflow.set_tracking_uri("http://localhost:5000")  # Set MLflow Server URI
mlflow.set_experiment("POS_Product_Classification")

# Load training data
data = pd.read_csv("Training_Data.csv", engine="python")

# Encode category labels
le = LabelEncoder()
data['Category'] = le.fit_transform(data['Category'])

# Save LabelEncoder mappings for later use
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
inverse_label_mapping = {v: k for k, v in label_mapping.items()}
pd.DataFrame.from_dict(label_mapping, orient="index").to_csv("label_mapping.csv", header=False)

# Train-test split
X = list(data["product_description"])
y = list(data["Category"])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

# Tokenize data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)

# Define metrics function
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    return {
        "accuracy": accuracy_score(labels, pred),
        "precision": precision_score(labels, pred, average="weighted"),
        "recall": recall_score(labels, pred, average="weighted"),
        "f1": f1_score(labels, pred, average="weighted"),
    }

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_mapping))
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Define training arguments
args = TrainingArguments(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir="logs",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
) 

# Start MLflow Run
with mlflow.start_run():
    mlflow.log_params({"epochs": 3, "batch_size": 8, "model": "bert-base-uncased"})
    
    # Train model
    trainer.train()

    # Evaluate model
    eval_results = trainer.evaluate()
    
    # Log Metrics
    mlflow.log_metrics(eval_results)

    # Save model
    model.save_pretrained("trained_model")
    tokenizer.save_pretrained("trained_model")

    # Log Model to MLflow
    mlflow.pytorch.log_model(model, "bert_model")

print("✅ Model training complete. MLflow tracking enabled.")
