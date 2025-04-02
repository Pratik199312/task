import torch
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np 
 
# Load trained model and tokenizer
model = BertForSequenceClassification.from_pretrained("trained_model")
tokenizer = BertTokenizer.from_pretrained("trained_model")
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval() 

# Load label mappings
label_mapping = pd.read_csv("label_mapping.csv", header=None, index_col=1, squeeze=True).to_dict()
inverse_label_mapping = {v: k for k, v in label_mapping.items()}

# FastAPI app
app = FastAPI()

class InputText(BaseModel):
    product_description: str

@app.post("/predict")
def predict_category(input_data: InputText):
    text = input_data.product_description
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(predictions, dim=1).item()
    
    predicted_category = inverse_label_mapping[predicted_label]
    return {"product_description": text, "predicted_category": predicted_category}

# Run: uvicorn serve_model:app --host 0.0.0.0 --port 8000
