import pandas as pd
import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

def prepare_data(csv_path):
    print("--- Loading Mental Health Dataset ---")
    # Load your Kaggle dataset
    df = pd.read_csv(csv_path)
    
    # Clean the data (Assuming columns 'text' and 'label')
    df = df.dropna()
    
    # Initialize BERT Tokenizer (The first part of your BERT-LSTM-SVM chain)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # This function converts sentences into numbers that BERT understands
    def tokenize_function(text):
        return tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )

    print("--- Data Prepared for BERT Embedding ---")
    return df, tokenizer

# For your project report: 
# BERT handles the 'Context'
# LSTM handles the 'Sequence' (Longer chats)
# SVM handles the 'Classification' (Final Diagnosis)
