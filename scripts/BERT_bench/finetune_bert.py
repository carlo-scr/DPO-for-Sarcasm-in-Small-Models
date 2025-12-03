"""
Fine-tune BERT for Sarcasm Detection on GEN Dataset
"""

import os
import json
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
class Config:
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "splits"
    OUTPUT_DIR = PROJECT_ROOT / "models" / "bert_sarcasm"
    
    # Model settings
    MODEL_NAME = "bert-base-uncased"
    MAX_LENGTH = 128
    
    # Training settings
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 500
    MAX_GRAD_NORM = 1.0
    
    # Device
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {Config.DEVICE}")

# Create output directory
Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Custom Dataset
class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data():
    """Load and prepare the GEN dataset"""
    print("Loading data...")
    train_df = pd.read_csv(Config.DATA_DIR / "gen_train.csv")
    test_df = pd.read_csv(Config.DATA_DIR / "gen_test.csv")
    
    # Convert labels to binary (0: notsarc, 1: sarc)
    label_map = {"notsarc": 0, "sarc": 1}
    train_df['label'] = train_df['class'].map(label_map)
    test_df['label'] = test_df['class'].map(label_map)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Class distribution (train): {train_df['label'].value_counts().to_dict()}")
    print(f"Class distribution (test): {test_df['label'].value_counts().to_dict()}")
    
    return train_df, test_df

def train_epoch(model, data_loader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(data_loader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy

def evaluate(model, data_loader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary'
    )
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predictions,
        'true_labels': true_labels
    }

def main():
    print("=" * 60)
    print("BERT Fine-tuning for Sarcasm Detection")
    print("=" * 60)
    
    # Load data
    train_df, test_df = load_data()
    
    # Initialize tokenizer
    print("\nInitializing tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Create datasets
    train_dataset = SarcasmDataset(
        train_df['text'].values,
        train_df['label'].values,
        tokenizer,
        Config.MAX_LENGTH
    )
    
    test_dataset = SarcasmDataset(
        test_df['text'].values,
        test_df['label'].values,
        tokenizer,
        Config.MAX_LENGTH
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
    
    # Initialize model
    model = BertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=2
    )
    model.to(Config.DEVICE)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    total_steps = len(train_loader) * Config.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config.WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"\nStarting training for {Config.NUM_EPOCHS} epochs...")
    print(f"Total training steps: {total_steps}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Learning rate: {Config.LEARNING_RATE}")
    
    best_f1 = 0
    training_history = []
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print(f"{'=' * 60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, Config.DEVICE
        )
        
        # Evaluate
        eval_results = evaluate(model, test_loader, Config.DEVICE)
        
        # Log results
        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'eval_loss': eval_results['loss'],
            'eval_accuracy': eval_results['accuracy'],
            'eval_precision': eval_results['precision'],
            'eval_recall': eval_results['recall'],
            'eval_f1': eval_results['f1']
        }
        training_history.append(epoch_results)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Eval Loss: {eval_results['loss']:.4f} | Eval Acc: {eval_results['accuracy']:.4f}")
        print(f"Precision: {eval_results['precision']:.4f} | Recall: {eval_results['recall']:.4f} | F1: {eval_results['f1']:.4f}")
        
        # Save best model
        if eval_results['f1'] > best_f1:
            best_f1 = eval_results['f1']
            print(f"\n🎯 New best F1 score: {best_f1:.4f}! Saving model...")
            
            model.save_pretrained(Config.OUTPUT_DIR)
            tokenizer.save_pretrained(Config.OUTPUT_DIR)
    
    # Final evaluation and classification report
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    final_results = evaluate(model, test_loader, Config.DEVICE)
    
    print("\nClassification Report:")
    print(classification_report(
        final_results['true_labels'],
        final_results['predictions'],
        target_names=['Not Sarcastic', 'Sarcastic']
    ))
    
    # Save training history and results
    results_file = Config.OUTPUT_DIR / "training_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'model_name': Config.MODEL_NAME,
                'max_length': Config.MAX_LENGTH,
                'batch_size': Config.BATCH_SIZE,
                'learning_rate': Config.LEARNING_RATE,
                'num_epochs': Config.NUM_EPOCHS,
                'device': str(Config.DEVICE)
            },
            'training_history': training_history,
            'final_results': {
                'accuracy': final_results['accuracy'],
                'precision': final_results['precision'],
                'recall': final_results['recall'],
                'f1': final_results['f1'],
                'best_f1': best_f1
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"📁 Model saved to: {Config.OUTPUT_DIR}")
    print(f"📊 Results saved to: {results_file}")
    print(f"🏆 Best F1 Score: {best_f1:.4f}")

if __name__ == "__main__":
    main()
