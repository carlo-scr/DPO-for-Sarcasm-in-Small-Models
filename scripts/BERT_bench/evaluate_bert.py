"""
Evaluate BERT model on GEN test dataset
"""

import json
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report,
    confusion_matrix
)
import pandas as pd
import numpy as np
from tqdm import tqdm

# Configuration
class Config:
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "splits"
    MODEL_DIR = PROJECT_ROOT / "models" / "bert_sarcasm"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # Evaluation settings
    MAX_LENGTH = 128
    BATCH_SIZE = 32
    NUM_SAMPLES = 500  # Number of samples to evaluate
    
    # Device
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {Config.DEVICE}")

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
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text
        }

def load_test_data(num_samples=None):
    """Load test dataset"""
    print(f"Loading test data from {Config.DATA_DIR}...")
    test_df = pd.read_csv(Config.DATA_DIR / "gen_test.csv")
    
    # Sample if requested
    if num_samples and num_samples < len(test_df):
        test_df = test_df.sample(n=num_samples, random_state=42).reset_index(drop=True)
        print(f"Sampled {num_samples} examples from test set")
    
    # Convert labels to binary (0: notsarc, 1: sarc)
    label_map = {"notsarc": 0, "sarc": 1}
    test_df['label'] = test_df['class'].map(label_map)
    
    print(f"Test samples: {len(test_df)}")
    print(f"Class distribution: {test_df['label'].value_counts().to_dict()}")
    
    return test_df

def evaluate_model(model, data_loader, device):
    """Evaluate the model and return detailed results"""
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_texts = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get predictions and probabilities
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_texts.extend(batch['text'])
    
    return {
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'labels': all_labels,
        'texts': all_texts
    }

def calculate_metrics(labels, predictions, probabilities):
    """Calculate comprehensive metrics"""
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Confidence scores
    confidences = [probs[pred] for probs, pred in zip(probabilities, predictions)]
    avg_confidence = np.mean(confidences)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'avg_confidence': float(avg_confidence),
        'per_class': {
            'notsarc': {
                'precision': float(precision_per_class[0]),
                'recall': float(recall_per_class[0]),
                'f1': float(f1_per_class[0]),
                'support': int(support[0])
            },
            'sarc': {
                'precision': float(precision_per_class[1]),
                'recall': float(recall_per_class[1]),
                'f1': float(f1_per_class[1]),
                'support': int(support[1])
            }
        },
        'confusion_matrix': {
            'true_negative': int(cm[0][0]),
            'false_positive': int(cm[0][1]),
            'false_negative': int(cm[1][0]),
            'true_positive': int(cm[1][1])
        }
    }

def find_misclassifications(results):
    """Find and analyze misclassified examples"""
    misclassified = []
    
    for i, (pred, label, text, probs) in enumerate(zip(
        results['predictions'],
        results['labels'],
        results['texts'],
        results['probabilities']
    )):
        if pred != label:
            confidence = probs[pred]
            misclassified.append({
                'index': i,
                'text': text,
                'true_label': 'sarc' if label == 1 else 'notsarc',
                'predicted_label': 'sarc' if pred == 1 else 'notsarc',
                'confidence': float(confidence),
                'prob_notsarc': float(probs[0]),
                'prob_sarc': float(probs[1])
            })
    
    return misclassified

def main():
    print("=" * 60)
    print("BERT Model Evaluation on GEN Test Set")
    print("=" * 60)
    
    # Check if model exists
    if not Config.MODEL_DIR.exists():
        print(f"\n❌ Error: Model not found at {Config.MODEL_DIR}")
        print("Please train the model first using finetune_bert.py")
        return
    
    # Load test data
    test_df = load_test_data(num_samples=Config.NUM_SAMPLES)
    
    # Load tokenizer and model
    print(f"\nLoading model from {Config.MODEL_DIR}...")
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(Config.MODEL_DIR)
    model.to(Config.DEVICE)
    model.eval()
    
    print("✓ Model loaded successfully")
    
    # Create dataset and dataloader
    test_dataset = SarcasmDataset(
        test_df['text'].values,
        test_df['label'].values,
        tokenizer,
        Config.MAX_LENGTH
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
    
    # Evaluate
    results = evaluate_model(model, test_loader, Config.DEVICE)
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    metrics = calculate_metrics(
        results['labels'],
        results['predictions'],
        results['probabilities']
    )
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Avg Confidence: {metrics['avg_confidence']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"  Not Sarcastic:")
    print(f"    Precision: {metrics['per_class']['notsarc']['precision']:.4f}")
    print(f"    Recall:    {metrics['per_class']['notsarc']['recall']:.4f}")
    print(f"    F1 Score:  {metrics['per_class']['notsarc']['f1']:.4f}")
    print(f"    Support:   {metrics['per_class']['notsarc']['support']}")
    
    print(f"  Sarcastic:")
    print(f"    Precision: {metrics['per_class']['sarc']['precision']:.4f}")
    print(f"    Recall:    {metrics['per_class']['sarc']['recall']:.4f}")
    print(f"    F1 Score:  {metrics['per_class']['sarc']['f1']:.4f}")
    print(f"    Support:   {metrics['per_class']['sarc']['support']}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"                  Predicted")
    print(f"                  Not Sarc  Sarcastic")
    print(f"  Actual Not Sarc    {cm['true_negative']:4d}      {cm['false_positive']:4d}")
    print(f"  Actual Sarcastic   {cm['false_negative']:4d}      {cm['true_positive']:4d}")
    
    # Classification report
    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(
        results['labels'],
        results['predictions'],
        target_names=['Not Sarcastic', 'Sarcastic'],
        digits=4
    ))
    
    # Find misclassifications
    misclassified = find_misclassifications(results)
    print(f"\nMisclassified Examples: {len(misclassified)}/{len(results['labels'])} ({len(misclassified)/len(results['labels'])*100:.2f}%)")
    
    if misclassified:
        print("\nSample Misclassifications (first 5):")
        for i, example in enumerate(misclassified[:5], 1):
            print(f"\n{i}. Text: {example['text'][:100]}...")
            print(f"   True: {example['true_label']} | Predicted: {example['predicted_label']} | Confidence: {example['confidence']:.4f}")
    
    # Save results
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = Config.RESULTS_DIR / "bert_evaluation_results.json"
    
    output_data = {
        'evaluation_date': datetime.now().isoformat(),
        'model_path': str(Config.MODEL_DIR),
        'test_samples': len(results['labels']),
        'metrics': metrics,
        'misclassified_count': len(misclassified),
        'misclassified_examples': misclassified[:20]  # Save first 20 misclassifications
    }
    
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Evaluation complete!")
    print(f"📊 Results saved to: {results_file}")
    print(f"🎯 Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"🏆 F1 Score: {metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
