"""
Quick DPO evaluation script for MobileLLM models.
Tests the DPO model on the GEN test split to verify it hasn't regressed.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
import os


def evaluate_model(model, tokenizer, test_df, device, model_name="Model"):
    """Evaluate model on test set."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing"):
            text = row['text']
            label = 1 if row['class'] == 'sarc' else 0
            
            prompt = f"""Is the following text sarcastic? Answer with only 'Yes' or 'No'.

Text: {text}

Answer:"""
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer after "Answer:"
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip().lower()
            else:
                answer = response.strip().lower()
            
            # Determine prediction
            if answer.startswith("yes"):
                pred = 1
            elif answer.startswith("no"):
                pred = 0
            else:
                # Default to 0 if unclear
                pred = 0
            
            predictions.append(pred)
            labels.append(label)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    print(f"\nResults for {model_name}:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(labels, predictions, target_names=['Not Sarcastic', 'Sarcastic']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    # Paths
    base_model_name = "facebook/MobileLLM-R1.5-360M"
    sft_model_path = "models/mobilellm_sft"
    dpo_model_path = "models/mobilellm_dpo"
    test_data_path = "data/splits/gen_test.csv"
    
    # Fix relative paths
    if not os.path.exists(test_data_path):
        test_data_path = os.path.join("..", test_data_path)
    if not os.path.exists(sft_model_path):
        sft_model_path = os.path.join("..", sft_model_path)
    if not os.path.exists(dpo_model_path):
        dpo_model_path = os.path.join("..", dpo_model_path)
    
    # Load test data
    print(f"Loading test data from: {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    print(f"Test samples: {len(test_df)}")
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = {}
    
    # Evaluate base model
    print("\n" + "="*60)
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    results['base'] = evaluate_model(base_model, tokenizer, test_df, device, "Base MobileLLM (Zero-shot)")
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Evaluate SFT model
    if os.path.exists(sft_model_path):
        print("\n" + "="*60)
        print("Loading SFT model...")
        base_for_sft = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        sft_model = PeftModel.from_pretrained(base_for_sft, sft_model_path)
        results['sft'] = evaluate_model(sft_model, tokenizer, test_df, device, "SFT MobileLLM")
        del sft_model, base_for_sft
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        print(f"\n⚠️  SFT model not found at {sft_model_path}")
    
    # Evaluate DPO model
    if os.path.exists(dpo_model_path):
        print("\n" + "="*60)
        print("Loading DPO model...")
        base_for_dpo = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        dpo_model = PeftModel.from_pretrained(base_for_dpo, dpo_model_path)
        results['dpo'] = evaluate_model(dpo_model, tokenizer, test_df, device, "DPO MobileLLM")
        del dpo_model, base_for_dpo
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        print(f"\n⚠️  DPO model not found at {dpo_model_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY (MobileLLM)")
    print("="*60)
    for stage, metrics in results.items():
        print(f"{stage.upper():12} Accuracy: {metrics['accuracy']*100:.1f}%  F1: {metrics['f1']:.3f}")
    
    if 'sft' in results and 'dpo' in results:
        acc_diff = (results['dpo']['accuracy'] - results['sft']['accuracy']) * 100
        f1_diff = results['dpo']['f1'] - results['sft']['f1']
        sign = "+" if acc_diff >= 0 else ""
        print(f"\nDPO vs SFT: {sign}{acc_diff:.1f}pp accuracy, {sign}{f1_diff:.3f} F1")
        
        if acc_diff < -2:
            print("⚠️  Warning: DPO shows regression from SFT!")
        elif acc_diff > 0:
            print("✓ DPO improved over SFT")


if __name__ == "__main__":
    main()
