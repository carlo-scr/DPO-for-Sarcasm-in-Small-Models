"""
Direct Preference Optimization (DPO) for sarcasm detection - SARC Dataset Version.

This script trains DPO on SARC dataset after SFT on SARC.
- Uses SARC data from data/SARC/splits/dpo_data.csv
- Trains with comprehensive diagnostics and metrics tracking
- Does NOT merge adapters - keeps LoRA structure throughout
"""

import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from datasets import Dataset 
from peft import PeftModel
from trl import DPOTrainer, DPOConfig
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm


class DPOMetricsLogger:
    """Comprehensive logger for DPO training diagnostics."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.batch_history = []
        self.epoch_history = []
        self.kl_history = []
        os.makedirs(output_dir, exist_ok=True)
    
    def log_batch(self, step, metrics):
        """Log per-batch metrics."""
        entry = {'step': step, 'timestamp': datetime.now().isoformat(), **metrics}
        self.batch_history.append(entry)
        
        # Save incrementally
        with open(f"{self.output_dir}/dpo_batch_metrics.json", 'w') as f:
            json.dump(self.batch_history, f, indent=2)
    
    def log_epoch(self, epoch, metrics):
        """Log per-epoch metrics."""
        entry = {'epoch': epoch, 'timestamp': datetime.now().isoformat(), **metrics}
        self.epoch_history.append(entry)
        
        with open(f"{self.output_dir}/dpo_epoch_metrics.json", 'w') as f:
            json.dump(self.epoch_history, f, indent=2)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch} SUMMARY:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
        print(f"{'='*80}\n")
    
    def log_kl(self, step, kl_div):
        """Log KL divergence."""
        self.kl_history.append({'step': step, 'kl_div': kl_div})
    
    def plot_training_curves(self):
        """Create comprehensive visualization of DPO training."""
        if len(self.epoch_history) < 2:
            return
        
        epochs = [h['epoch'] for h in self.epoch_history]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss curves
        if 'train_loss' in self.epoch_history[0]:
            train_losses = [h.get('train_loss') for h in self.epoch_history]
            axes[0, 0].plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
        if 'eval_loss' in self.epoch_history[0]:
            eval_losses = [h.get('eval_loss') for h in self.epoch_history]
            axes[0, 0].plot(epochs, eval_losses, 'r-o', label='Eval Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=11)
        axes[0, 0].set_ylabel('Loss', fontsize=11)
        axes[0, 0].set_title('DPO Loss', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # KL Divergence
        if self.kl_history:
            steps = [h['step'] for h in self.kl_history]
            kl_divs = [h['kl_div'] for h in self.kl_history]
            axes[0, 1].plot(steps, kl_divs, 'purple', linewidth=1.5)
            axes[0, 1].set_xlabel('Training Step', fontsize=11)
            axes[0, 1].set_ylabel('KL Divergence', fontsize=11)
            axes[0, 1].set_title('KL Divergence from Reference Model', fontsize=12, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Reward Ratio
        if 'avg_reward_ratio' in self.epoch_history[0]:
            ratios = [h.get('avg_reward_ratio') for h in self.epoch_history]
            axes[0, 2].plot(epochs, ratios, 'g-o', label='Chosen/Rejected Ratio', linewidth=2)
            axes[0, 2].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Equal Rewards')
            axes[0, 2].set_xlabel('Epoch', fontsize=11)
            axes[0, 2].set_ylabel('Reward Ratio', fontsize=11)
            axes[0, 2].set_title('Chosen vs Rejected Rewards', fontsize=12, fontweight='bold')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # F1 Score
        if 'eval_f1' in self.epoch_history[0]:
            f1_scores = [h.get('eval_f1') for h in self.epoch_history]
            axes[1, 0].plot(epochs, f1_scores, 'orange', marker='o', linewidth=2)
            axes[1, 0].set_xlabel('Epoch', fontsize=11)
            axes[1, 0].set_ylabel('F1 Score', fontsize=11)
            axes[1, 0].set_title('Validation F1 Score', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Accuracy
        if 'eval_accuracy' in self.epoch_history[0]:
            accuracies = [h.get('eval_accuracy') for h in self.epoch_history]
            axes[1, 1].plot(epochs, accuracies, 'b-o', linewidth=2)
            axes[1, 1].set_xlabel('Epoch', fontsize=11)
            axes[1, 1].set_ylabel('Accuracy', fontsize=11)
            axes[1, 1].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Precision & Recall
        if 'eval_precision' in self.epoch_history[0] and 'eval_recall' in self.epoch_history[0]:
            precisions = [h.get('eval_precision') for h in self.epoch_history]
            recalls = [h.get('eval_recall') for h in self.epoch_history]
            axes[1, 2].plot(epochs, precisions, 'b-o', label='Precision', linewidth=2)
            axes[1, 2].plot(epochs, recalls, 'r-o', label='Recall', linewidth=2)
            axes[1, 2].set_xlabel('Epoch', fontsize=11)
            axes[1, 2].set_ylabel('Score', fontsize=11)
            axes[1, 2].set_title('Precision & Recall', fontsize=12, fontweight='bold')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f"{self.output_dir}/dpo_training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved DPO training curves to {plot_path}")


def prepare_dpo_dataset(csv_path, split_ratio=0.9):
    """
    Prepare SARC dataset with preference pairs.
    
    Args:
        csv_path: Path to SARC DPO data CSV (data/SARC/splits/dpo_data.csv)
        split_ratio: Train/val split ratio
    
    Returns:
        train_dataset, val_dataset, val_df (for evaluation)
    """
    print(f"\n{'='*80}")
    print("LOADING SARC DPO DATASET")
    print(f"{'='*80}")
    print(f"Source: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # SARC format: 'comment' (text) and 'label' (0=not sarcastic, 1=sarcastic)
    # Rename for consistency
    df = df.rename(columns={'label': 'sarcastic', 'comment': 'text'})
    
    print(f"\nTotal samples: {len(df)}")
    print(f"  Sarcastic: {df['sarcastic'].sum()} ({df['sarcastic'].mean():.1%})")
    print(f"  Non-sarcastic: {len(df) - df['sarcastic'].sum()} ({1-df['sarcastic'].mean():.1%})")
    
    # Subsample to 1000 examples (500 per class for balance)
    sarcastic_df = df[df['sarcastic'] == 1]
    non_sarcastic_df = df[df['sarcastic'] == 0]
    
    sarcastic_sample = sarcastic_df.sample(n=min(500, len(sarcastic_df)), random_state=42)
    non_sarcastic_sample = non_sarcastic_df.sample(n=min(500, len(non_sarcastic_df)), random_state=42)
    
    df = pd.concat([sarcastic_sample, non_sarcastic_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nSubsampled to 1000 examples (balanced):")
    print(f"  Sarcastic: {df['sarcastic'].sum()} ({df['sarcastic'].mean():.1%})")
    print(f"  Non-sarcastic: {len(df) - df['sarcastic'].sum()} ({1-df['sarcastic'].mean():.1%})")
    
    # Create preference pairs
    dpo_data = []
    
    for _, row in df.iterrows():
        text = row['text']
        is_sarcastic = row['sarcastic']
        
        # Base prompt
        prompt = f"""Is the following text sarcastic? Sarcasm involves saying the opposite of what is meant, often using irony, exaggeration, or mockery.

Text: "{text}"

Answer:"""
        
        if is_sarcastic == 1:
            # Sarcastic: Chosen is "Yes", Rejected is "No"
            chosen = " Yes. This text is sarcastic based on contextual cues."
            rejected = " No. This appears to be a literal statement without sarcastic intent."
        else:
            # Not sarcastic: Chosen is "No", Rejected is "Yes"
            chosen = " No. This is a straightforward, literal statement."
            rejected = " Yes. This text seems sarcastic."
        
        dpo_data.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected
        })
    
    # Shuffle
    np.random.seed(42)
    indices = np.random.permutation(len(dpo_data))
    dpo_data = [dpo_data[i] for i in indices]
    df = df.iloc[indices].reset_index(drop=True)
    
    # Split into train/val
    split_idx = int(len(dpo_data) * split_ratio)
    train_data = dpo_data[:split_idx]
    val_data = dpo_data[split_idx:]
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"\n✓ Created {len(train_data)} training pairs")
    print(f"✓ Created {len(val_data)} validation pairs")
    print(f"{'='*80}\n")
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return train_dataset, val_dataset, val_df


def evaluate_dpo_model(model, tokenizer, val_df, device):
    """
    Evaluate DPO model with comprehensive metrics.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        val_df: Validation DataFrame with text/sarcastic columns
        device: Device to use
    
    Returns:
        Dictionary of metrics
    """
    print("\nEvaluating DPO model on validation set...")
    
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Evaluating"):
            text = row['text']
            label = row['sarcastic']
            
            prompt = f"""Is the following text sarcastic? Answer with 'Yes' or 'No'.

Text: "{text}"

Answer:"""
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7, do_sample=False)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract prediction - look for the answer after the prompt
            # Split by "Answer:" and take the part after it
            if "Answer:" in response:
                answer_part = response.split("Answer:")[-1].strip().lower()
            else:
                answer_part = response.lower()
            
            # Check first few words for yes/no
            first_words = answer_part.split()[:15]
            first_text = ' '.join(first_words)
            
            # Check for sarcastic indicators
            if 'not sarcastic' in first_text or 'no.' in first_text or (first_words and first_words[0] == 'no'):
                pred = 0
            elif 'yes' in first_text or 'sarcastic' in first_text.replace('not sarcastic', ''):
                pred = 1
            else:
                # Default to 0 if unclear
                pred = -1
            
            predictions.append(pred)
            labels.append(label)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    print(f"\nValidation Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Per-class breakdown
    print(f"\nPer-class Report:")
    print(classification_report(labels, predictions, target_names=['Not Sarcastic', 'Sarcastic']))
    
    return {
        'eval_accuracy': accuracy,
        'eval_precision': precision,
        'eval_recall': recall,
        'eval_f1': f1
    }


def train_dpo_sarc(
    sft_model_path="models/sft",
    dpo_data_path="data/SARC/splits/dpo_data.csv",
    output_dir="models/dpo_enhanced",
    beta=0.1,
    num_epochs=3,
    learning_rate=5e-5
):
    """
    Train DPO model on SARC dataset with comprehensive diagnostics.
    
    Args:
        sft_model_path: Path to SFT model adapters
        dpo_data_path: Path to SARC DPO data CSV
        output_dir: Output directory for DPO model
        beta: DPO beta parameter (KL regularization strength)
        num_epochs: Number of training epochs
        learning_rate: Learning rate
    
    Returns:
        Trainer object
    """
    print(f"\n{'='*80}")
    print(f"DPO TRAINING CONFIGURATION - SARC DATASET")
    print(f"{'='*80}")
    print(f"SFT Model: {sft_model_path}")
    print(f"DPO Data: {dpo_data_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Beta (KL strength): {beta}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"{'='*80}\n")
    
    # Setup metrics logger
    metrics_logger = DPOMetricsLogger(output_dir)
    
    # Load base model and tokenizer
    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading base model: {base_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    # Load SFT adapter (keep as adapter, DON'T merge!)
    print(f"Loading SFT adapter from: {sft_model_path}")
    model = PeftModel.from_pretrained(base_model, sft_model_path)
    print("✓ Training model continues from SFT's existing LoRA adapters")
    
    # Create reference model (merged version for stable reference)
    print("Creating reference model (merged SFT for KL tracking)...")
    reference_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        ),
        sft_model_path
    )
    reference_model = reference_model.merge_and_unload()
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    print("✓ Reference model created (frozen merged SFT model)")
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Load and prepare datasets
    train_dataset, val_dataset, val_df = prepare_dpo_dataset(dpo_data_path)
    
    # DPO training arguments
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=learning_rate,
        warmup_steps=50,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        bf16=False,
        report_to="none",
        beta=beta,
        max_length=512,
        max_prompt_length=256,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # Custom callback for epoch-level evaluation
    class DPOEvaluationCallback(TrainerCallback):
        def __init__(self, val_df, tokenizer, metrics_logger):
            self.val_df = val_df
            self.tokenizer = tokenizer
            self.metrics_logger = metrics_logger
            self.best_f1 = 0.0
        
        def on_epoch_end(self, args, state, control, model, **kwargs):
            device = next(model.parameters()).device
            metrics = evaluate_dpo_model(model, self.tokenizer, self.val_df, device)
            
            # Safely get train loss from log history
            if state.log_history:
                metrics['train_loss'] = state.log_history[-1].get('loss', 0)
            else:
                metrics['train_loss'] = 0
            
            epoch = int(state.epoch)
            self.metrics_logger.log_epoch(epoch, metrics)
            
            # Early stopping based on F1
            if metrics['eval_f1'] > self.best_f1:
                self.best_f1 = metrics['eval_f1']
                print(f"✓ New best F1: {self.best_f1:.4f}")
            
            return control
        
        def on_train_end(self, args, state, control, **kwargs):
            self.metrics_logger.plot_training_curves()
    
    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=reference_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[DPOEvaluationCallback(val_df, tokenizer, metrics_logger)]
    )
    
    # Train
    print(f"\n{'='*80}")
    print("STARTING DPO TRAINING")
    print(f"{'='*80}\n")
    
    trainer.train()
    
    # Save final model (adapter only, not merged)
    print(f"\n✓ Saving DPO model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training summary
    summary = {
        "sft_model": sft_model_path,
        "dpo_dataset": dpo_data_path,
        "dataset_name": "SARC",
        "beta": beta,
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "training_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "final_epoch": int(trainer.state.epoch),
        "training_completed": datetime.now().isoformat()
    }
    
    with open(f"{output_dir}/dpo_training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("DPO TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"✓ Model saved to: {output_dir}")
    print(f"✓ Metrics saved to: {output_dir}/dpo_epoch_metrics.json")
    print(f"✓ Batch logs saved to: {output_dir}/dpo_batch_metrics.json")
    print(f"✓ Training curves: {output_dir}/dpo_training_curves.png")
    print(f"✓ Summary: {output_dir}/dpo_training_summary.json")
    
    return trainer


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DPO Training on SARC Dataset")
    parser.add_argument("--sft_model", type=str, default="models/sft", help="Path to SFT model")
    parser.add_argument("--dpo_data", type=str, default="data/SARC/splits/dpo_data.csv", help="Path to DPO data CSV")
    parser.add_argument("--output_dir", type=str, default="models/dpo_enhanced", help="Output directory")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    train_dpo_sarc(
        sft_model_path=args.sft_model,
        dpo_data_path=args.dpo_data,
        output_dir=args.output_dir,
        beta=args.beta,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()
