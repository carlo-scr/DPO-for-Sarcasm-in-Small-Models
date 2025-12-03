"""
Fine-tune facebook/MobileLLM-R1.5-360M on GEN-sarc-notsarc dataset using LoRA (Phase 1: SFT).
This script trains on the GEN training split to learn general sarcasm patterns.
Phase 2 (DPO) will use the same training split for preference alignment.

MobileLLM-R1.5-360M is a compact 360M parameter model optimized for mobile deployment.
"""

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import json
import os

def load_and_prepare_data(csv_path, tokenizer, max_length=256, sample_size=None):
    """Load and prepare the dataset for training."""
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check dataset format
    if 'class' in df.columns and 'text' in df.columns:
        # GEN dataset format
        print("Detected GEN-sarc-notsarc dataset format")
        # Convert to standard format
        df['label'] = (df['class'] == 'sarc').astype(int)
        text_col = 'text'
        label_col = 'label'
    elif 'comment' in df.columns:
        # SARC dataset format
        text_col = 'comment'
        label_col = 'label'
        print("Detected SARC dataset format")
    elif 'tweet' in df.columns:
        # iSarcasm dataset format
        df = df.set_index(df.columns[0]) if df.columns[0] == 'Unnamed: 0' else df
        text_col = 'tweet'
        label_col = 'sarcastic'
        print("Detected iSarcasm dataset format")
    else:
        raise ValueError("Unknown dataset format")
    
    # Sample if specified
    if sample_size and len(df) > sample_size:
        # Sample while maintaining class balance
        sarc_df = df[df[label_col] == 1]
        notsarc_df = df[df[label_col] == 0]
        
        n_per_class = sample_size // 2
        sarc_sample = sarc_df.sample(n=min(n_per_class, len(sarc_df)), random_state=42)
        notsarc_sample = notsarc_df.sample(n=min(n_per_class, len(notsarc_df)), random_state=42)
        
        df = pd.concat([sarc_sample, notsarc_sample]).sample(frac=1, random_state=42)
        print(f"Sampled {len(df)} examples from dataset (balanced)")
    
    print(f"Total samples: {len(df)}")
    print(f"Sarcastic: {df[label_col].sum()}, Non-sarcastic: {len(df) - df[label_col].sum()}")
    
    # Create prompts with labels
    def create_training_prompt(row):
        text = row[text_col]
        label = "Yes" if row[label_col] == 1 else "No"
        
        prompt = f"""Is the following text sarcastic? Answer with only 'Yes' or 'No'.

Text: {text}

Answer: {label}"""
        return prompt
    
    df['text'] = df.apply(create_training_prompt, axis=1)
    
    # Split into train/validation (80/20)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Convert to HF Dataset
    train_dataset = Dataset.from_pandas(train_df[['text']])
    val_dataset = Dataset.from_pandas(val_df[['text']])
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
    
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    return train_dataset, val_dataset

def setup_lora_model(model_name="facebook/MobileLLM-R1.5-360M"):
    """Setup model with LoRA adapters."""
    print(f"Loading model: {model_name}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True  # MobileLLM may require this
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # MobileLLM may not have a pad token set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure LoRA - adjust target modules for MobileLLM architecture
    # MobileLLM uses standard transformer attention layers
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention layers
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Enable gradient checkpointing for memory efficiency
    model.enable_input_require_grads()
    
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_model(model, tokenizer, train_dataset, val_dataset, output_dir="./mobilellm_sarcasm_lora"):
    """Train the model with LoRA."""
    
    # Training arguments (optimized for memory efficiency)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=2,  # Reduced to 2 for memory
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size of 16
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=1,  # Only keep 1 checkpoint to save disk space
        fp16=False,  # Disable FP16 on MPS (can cause issues)
        report_to="none",
        load_best_model_at_end=False,  # Disable to save memory
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        dataloader_num_workers=0,  # Disable parallel loading to save memory
        max_grad_norm=1.0,
        disable_tqdm=False,
        logging_first_step=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return trainer

def main():
    # Setup
    model_name = "facebook/MobileLLM-R1.5-360M"
    
    # PHASE 1: Train on GEN dataset training split
    gen_train_path = "data/splits/gen_train.csv"
    output_dir = "models/mobilellm_sft"
    
    print("="*70)
    print("PHASE 1: Supervised Fine-Tuning MobileLLM on GEN Dataset (Training Split)")
    print("="*70)
    print(f"Model: {model_name} (360M parameters)")
    print("Strategy: Train on GEN training split for sarcasm detection")
    print("Test set (data/splits/gen_test.csv) is held-out for evaluation only")
    print("Next Phase: DPO on same training data for preference alignment")
    print("="*70)
    
    # Check if split exists
    if not os.path.exists(gen_train_path):
        # Try parent directory
        parent_path = os.path.join("..", gen_train_path)
        if os.path.exists(parent_path):
            gen_train_path = parent_path
        else:
            print(f"\n❌ Training split not found at {gen_train_path}")
            print("Run 'python scripts/split_gen_dataset.py' first to create train/test splits")
            return
    
    # Also fix output dir if running from scripts folder
    if not os.path.isabs(output_dir) and not os.path.exists(os.path.dirname(output_dir) or '.'):
        parent_output = os.path.join("..", output_dir)
        output_dir = parent_output
    
    # Load model and tokenizer
    model, tokenizer = setup_lora_model(model_name)
    
    # Prepare data
    # For full training, set sample_size=None
    # For faster training, use sample_size (e.g., 2000, 4000)
    train_dataset, val_dataset = load_and_prepare_data(
        gen_train_path, 
        tokenizer,
        sample_size=4000  # Using 4k samples for reasonable training time
        # Set to None to use full training set (~5k samples)
    )
    
    # Train
    trainer = train_model(model, tokenizer, train_dataset, val_dataset, output_dir)
    
    # Save training summary
    summary = {
        "model": model_name,
        "parameters": "360M",
        "dataset": "GEN-sarc-notsarc (training split)",
        "method": "SFT with LoRA",
        "lora_r": 16,
        "lora_alpha": 32,
        "sample_size": 4000,
        "epochs": 2,
        "output_dir": output_dir
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Model saved to: {output_dir}")
    print(f"Test set remains untouched at: data/splits/gen_test.csv")

if __name__ == "__main__":
    main()
