"""
Direct Preference Optimization (DPO) for sarcasm detection - Phase 2.
This script uses iSarcasm dataset for preference alignment after SFT on SARC.
Strategy: Refine model with high-quality preference pairs from expert annotations.
"""

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig
import json

def prepare_dpo_dataset(csv_path):
    """
    Prepare iSarcasm dataset in DPO format with rich preference pairs.
    Uses multi-dimensional annotations (irony, satire, overstatement, etc.)
    to create informative chosen/rejected pairs.
    
    For sarcasm detection:
    - Chosen: Correct label with contextual reasoning
    - Rejected: Incorrect label representing common failure modes
    """
    print(f"Loading iSarcasm dataset for DPO from: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    
    print(f"Total samples: {len(df)}")
    print(f"Sarcastic: {df['sarcastic'].sum()}, Non-sarcastic: {len(df) - df['sarcastic'].sum()}")
    
    dpo_data = []
    
    for _, row in df.iterrows():
        text = row['tweet']
        is_sarcastic = row['sarcastic']
        
        # Base prompt
        prompt = f"""Is the following text sarcastic? Answer with only 'Yes' or 'No'.

Text: {text}

Answer:"""
        
        # Create richer chosen/rejected pairs based on sarcasm types
        if is_sarcastic == 1:
            # Collect sarcasm indicators
            indicators = []
            if row.get('irony', 0) == 1:
                indicators.append('irony')
            if row.get('satire', 0) == 1:
                indicators.append('satire')
            if row.get('overstatement', 0) == 1:
                indicators.append('overstatement')
            if row.get('understatement', 0) == 1:
                indicators.append('understatement')
            if row.get('rhetorical_question', 0) == 1:
                indicators.append('rhetorical question')
            
            # Chosen: Correct with reasoning
            if indicators:
                chosen = f" Yes (contains {', '.join(indicators)})"
            else:
                chosen = " Yes"
            
            # Rejected: Common failure - taking literally
            rejected = " No"
            
        else:
            # Non-sarcastic text
            chosen = " No"
            # Rejected: False positive - over-interpreting
            rejected = " Yes"
        
        dpo_data.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected
        })
    
    print(f"Created {len(dpo_data)} DPO preference pairs")
    return Dataset.from_list(dpo_data)

def load_model_for_dpo(base_model_name="Qwen/Qwen2.5-0.5B-Instruct", adapter_path=None):
    """Load model and apply LoRA if adapter path provided."""
    print(f"Loading model: {base_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float32,  # Changed from torch_dtype to dtype
        device_map="auto"
    )
    
    # Load fine-tuned adapter if provided
    if adapter_path:
        print(f"Loading fine-tuned adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        # Apply fresh LoRA config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def train_dpo(csv_path, output_dir="./qwen_sarcasm_dpo", adapter_path=None):
    """Train model using DPO."""
    
    # Prepare dataset
    print("Preparing DPO dataset...")
    dataset = prepare_dpo_dataset(csv_path)
    
    # Split train/val
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Load model
    model, tokenizer = load_model_for_dpo(adapter_path=adapter_path)
    
    # For newer TRL versions, we need a reference model
    ref_model = None  # DPOTrainer will create a copy if needed
    
    # DPO Configuration (using DPOConfig for TRL v0.25+)
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Reduced for memory
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Increased to maintain effective batch
        learning_rate=5e-5,
        warmup_steps=50,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=1,
        bf16=False,  # Disable for MPS
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_length=384,  # Max sequence length for DPO
        max_prompt_length=256,  # Max prompt length for DPO
        beta=0.1,  # DPO temperature parameter
    )
    
    # DPO Trainer (updated for TRL v0.25+)
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # Train
    print("\nStarting DPO training...")
    dpo_trainer.train()
    
    # Save
    print(f"\nSaving DPO model to {output_dir}")
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return dpo_trainer

def main():
    # PHASE 2: DPO on iSarcasm dataset
    isarcasm_path = "data/isarcasm2022.csv"
    sft_adapter_path = "./qwen_sarc_sft"  # From Phase 1 SFT
    output_dir = "./qwen_sarcasm_dpo"
    
    print("="*70)
    print("PHASE 2: Direct Preference Optimization on iSarcasm Dataset")
    print("="*70)
    print("Strategy: Refine SARC-trained model with high-quality iSarcasm preferences")
    print(f"Loading SFT checkpoint from: {sft_adapter_path}")
    print("="*70)
    
    # Train DPO starting from SARC SFT model
    train_dpo(
        isarcasm_path, 
        output_dir=output_dir, 
        adapter_path=sft_adapter_path
    )
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Phase 1 (SFT): {sft_adapter_path}")
    print(f"Phase 2 (DPO): {output_dir}")
    print("\nWorkflow Summary:")
    print("  1. SFT on SARC → Learn general sarcasm patterns")
    print("  2. DPO on iSarcasm → Refine with expert preferences")
    print("="*70)

if __name__ == "__main__":
    main()
