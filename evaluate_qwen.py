"""
Evaluate Qwen3-0.6B model on iSarcasm dataset for sarcasm detection.
This script uses zero-shot prompting to assess the model's baseline performance.
"""

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
from datetime import datetime

def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """Load the Qwen model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return model, tokenizer

def create_prompt(text):
    """Create a prompt for sarcasm detection."""
    prompt = f"""Is the following text sarcastic? Answer with only 'Yes' or 'No'.

Text: {text}

Answer:"""
    return prompt

def get_prediction(model, tokenizer, text, max_new_tokens=100, use_thinking=False):
    """Get model prediction for a single text."""
    # Use chat template for Qwen3 (works for both Qwen2.5 and Qwen3)
    messages = [
        {"role": "user", "content": f"Is the following text sarcastic? Answer with only 'Yes' or 'No'.\n\nText: {text}\n\nAnswer:"}
    ]
    
    # Apply chat template with thinking mode control
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=use_thinking  # For Qwen3
        )
    except TypeError:
        # Fallback for Qwen2.5 which doesn't have enable_thinking
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response with proper sampling parameters
    with torch.no_grad():
        if use_thinking:
            # Thinking mode settings for Qwen3
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            # Non-thinking mode or Qwen2.5 settings
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    response = response.strip().lower()
    
    # Parse response (handle thinking tags if present)
    # Remove thinking content if it exists
    if '</think>' in response:
        response = response.split('</think>')[-1].strip()
    
    # Parse response
    if 'yes' in response:
        return 1
    elif 'no' in response:
        return 0
    else:
        return -1  # unclear response

def evaluate_dataset(model, tokenizer, df, sample_size=None, use_thinking=False):
    """Evaluate model on the dataset."""
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    predictions = []
    correct = 0
    unclear = 0
    
    print(f"\nEvaluating on {len(df)} samples (thinking_mode={use_thinking})...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row['tweet']
        true_label = row['sarcastic']
        
        pred = get_prediction(model, tokenizer, text, use_thinking=use_thinking)
        predictions.append(pred)
        
        if pred == -1:
            unclear += 1
        elif pred == true_label:
            correct += 1
    
    # Calculate metrics
    valid_predictions = sum(1 for p in predictions if p != -1)
    accuracy = correct / valid_predictions if valid_predictions > 0 else 0
    
    results = {
        'total_samples': len(df),
        'valid_predictions': valid_predictions,
        'unclear_responses': unclear,
        'correct_predictions': correct,
        'accuracy': accuracy,
        'thinking_mode': use_thinking,
        'timestamp': datetime.now().isoformat()
    }
    
    return results, predictions

def main():
    # Load dataset
    print("Loading iSarcasm dataset...")
    df = pd.read_csv('data/isarcasm2022.csv', index_col=0)
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Sarcastic samples: {df['sarcastic'].sum()}")
    print(f"Non-sarcastic samples: {len(df) - df['sarcastic'].sum()}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Detect if this is Qwen3 (has thinking capability)
    is_qwen3 = "Qwen3" in model.config._name_or_path or "qwen3" in model.config.model_type
    print(f"\nDetected model type: {'Qwen3 (with thinking)' if is_qwen3 else 'Qwen2.5 or earlier'}")
    
    # For Qwen3, you can test both modes:
    if is_qwen3:
        print("\n" + "="*50)
        print("Testing Qwen3 in NON-THINKING mode first (faster)")
        print("="*50)
        results_no_think, predictions_no_think = evaluate_dataset(
            model, tokenizer, df, sample_size=100, use_thinking=False
        )
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS (Non-Thinking Mode)")
        print("="*50)
        print(f"Total samples: {results_no_think['total_samples']}")
        print(f"Valid predictions: {results_no_think['valid_predictions']}")
        print(f"Unclear responses: {results_no_think['unclear_responses']}")
        print(f"Correct predictions: {results_no_think['correct_predictions']}")
        print(f"Accuracy: {results_no_think['accuracy']:.2%}")
        
        # Optionally test thinking mode (slower but may be more accurate)
        # Uncomment below to test:
        # print("\n" + "="*50)
        # print("Testing Qwen3 in THINKING mode (slower, uses reasoning)")
        # print("="*50)
        # results_think, predictions_think = evaluate_dataset(
        #     model, tokenizer, df, sample_size=100, use_thinking=True
        # )
        # print(f"Thinking mode accuracy: {results_think['accuracy']:.2%}")
        
        results = results_no_think
        predictions = predictions_no_think
    else:
        # Qwen2.5 - single mode
        results, predictions = evaluate_dataset(
            model, tokenizer, df, sample_size=100, use_thinking=False
        )
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Total samples: {results['total_samples']}")
        print(f"Valid predictions: {results['valid_predictions']}")
        print(f"Unclear responses: {results['unclear_responses']}")
        print(f"Correct predictions: {results['correct_predictions']}")
        print(f"Accuracy: {results['accuracy']:.2%}")
        print("="*50)
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to evaluation_results.json")

if __name__ == "__main__":
    main()
