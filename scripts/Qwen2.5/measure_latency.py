"""
Measure inference latency for all model stages:
- Base Qwen2.5-0.5B (zero-shot)
- SFT model
- DPO model  
- GPT-4 (API call)

Reports mean, std, min, max latency over multiple runs.
"""

import torch
import time
import statistics
import json
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv
from openai import OpenAI
import argparse

# Load environment variables
load_dotenv()

# Test texts for latency measurement
TEST_TEXTS = [
    "Oh great, another Monday morning. Just what I needed.",
    "I love waiting in line for hours. It's my favorite activity.",
    "The weather is beautiful today, perfect for a walk in the park.",
    "Sure, because working overtime without pay is everyone's dream.",
    "This is a factual statement about climate change and its effects.",
]


def load_qwen_model(model_path=None, is_adapter=False, base_model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """Load Qwen model (base or with adapter)."""
    print(f"Loading model: {model_path or base_model_name}")
    
    # Determine device and dtype
    if torch.cuda.is_available():
        dtype = torch.float16
        device = "cuda"
    elif torch.backends.mps.is_available():
        dtype = torch.float32
        device = "mps"
    else:
        dtype = torch.float32
        device = "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    if is_adapter and model_path:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map="auto"
        )
    
    return model, tokenizer, device


def measure_qwen_latency(model, tokenizer, text, device, num_runs=10, warmup_runs=3):
    """Measure latency for a single Qwen model inference."""
    
    prompt = f"""Is the following text sarcastic? Answer with only 'Yes' or 'No'.

Text: {text}

Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Warmup runs
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
    
    # Synchronize before timing (for GPU/MPS)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Timed runs
    latencies = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Synchronize after generation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    return latencies


def measure_gpt4_latency(text, model="gpt-4", num_runs=5, warmup_runs=1):
    """Measure latency for GPT-4 API call."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⚠️  OPENAI_API_KEY not found in environment")
        return []
    
    client = OpenAI()
    
    # Warmup
    for _ in range(warmup_runs):
        try:
            _ = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You MUST respond with EXACTLY one word: either 'Yes' or 'No'. No other text."},
                    {"role": "user", "content": f"Is this text sarcastic? Text: {text}"}
                ],
                temperature=0,
                max_tokens=5
            )
        except Exception as e:
            print(f"GPT-4 warmup error: {e}")
    
    # Timed runs
    latencies = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        
        try:
            _ = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You MUST respond with EXACTLY one word: either 'Yes' or 'No'. No other text."},
                    {"role": "user", "content": f"Is this text sarcastic? Text: {text}"}
                ],
                temperature=0,
                max_tokens=5
            )
        except Exception as e:
            print(f"GPT-4 error: {e}")
            continue
        
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    return latencies


def compute_stats(latencies):
    """Compute latency statistics."""
    if not latencies:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0, "n_runs": 0}
    
    return {
        "mean": statistics.mean(latencies),
        "std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "min": min(latencies),
        "max": max(latencies),
        "median": statistics.median(latencies),
        "n_runs": len(latencies)
    }


def main():
    parser = argparse.ArgumentParser(description="Measure model latency")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of timed runs per text")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--skip_gpt4", action="store_true", help="Skip GPT-4 measurement")
    parser.add_argument("--gpt4_model", type=str, default="gpt-4", help="GPT model to use")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()
    
    print("="*70)
    print("MODEL LATENCY MEASUREMENT")
    print("="*70)
    print(f"Test texts: {len(TEST_TEXTS)}")
    print(f"Runs per text: {args.num_runs}")
    print(f"Warmup runs: {args.warmup}")
    print("="*70)
    
    results = {}
    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Fix paths if running from scripts/Qwen2.5 directory
    sft_path = "models/sft"
    dpo_path = "models/qwen_dpo_mistakes"
    
    if not os.path.exists(sft_path):
        sft_path = "../../models/sft"
        dpo_path = "../../models/qwen_dpo_mistakes"
    
    # ========== Base Qwen ==========
    print(f"\n{'='*70}")
    print("1. BASE QWEN2.5-0.5B (Zero-shot)")
    print("="*70)
    
    model, tokenizer, device = load_qwen_model(base_model_name, is_adapter=False)
    print(f"Device: {device}")
    
    base_latencies = []
    for i, text in enumerate(TEST_TEXTS):
        print(f"  Text {i+1}/{len(TEST_TEXTS)}...", end=" ")
        lats = measure_qwen_latency(model, tokenizer, text, device, args.num_runs, args.warmup)
        base_latencies.extend(lats)
        print(f"mean: {statistics.mean(lats):.1f}ms")
    
    results["base_qwen"] = compute_stats(base_latencies)
    print(f"\n  Overall: {results['base_qwen']['mean']:.1f}ms ± {results['base_qwen']['std']:.1f}ms")
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ========== SFT Model ==========
    print(f"\n{'='*70}")
    print("2. SFT MODEL (Qwen2.5-0.5B + LoRA)")
    print("="*70)
    
    if os.path.exists(sft_path):
        model, tokenizer, device = load_qwen_model(sft_path, is_adapter=True, base_model_name=base_model_name)
        
        sft_latencies = []
        for i, text in enumerate(TEST_TEXTS):
            print(f"  Text {i+1}/{len(TEST_TEXTS)}...", end=" ")
            lats = measure_qwen_latency(model, tokenizer, text, device, args.num_runs, args.warmup)
            sft_latencies.extend(lats)
            print(f"mean: {statistics.mean(lats):.1f}ms")
        
        results["sft"] = compute_stats(sft_latencies)
        print(f"\n  Overall: {results['sft']['mean']:.1f}ms ± {results['sft']['std']:.1f}ms")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        print(f"  ⚠️  SFT model not found at {sft_path}")
        results["sft"] = None
    
    # ========== DPO Model ==========
    print(f"\n{'='*70}")
    print("3. DPO MODEL (Qwen2.5-0.5B + SFT + DPO)")
    print("="*70)
    
    if os.path.exists(dpo_path):
        model, tokenizer, device = load_qwen_model(dpo_path, is_adapter=True, base_model_name=base_model_name)
        
        dpo_latencies = []
        for i, text in enumerate(TEST_TEXTS):
            print(f"  Text {i+1}/{len(TEST_TEXTS)}...", end=" ")
            lats = measure_qwen_latency(model, tokenizer, text, device, args.num_runs, args.warmup)
            dpo_latencies.extend(lats)
            print(f"mean: {statistics.mean(lats):.1f}ms")
        
        results["dpo"] = compute_stats(dpo_latencies)
        print(f"\n  Overall: {results['dpo']['mean']:.1f}ms ± {results['dpo']['std']:.1f}ms")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        print(f"  ⚠️  DPO model not found at {dpo_path}")
        results["dpo"] = None
    
    # ========== GPT-4 ==========
    if not args.skip_gpt4:
        print(f"\n{'='*70}")
        print(f"4. GPT-4 ({args.gpt4_model}) - API Call")
        print("="*70)
        
        gpt4_latencies = []
        for i, text in enumerate(TEST_TEXTS):
            print(f"  Text {i+1}/{len(TEST_TEXTS)}...", end=" ")
            lats = measure_gpt4_latency(text, args.gpt4_model, num_runs=args.num_runs, warmup_runs=1)
            if lats:
                gpt4_latencies.extend(lats)
                print(f"mean: {statistics.mean(lats):.1f}ms")
            else:
                print("skipped")
        
        if gpt4_latencies:
            results["gpt4"] = compute_stats(gpt4_latencies)
            print(f"\n  Overall: {results['gpt4']['mean']:.1f}ms ± {results['gpt4']['std']:.1f}ms")
        else:
            results["gpt4"] = None
    else:
        print(f"\n{'='*70}")
        print("4. GPT-4 - SKIPPED")
        print("="*70)
        results["gpt4"] = None
    
    # ========== Summary ==========
    print(f"\n{'='*70}")
    print("LATENCY SUMMARY")
    print("="*70)
    print(f"{'Model':<30} {'Mean (ms)':<15} {'Std (ms)':<15} {'Median (ms)':<15}")
    print("-"*70)
    
    model_names = {
        "base_qwen": "Qwen2.5-0.5B (Base)",
        "sft": "Qwen2.5-0.5B + SFT",
        "dpo": "Qwen2.5-0.5B + SFT + DPO",
        "gpt4": f"GPT-4 ({args.gpt4_model})"
    }
    
    for key, name in model_names.items():
        if results.get(key):
            stats = results[key]
            print(f"{name:<30} {stats['mean']:<15.1f} {stats['std']:<15.1f} {stats['median']:<15.1f}")
        else:
            print(f"{name:<30} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    
    print("="*70)
    
    # Speedup comparison
    if results.get("gpt4") and results.get("dpo"):
        speedup = results["gpt4"]["mean"] / results["dpo"]["mean"]
        print(f"\n🚀 Qwen2.5+DPO is {speedup:.1f}x faster than GPT-4")
    
    if results.get("base_qwen") and results.get("sft"):
        overhead = ((results["sft"]["mean"] - results["base_qwen"]["mean"]) / results["base_qwen"]["mean"]) * 100
        print(f"📊 LoRA adapter overhead: {overhead:+.1f}%")
    
    # Save results
    output_path = args.output
    if output_path is None:
        output_path = "results/latency_results.json"
        if not os.path.exists("results"):
            output_path = "../../results/latency_results.json"
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_runs": args.num_runs,
            "warmup_runs": args.warmup,
            "num_texts": len(TEST_TEXTS),
            "gpt4_model": args.gpt4_model
        },
        "results": results
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
