# MobileLLM Scripts

This folder contains training scripts for **facebook/MobileLLM-R1.5-360M** (360M parameters) - a compact model optimized for mobile deployment.

## Scripts

### 1. `finetune_mobilellm.py` - Supervised Fine-Tuning (SFT)
Fine-tune MobileLLM on the GEN sarcasm dataset using LoRA.

```bash
# From project root
python scripts/MobileLLM/finetune_mobilellm.py
```

Output: `models/mobilellm_sft/`

### 2. `mine_sft_preferences.py` - Mine Preference Pairs
Run SFT model on training data to create DPO preference pairs from mistakes.

```bash
python scripts/MobileLLM/mine_sft_preferences.py --filter confident_mistakes
```

Output: `data/mobilellm_preferences.json`

### 3. `dpo_train.py` - Direct Preference Optimization
Train DPO on preference pairs to improve the model.

```bash
# Using mined preferences (recommended)
python scripts/MobileLLM/dpo_train.py --preference_data data/mobilellm_preferences.json

# Or using iSarcasm synthetic pairs
python scripts/MobileLLM/dpo_train.py
```

Output: `models/mobilellm_dpo/`

### 4. `dpo_eval.py` - Evaluate DPO Model
Quick evaluation script to verify DPO hasn't regressed.

```bash
python scripts/MobileLLM/dpo_eval.py
```

## Training Pipeline

```
1. Create data splits:
   python scripts/split_gen_dataset.py

2. SFT Training:
   python scripts/MobileLLM/finetune_mobilellm.py

3. Mine preferences:
   python scripts/MobileLLM/mine_sft_preferences.py

4. DPO Training:
   python scripts/MobileLLM/dpo_train.py --preference_data data/mobilellm_preferences.json

5. Evaluate:
   python scripts/MobileLLM/dpo_eval.py
```

## Model Details

- **Base Model**: facebook/MobileLLM-R1.5-360M
- **Parameters**: 360M
- **LoRA Config**: r=16, alpha=32, dropout=0.05
- **Target Modules**: q_proj, k_proj, v_proj, o_proj
