# Sarcasm Detection with Qwen2.5-0.5B

This project uses a **two-phase training strategy** for sarcasm detection:
1. **Phase 1 (SFT)**: Supervised fine-tuning on large SARC dataset
2. **Phase 2 (DPO)**: Direct Preference Optimization on high-quality iSarcasm dataset

## Training Strategy

### Why Two Phases?

**Phase 1: SARC (Volume)**
- Large dataset (~1M samples) provides broad pattern learning
- Teaches model general sarcasm indicators
- Balanced sarcastic/non-sarcastic examples

**Phase 2: iSarcasm (Quality)**
- Smaller (4k samples) but expert-annotated
- Rich multi-dimensional labels (irony, satire, overstatement, etc.)
- DPO refines model's decision boundaries and confidence
- Teaches "what good sarcasm detection looks like"

This approach combines **quantity** (SARC) with **quality** (iSarcasm) for optimal performance.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Check your datasets:
```bash
python prepare_data.py
```

## Complete Workflow

### Step 0: Baseline Evaluation (Optional but Recommended)
Evaluate the pre-trained model before fine-tuning:

```bash
python evaluate_qwen.py
```

This provides a baseline to compare against after training.

**Or run comprehensive evaluation across all stages:**
```bash
python evaluate_all_stages.py
```
This will evaluate:
1. Base model (zero-shot)
2. After SFT (if checkpoint exists)
3. After DPO (if checkpoint exists)

And produce a comparative analysis showing improvements at each stage.

### Step 1: Phase 1 - SFT on SARC
Train on the large SARC dataset:

```bash
python finetune_qwen.py
```

This will:
- Load Qwen2.5-0.5B-Instruct base model
- Apply LoRA for efficient fine-tuning
- Train on SARC dataset (50k samples by default, adjust in script)
- Save checkpoint to `./qwen_sarc_sft/`
- Takes ~1-3 hours depending on sample size and hardware

**Training parameters:**
- Dataset: `data/SARC/train-balanced-sarcasm.csv`
- Sample size: 50,000 (adjustable in script)
- LoRA rank: 16
- Epochs: 3
- Batch size: 4 (effective: 16 with gradient accumulation)

### Step 2: Phase 2 - DPO on iSarcasm
Refine with preference optimization:

```bash
python dpo_train.py
```

This will:
- Load the Phase 1 SFT checkpoint from `./qwen_sarc_sft/`
- Create preference pairs from iSarcasm annotations
- Use DPO to align model preferences
- Save final model to `./qwen_sarcasm_dpo/`
- Takes ~30-60 minutes

**DPO features:**
- Uses iSarcasm's rich annotations (irony, satire, etc.)
- Chosen: Correct label with contextual reasoning
- Rejected: Common failure modes
- Beta parameter: 0.1 (DPO temperature)

### Step 3: Evaluate Final Model
Test the DPO-trained model:

```bash
# Modify evaluate_qwen.py to load your trained model:
# model_name = "./qwen_sarcasm_dpo"
python evaluate_qwen.py
```

**Or run comprehensive comparison:**
```bash
python evaluate_all_stages.py
```

This will:
- Evaluate all three stages (Base, SFT, DPO)
- Calculate accuracy, precision, recall, F1 for each
- Show improvement at each stage
- Save detailed results to JSON files
- Cache results to avoid re-running expensive evaluations

**Results saved to:**
- `comparative_results.json` - All results in one file
- `results_base_model.json` - Base model results (cached)
- `results_sft_model.json` - SFT model results
- `results_dpo_model.json` - DPO model results

## Dataset Information

### SARC Dataset
- **Location**: `data/SARC/train-balanced-sarcasm.csv`
- **Size**: ~255MB, ~1M samples
- **Format**: Reddit comments with sarcasm labels
- **Use**: Phase 1 (SFT) - Pattern learning

### iSarcasm Dataset
- **Location**: `data/isarcasm2022.csv`  
- **Size**: 4,014 samples
- **Format**: Tweets with rich annotations
- **Labels**:
  - `sarcastic`: Binary sarcasm label
  - `irony`, `satire`, `overstatement`, `understatement`, `rhetorical_question`: Sarcasm types
- **Use**: Phase 2 (DPO) - Preference alignment

## Model Details

- **Base Model**: Qwen/Qwen2.5-0.5B-Instruct
- **Method**: LoRA fine-tuning → DPO alignment
- **Parameters**: 0.49B total (0.36B non-embedding)
- **Task**: Binary sarcasm classification

## Training Configuration

### LoRA Settings (Phase 1)
```python
r=16                    # Rank
lora_alpha=32          # Scaling factor
target_modules=[       # Attention layers
    "q_proj", "k_proj", 
    "v_proj", "o_proj"
]
lora_dropout=0.05
```

### DPO Settings (Phase 2)
```python
beta=0.1               # DPO temperature
learning_rate=5e-5     # Lower LR for fine refinement
epochs=3
```

## Expected Results

- **Zero-shot baseline**: ~50-65% accuracy
- **After Phase 1 (SFT)**: ~75-85% accuracy
- **After Phase 2 (DPO)**: ~80-90% accuracy + better calibration

DPO particularly improves:
- Confidence calibration
- Edge case handling
- Type-specific accuracy (irony vs satire)

## Utilities

### Check Dataset Status
```bash
python prepare_data.py
```

Shows:
- Available datasets
- Sample counts and class balance
- Sarcasm type distributions
- Training strategy overview

### Create Train/Val Splits
The script automatically handles splitting, but you can customize:
```python
from prepare_data import create_train_val_split
create_train_val_split("data/isarcasm2022.csv", train_ratio=0.8)
```

## File Structure

```
.
├── data/
│   ├── SARC/
│   │   └── train-balanced-sarcasm.csv  # Phase 1 training data
│   └── isarcasm2022.csv                 # Phase 2 training data
├── evaluate_qwen.py                     # Evaluation script
├── finetune_qwen.py                     # Phase 1: SFT on SARC
├── dpo_train.py                         # Phase 2: DPO on iSarcasm
├── prepare_data.py                      # Data utilities
├── requirements.txt                     # Dependencies
└── README.md                            # This file

# Generated during training:
├── qwen_sarc_sft/                       # Phase 1 checkpoint
├── qwen_sarcasm_dpo/                    # Final model
└── evaluation_results.json              # Evaluation metrics
```

## Tips & Troubleshooting

1. **Memory Issues**: Reduce `sample_size` in finetune_qwen.py or decrease batch size
2. **Slow Training**: Start with 10k-50k samples from SARC, increase if needed
3. **PyArrow Errors**: Run `pip install --force-reinstall pyarrow`
4. **Model Loading**: Ensure you're using `transformers>=4.37.0`

## Customization

### Adjust Training Sample Size
In `finetune_qwen.py`:
```python
sample_size=50000  # Increase to 100k, 500k, or None (full dataset)
```

### Change LoRA Rank (more parameters)
```python
r=32  # Higher = more capacity, more memory
```

### Adjust DPO Beta
```python
beta=0.2  # Higher = stronger preference enforcement
```

## Citation

If you use this code or approach, please cite:
- Qwen2.5: https://qwenlm.github.io/blog/qwen2.5/
- iSarcasm: https://github.com/iabufarha/iSarcasmEval
- SARC: https://nlp.cs.princeton.edu/SARC/
