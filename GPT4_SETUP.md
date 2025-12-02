# GPT-4 Evaluation Setup Guide

## Prerequisites

1. **OpenAI API Key**: You need an OpenAI API key with access to GPT-4

## Setup Steps

### 1. Get Your OpenAI API Key

- Go to https://platform.openai.com/api-keys
- Create a new API key or use an existing one
- Copy the key (starts with `sk-...`)

### 2. Set Environment Variable

Create a `.env` file in the project root (already done if you cloned this repo):
```bash
OPENAI_API_KEY=sk-your-api-key-here
```

The scripts will automatically load from `.env`. Alternatively, set it in your terminal:
```bash
export OPENAI_API_KEY='sk-your-api-key-here'
```

### 3. Install OpenAI Package (if not already installed)

```bash
pip install openai
```

## Usage

### Run Full Evaluation (including GPT-4)

```bash
python scripts/evaluate_all_stages.py
```

This will evaluate:
- GPT-4 (zero-shot)
- Base Qwen model (zero-shot)
- SFT model
- DPO model

### Skip GPT-4 (to save API costs)

```bash
python scripts/evaluate_all_stages.py --skip-gpt4
```

### Evaluate Only GPT-4

```bash
python scripts/evaluate_all_stages.py --gpt4-only
```

### Use a Different GPT Model

```bash
# Use GPT-4 Turbo (cheaper and faster)
python scripts/evaluate_all_stages.py --gpt4-model gpt-4-turbo

# Use GPT-3.5 Turbo (much cheaper, but less capable)
python scripts/evaluate_all_stages.py --gpt4-model gpt-3.5-turbo
```

## API Cost Estimates

Based on OpenAI pricing (as of Dec 2024):

- **GPT-4**: ~$0.03 per 1K input tokens, ~$0.06 per 1K output tokens
- **GPT-4 Turbo**: ~$0.01 per 1K input tokens, ~$0.03 per 1K output tokens
- **GPT-3.5 Turbo**: ~$0.001 per 1K input tokens, ~$0.002 per 1K output tokens

For 500 samples with ~50 tokens per prompt:
- **GPT-4**: ~$1.50-$3.00
- **GPT-4 Turbo**: ~$0.50-$1.00
- **GPT-3.5 Turbo**: ~$0.05-$0.10

## Troubleshooting

### "Could not evaluate GPT-4" Error

1. Check that your API key is set:
   ```bash
   echo $OPENAI_API_KEY
   ```

2. Verify the API key is valid at https://platform.openai.com/api-keys

3. Make sure you have API credits available

### Rate Limiting

If you hit rate limits, the script includes a 0.1s delay between requests. For very strict rate limits, you can modify the `time.sleep(0.1)` value in `evaluate_gpt4()` function to a higher value.

### Model Access

Make sure your API key has access to GPT-4. If not, you can use GPT-3.5-turbo:
```bash
python scripts/evaluate_all_stages.py --gpt4-model gpt-3.5-turbo
```
