"""
Create train/test splits for iSarcasm dataset.
Train split (80%) will be used for DPO training.
Test split (20%) will be held out for evaluation only.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os

def create_isarcasm_splits():
    """Create stratified train/test splits for iSarcasm."""
    print("Creating iSarcasm train/test splits...")
    
    # Load full dataset
    df = pd.read_csv('data/isarcasm2022.csv', index_col=0)
    print(f"\nTotal samples: {len(df)}")
    print(f"Sarcastic: {df['sarcastic'].sum()} ({df['sarcastic'].mean():.1%})")
    print(f"Non-sarcastic: {len(df) - df['sarcastic'].sum()} ({1-df['sarcastic'].mean():.1%})")
    
    # Create stratified split
    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['sarcastic']
    )
    
    # Save splits
    df_train.to_csv('data/splits/isarcasm_train.csv')
    df_test.to_csv('data/splits/isarcasm_test.csv')
    
    print(f"\n✓ Train split: {len(df_train)} samples")
    print(f"  Sarcastic: {df_train['sarcastic'].sum()} ({df_train['sarcastic'].mean():.1%})")
    print(f"  Non-sarcastic: {len(df_train) - df_train['sarcastic'].sum()}")
    
    print(f"\n✓ Test split: {len(df_test)} samples")
    print(f"  Sarcastic: {df_test['sarcastic'].sum()} ({df_test['sarcastic'].mean():.1%})")
    print(f"  Non-sarcastic: {len(df_test) - df_test['sarcastic'].sum()}")
    
    print(f"\n✓ Saved to data/splits/")
    print("  - isarcasm_train.csv (for DPO training)")
    print("  - isarcasm_test.csv (for evaluation only)")
    return

def split_sarc_data():
    df_sarc = pd.read_csv('data/SARC/train-balanced-sarcasm.csv')
    df_sarc.columns
    df_sarc_non_sarc = df_sarc[df_sarc['label'] == 0]
    df_sarc_sarc = df_sarc[df_sarc['label'] == 1]
    return df_sarc_non_sarc, df_sarc_sarc

def sarc_sft_dpo_split():
    os.makedirs('data/SARC/splits', exist_ok=True)
    df_sarc_non_sarc, df_sarc_sarc = split_sarc_data()
    # 80% non-sarc for sft, 10% non-sarc for dpo, 10% non-sarc for test
    non_sarc_sft = df_sarc_non_sarc.sample(frac=0.8, random_state=42)
    non_sarc_remaining = df_sarc_non_sarc.drop(non_sarc_sft.index)
    non_sarc_dpo = non_sarc_remaining.sample(frac=0.5, random_state=42)
    non_sarc_test = non_sarc_remaining.drop(non_sarc_dpo.index)
    
    # 80% sarc for sft, 10% sarc for dpo, 10% sarc for test
    sarc_sft = df_sarc_sarc.sample(frac=0.8, random_state=42)
    sarc_remaining = df_sarc_sarc.drop(sarc_sft.index)
    sarc_dpo = sarc_remaining.sample(frac=0.5, random_state=42)
    sarc_test = sarc_remaining.drop(sarc_dpo.index)
    
    # Combine non-sarc and sarc splits and shuffle
    sft_data = pd.concat([non_sarc_sft, sarc_sft]).sample(frac=1, random_state=42).reset_index(drop=True)
    dpo_data = pd.concat([non_sarc_dpo, sarc_dpo]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_data = pd.concat([non_sarc_test, sarc_test]).sample(frac=1, random_state=42).reset_index(drop=True)
    sft_data.to_csv('data/SARC/splits/sft_data.csv', index=False)
    dpo_data.to_csv('data/SARC/splits/dpo_data.csv', index=False)
    test_data.to_csv('data/SARC/splits/test_data.csv', index=False)
    print(f"Saved to data/SARC/splits - SFT data size: {len(sft_data)}")
    return 

if __name__ == "__main__":
    sarc_sft_dpo_split()
    # create_isarcasm_splits()
