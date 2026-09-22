import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np

def main():
    # 1. Paths
    eyepacs_path = 'final_processed/eyepacs/labels.csv'
    aptos_path = 'final_processed/aptos/labels.csv'
    idrid_train_path = 'final_processed/idrid/train/labels.csv'
    idrid_test_path = 'final_processed/idrid/test/labels.csv'
    
    output_dir = 'data/splits'
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Load Datasets
    print("Loading raw labels...")
    df_aptos = pd.read_csv(aptos_path)
    df_aptos['dataset_name'] = 'aptos'
    
    df_eyepacs = pd.read_csv(eyepacs_path)
    # Standardize column name
    if 'image_file' in df_eyepacs.columns:
        df_eyepacs.rename(columns={'image_file': 'id_code'}, inplace=True)
    df_eyepacs['dataset_name'] = 'eyepacs'
    
    df_idrid_tr = pd.read_csv(idrid_train_path)
    df_idrid_te = pd.read_csv(idrid_test_path)
    df_idrid = pd.concat([df_idrid_tr, df_idrid_te], ignore_index=True)
    df_idrid['dataset_name'] = 'idrid'
    
    # We ignore Messidor-2 because it is missing DR labels in this version of the processed dataset.
    # IDRiD alone provides exactly 516 images for our Lockbox Test Set.
    
    # 3. Create the Lockbox Test Set (IDRiD)
    print(f"Creating Lockbox Test Set with {len(df_idrid)} IDRiD images...")
    df_idrid.to_csv(os.path.join(output_dir, 'lockbox_test.csv'), index=False)
    
    # 4. Create the Working Training Dataset (APTOS + EyePACS)
    df_working = pd.concat([df_aptos, df_eyepacs], ignore_index=True)
    print(f"Initial Working Dataset size: {len(df_working)}")
    
    # 5. Aggressive Downsampling of EyePACS Grade 0
    # EyePACS Grade 0 severely dominates the dataset.
    np.random.seed(42)
    eyepacs_0_mask = (df_working['dataset_name'] == 'eyepacs') & (df_working['label'] == 0)
    
    # Get indices of EyePACS Grade 0
    eyepacs_0_indices = df_working[eyepacs_0_mask].index.tolist()
    print(f"Found {len(eyepacs_0_indices)} Grade 0 images in EyePACS.")
    
    # We want to drop ~80-90% of them to balance it. Let's keep 4000 of them.
    keep_count = min(4000, len(eyepacs_0_indices))
    keep_indices = np.random.choice(eyepacs_0_indices, size=keep_count, replace=False)
    
    # Create a mask for indices to drop
    drop_indices = set(eyepacs_0_indices) - set(keep_indices)
    df_working = df_working.drop(index=list(drop_indices)).reset_index(drop=True)
    
    print(f"Working Dataset size after Bias Correction: {len(df_working)}")
    
    # 6. Stratified 5-Fold Splitting
    print("Generating 5-Fold Stratification...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df_working['fold'] = -1
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_working, df_working['label'])):
        df_working.loc[val_idx, 'fold'] = fold
        
    # 7. Save and Verify
    out_path = os.path.join(output_dir, '5fold_splits.csv')
    df_working.to_csv(out_path, index=False)
    
    print(f"\nSaved cross-validation blueprint to: {out_path}")
    print("\n--- Final Fold Distribution by Grade ---")
    pivot = df_working.groupby(['fold', 'label']).size().unstack(fill_value=0)
    print(pivot)

if __name__ == "__main__":
    main()
