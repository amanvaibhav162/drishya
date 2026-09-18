import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

data_root = '/workspace/drishya/data/train_dataset'
records = []

for ds in sorted(os.listdir(data_root)):
    ds_dir = os.path.join(data_root, ds)
    if not os.path.isdir(ds_dir):
        continue
    prep_dir = os.path.join(ds_dir, 'train_preprocessed')
    mask_dir = os.path.join(ds_dir, 'train_masks')
    if not os.path.exists(prep_dir):
        continue
    for grade_str in sorted(os.listdir(prep_dir)):
        grade_dir = os.path.join(prep_dir, grade_str)
        if not os.path.isdir(grade_dir):
            continue
        try:
            grade = int(grade_str)
        except ValueError:
            continue
        for img_name in sorted(os.listdir(grade_dir)):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            base_name = os.path.splitext(img_name)[0]
            img_path = os.path.join(grade_dir, img_name)
            # Check for _composite.png mask
            mask_path = os.path.join(mask_dir, grade_str, base_name + '_composite.png')
            has_mask = int(os.path.exists(mask_path))

            # Patient-level grouping
            if ds == 'eyepacs' and ('_left' in base_name or '_right' in base_name):
                patient_id = base_name.replace('_left', '').replace('_right', '')
            elif ds == 'messidor2' and len(base_name.split('_')) >= 2:
                patient_id = '_'.join(base_name.split('_')[:2])
            else:
                patient_id = f'{ds}_{base_name}'

            records.append({
                'image_path': img_path,
                'mask_path': mask_path if has_mask else '',
                'grade': grade,
                'dataset': ds,
                'patient_id': patient_id,
                'has_mask': has_mask
            })

df = pd.DataFrame(records)
print(f'Total matched images: {len(df):,}')
print(f'Total matched masks:  {df["has_mask"].sum():,}')

# Stratified Group 5-Fold
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
df['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df['grade'], df['patient_id'])):
    df.loc[val_idx, 'fold'] = fold

print('\nFold summary:')
for f in range(5):
    f_df = df[df['fold'] == f]
    g_dist = dict(f_df['grade'].value_counts().sort_index())
    print(f'  Fold {f}: {len(f_df):,d} images | Grades: {g_dist}')

os.makedirs('/workspace/drishya/data/splits', exist_ok=True)
out_csv = '/workspace/drishya/data/splits/5fold_splits.csv'
df.to_csv(out_csv, index=False)
print(f'\nSUCCESS: Saved 5-fold split to {out_csv}')
