#!/usr/bin/env python
# coding: utf-8
# %%
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from utils.constants import TARGET_DICT

# %%
df = pd.read_csv("../data/raw/teknofest_train_final.csv",
                 sep="|")

# %%
# Length filtering
df['text_len'] = df.text.str.len()
df = df[df.text_len >= 3]

# %%
# Semantic contradiction filtering
df = df[~((df.target == 'OTHER') & (df.is_offensive == 1))]
df = df[~((df.target != 'OTHER') & (df.is_offensive == 0))]
df = df.reset_index(drop=True)

# %%
# Uncased conversion
df["text"] = df["text"].str.lower()

# %%
# Label Encoding
df['target_label'] = df['target'].map(TARGET_DICT)

# %%
# Creating public & private folds
def assign_split_ids(input_df: pd.DataFrame,
                     fold_name: str,
                     fold_count: int,
                     seed: int):
    skf = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=seed)
    splits = list(skf.split(input_df, input_df["target"]))
    input_df[fold_name] = 0
    for split_id, split in enumerate(splits):
        input_df.loc[split[1], fold_name] = split_id


assign_split_ids(input_df=df, fold_name="public_fold", fold_count=5, seed=1337)
assign_split_ids(input_df=df, fold_name="private_fold", fold_count=5, seed=42)

# %%
# Export
df.to_csv("../data/processed/data.csv", index=False)
