#!/usr/bin/env python
# coding: utf-8
# %%
import pandas as pd
from utils.constants import TARGET_DICT


# %%
df = pd.read_csv("../data/raw/teknofest_train_final.csv",
                 sep="|")


# %%
df['text_len'] = df.text.str.len()
df = df[df.text_len >= 3]


# %%
df = df[~((df.target == 'OTHER') & (df.is_offensive == 1))]
df = df[~((df.target != 'OTHER') & (df.is_offensive == 0))]
df = df.reset_index(drop=True)


# %%
df["text"] = df["text"].str.lower()

# %%
df['target_label'] = df['target'].map(TARGET_DICT)


# %%
df.to_csv("../data/processed/data.csv", index=False)

