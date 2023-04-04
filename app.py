# THIS SCRIPT IS PROVIDED BY THE ORGANIZATOR

import re
import pandas as pd
import os
import numpy as np
import gradio as gr
from src.utils.preprocess_utils import preprocess_text
from src.utils.constants import TARGET_DICT, TARGET_INV_DICT
from src.models.bert_model import BertModel


# CV Voting Model Load
# For model class import, model checkpoint looks for models sub-dir
# import sys
# sys.path.append("./src")

#

# Loading and transferring the model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # Competition Model
# model = torch.load("checkpoint/blend_model.bin",
#                    map_location=device)
# model.set_device(device)

# Competition Model
model = BertModel(model_path="l2reg/toxic-dbmdz-bert-base-turkish-128k-uncased",
                  auth_token=os.environ["model_token"])
model.load()

# Case-Unbiased Model
case_unbiased_model = BertModel(model_path="l2reg/toxic-dbmdz-bert-base-turkish-128k-uncased-casing-unbiased",
                                auth_token=os.environ["model_token"])
case_unbiased_model.load()

# Fully-Unbiased Model
fully_unbiased_model = BertModel(model_path="l2reg/toxic-dbmdz-bert-base-turkish-128k-uncased-fully-unbiased",
                                 auth_token=os.environ["model_token"])
fully_unbiased_model.load()


# Cased-Sentence ratio
def get_uppercase_sentence_ratio(input_df:pd.DataFrame):
    
    """
    Get uppercase ratio. 
    
    ---------
    param input_df: input dataframe
    return: Uppercase sentence ratio
    """
    
    def find_uppercase(text):
        pattern = '[A-Z]'
        rgx = re.compile(pattern)
        result = rgx.findall(''.join(sorted(text)))
        return result

    any_upper_letter = input_df["text"].apply(lambda x: find_uppercase(x))
    any_upper_letter = any_upper_letter.apply(lambda x: len(x) > 0)
    any_upper_letter = any_upper_letter.astype(int)

    return np.round(any_upper_letter.mean(), 3)


# Authorization routine
def auth(username,
         password):
    if username == "L2_Regulasyon" and password == os.environ["space_auth_pass"]:
        return True
    else:
        return False


def predict(df:pd.DataFrame):
    
    """
    Model inference for gradio app. 
    
    ---------
    param input_df: input dataframe
    return: Dataframe with wanted columns.   
    """
    
    df["is_offensive"] = 1
    df["target"] = "OTHER"

    # Case-Specific Competition Routine
    cased_ratio = get_uppercase_sentence_ratio(df)
    print(f"CR: {cased_ratio}")
    if (cased_ratio <= 0.35) and (cased_ratio >= 0.25):
        print(f"Using # routine...")
        df["proc_text"] = preprocess_text(df["text"],
                                          prevent_bias=0)
        pred_classes, _ = model.predict(df["proc_text"])
    else:
        print(f"Using lower routine...")
        df["proc_text"] = preprocess_text(df["text"],
                                          prevent_bias=1)
        pred_classes, _ = case_unbiased_model.predict(df["proc_text"])

    # Class ID > Text
    for pred_i, pred in enumerate(pred_classes):
        pred_classes[pred_i] = TARGET_INV_DICT[pred] if pred in [0, 1, 2, 3, 4] else pred

    df["target"] = pred_classes
    df.loc[df["target"] == "OTHER", "is_offensive"] = 0

    return df[["id", "text", "is_offensive", "target"]]


def get_file(file):
    output_file = "output_L2_Regulasyon.csv"

    # For windows users, replace path seperator
    file_name = file.name.replace("\\", "/")

    df = pd.read_csv(file_name, sep="|")
    print(f"Got {file_name}.\nIt consists of {len(df)} rows!")

    df = predict(df)
    df.to_csv(output_file, index=False, sep="|")
    return (output_file)


def demo_inference(selected_model,
                   input_text):
    input = pd.Series([input_text])

    if selected_model == "Yarışma Modeli":
        proc_input = preprocess_text(input,
                                     prevent_bias=0)
        pred_classes, pred_probas = model.predict(proc_input)

    elif selected_model == "Case-Unbiased Model":
        proc_input = preprocess_text(input,
                                     prevent_bias=1)
        pred_classes, pred_probas = case_unbiased_model.predict(proc_input)

    elif selected_model == "Fully-Unbiased Model (Ürün Modu)":
        proc_input = preprocess_text(input,
                                     prevent_bias=2)
        pred_classes, pred_probas = fully_unbiased_model.predict(proc_input)

    for pred_i, pred in enumerate(pred_classes):
        pred_classes[pred_i] = TARGET_INV_DICT[pred] if pred in [0, 1, 2, 3, 4] else pred

    return dict(zip(list(TARGET_DICT.keys()), pred_probas[0].tolist()))


model_selector = gr.Radio(["Yarışma Modeli", "Case-Unbiased Model", "Fully-Unbiased Model (Ürün Modu)"])


# Launch the interface with user password
competition_interface = gr.Interface(get_file, "file", gr.File())
demo_interface = gr.Interface(demo_inference, [model_selector, gr.Text()], gr.Label(num_top_classes=5))

if __name__ == "__main__":
    gr.TabbedInterface(
        [competition_interface, demo_interface], ["Yarışma Ekranı", "Demo Ekranı"]
    ).launch(server_name="0.0.0.0",
             share=False,
             auth=None if ("space_auth_pass" not in os.environ) else auth)
