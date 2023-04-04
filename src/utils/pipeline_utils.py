import time
import pandas as pd
import os
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
from .preprocess_utils import preprocess_text
from .data_utils import read_model_zoo, update_zoo, write_model_zoo
from .constants import MODEL_CV_RESULT_PATH, TARGET_DICT, TARGET_INV_DICT


def add_external_positive_data(x_series, y_series):
    external_path = "../data/external/tweetset.csv"
    if not os.path.exists(external_path):
        os.system(
            f"wget -O {external_path} https://github.com/ezgisubasi/turkish-tweets-sentiment-analysis/raw/main/data/tweetset.csv")
    ext_df = pd.read_csv(external_path,
                         encoding="windows-1254")
    ext_df = ext_df[ext_df["Tip"] == "Pozitif"].reset_index(drop=True)
    ext_df["target"] = 0
    x_train_ext = ext_df["Paylaşım"].str.lower().rename("text")
    y_train_ext = ext_df["target"]
    x_series = pd.concat([x_series, x_train_ext], ignore_index=True)
    y_series = pd.concat([y_series, y_train_ext], ignore_index=True)
    return x_series, y_series

def run_cv(model_obj,
           model_params: dict,
           input_df: pd.DataFrame,
           fold_col: str,
           x_col: str,
           y_col: str,
           experiment_name: str = "NONAME",
           add_to_zoo: bool = False,
           is_nn: bool = False,
           prevent_bias: int = 0):
    
    """
    Run the selected model, evaluate the results and save the experiment to the model zoo if selected.
    
    ---------
    :param model_obj: Model class
    :param model_params: Model paramaters, may vary depending on model architecture.
    :param input_df: Competition train dataframe with CV folds.
    :param fold_col: Fold column name. There are two types of fold schema: Public(public_fold) and Private(private_fold).
    :param x_col: Text feature column.
    :param y_col: Target column.
    :param experiment_name: If the model is to be saved in the model zoo, this parameter must be a unique string: 'dbmdz/bert-base-turkish-128k-uncased | epochs: 3 | batch_size: 32 | max_len: 128', e.g. 
    :param add_to_zoo: Add to the model zoo for model tracking.
    :param is_nn: If the model comes from NN based approach, set true.
    :return: Prints classification results, save OOF results, add to the model zoo with experiment name if add_to_zoo is set to True.
    """
    print()
    print("*"*30)
    print("Started CV Training")
    print("*"*30)
    print(f"Experiment: '{experiment_name}'")
    print(f"Fold: '{fold_col}'")
    print(f"Update Zoo: '{add_to_zoo}'")
    print("*"*30)
    print()

    input_df["target"] = input_df["target"].map(TARGET_DICT)

    elapsed_times = []

    for fold_id in tqdm(range(0, input_df[fold_col].max()+1), desc="Training.. Fold"):
        start_time = time.time()
        X_train = input_df[input_df[fold_col] != fold_id][x_col]
        y_train = input_df[input_df[fold_col] != fold_id][y_col]
        X_val = input_df[input_df[fold_col] == fold_id][x_col]
        y_val = input_df[input_df[fold_col] == fold_id][y_col]

        X_train = preprocess_text(X_train, prevent_bias=prevent_bias)
        X_val = preprocess_text(X_val, prevent_bias=prevent_bias)

        val_idx = y_val.index.tolist()

        model = model_obj(**model_params)

        # Using external data (Fully-Unbiasing)
        if prevent_bias == 2:
            X_train, y_train = add_external_positive_data(x_series=X_train, y_series=y_train)

        if is_nn:
            model.train(X_train, y_train, X_val, y_val, fold_id=f"fold{fold_id}")
        else:
            model.train(X_train, y_train)

        preds, pred_probas = model.predict(X_val)
        input_df.loc[val_idx, TARGET_DICT.keys()] = pred_probas

        for pred_i, pred in enumerate(preds):
            preds[pred_i] = TARGET_INV_DICT[pred] if pred in [0, 1, 2, 3, 4] else pred

        input_df.loc[val_idx, "pred"] = preds
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_times.append(elapsed_time)

    print("\nTraining finished! Result:\n")

    input_df["target"] = input_df["target"].map(TARGET_INV_DICT)

    print(classification_report(input_df["target"],
                                input_df["pred"],
                                output_dict=False,
                                digits=4))
    mean_fold_time = np.round(np.mean(elapsed_times), 2)
    std_fold_time = np.round(np.std(elapsed_times), 2)
    print(f"Mean Fold Time: {mean_fold_time}s +- {std_fold_time}s")

    input_df.to_csv(f"{MODEL_CV_RESULT_PATH}/{experiment_name}_OOF.csv",
                    index=False)

    if add_to_zoo:
        zoo_member_dict = {fold_col: {experiment_name: classification_report(input_df["target"], input_df["pred"],
                                                                             output_dict=True)}}

        zoo_member_dict[fold_col][experiment_name]["mean_fold_time"] = mean_fold_time
        zoo_member_dict[fold_col][experiment_name]["std_fold_time"] = std_fold_time

        zoo = read_model_zoo()
        zoo = update_zoo(zoo, zoo_member_dict)
        write_model_zoo(zoo)
