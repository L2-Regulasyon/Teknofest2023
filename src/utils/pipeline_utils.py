from tqdm.auto import tqdm
from sklearn.metrics import classification_report
from .data_utils import read_model_zoo, write_model_zoo, update_zoo
from .constants import TARGET_INV_DICT

def run_cv(model_obj,
           model_params,
           input_df,
           fold_col,
           x_col,
           y_col,
           experiment_name="NONAME",
           add_to_zoo=False,
           is_nn=False):

    print()
    print("*"*30)
    print("Started CV Training")
    print("*"*30)
    print(f"Experiment: '{experiment_name}'")
    print(f"Fold: '{fold_col}'")
    print(f"Update Zoo: '{add_to_zoo}'")
    print("*"*30)
    print()

    for fold_id in tqdm(sorted(input_df[fold_col].unique()), desc="Training.. Fold"):
        X_train = input_df[input_df[fold_col] != fold_id][x_col]
        y_train = input_df[input_df[fold_col] != fold_id][y_col]
        X_val = input_df[input_df[fold_col] == fold_id][x_col]
        y_val = input_df[input_df[fold_col] == fold_id][y_col]

        val_idx = y_val.index.tolist()

        model = model_obj(**model_params)

        if is_nn:
            model.train(X_train, y_train, X_val, y_val)
        else:
            model.train(X_train, y_train)

        preds = model.predict(X_val)

        for pred_i, pred in enumerate(preds):
            preds[pred_i] = TARGET_INV_DICT[pred] if pred in [0, 1, 2, 3, 4] else pred

        input_df.loc[val_idx, "pred"] = preds

    print("\nTraining finished! Result:\n")

    print(classification_report(input_df["target"],
                                input_df["pred"],
                                output_dict=False,
                                digits=4))
    if add_to_zoo:
        zoo_member_dict = {fold_col: {experiment_name: classification_report(input_df["target"], input_df["pred"],
                                                                             output_dict=True)}}
        zoo = read_model_zoo()
        zoo = update_zoo(zoo, zoo_member_dict)
        write_model_zoo(zoo)
