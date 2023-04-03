#!/usr/bin/env python
# coding: utf-8

import argparse

from utils.data_utils import read_training_data
from utils.pipeline_utils import run_cv, add_external_positive_data
from utils.preprocess_utils import preprocess_text
from models.bert_model import BertModel


def main(args):
    df = read_training_data()

    bias_naming = ""
    if args.prevent_bias == 2:
        bias_naming = "-fully-unbiased"
    elif args.prevent_bias == 1:
        bias_naming = "-casing-unbiased"

    experiment_name = f"toxic-{args.model_path.replace('/', '-')}{bias_naming}"
    model_params = {"model_path": args.model_path,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "tokenizer_max_len": args.tokenizer_max_len,
                    "learning_rate": args.learning_rate,
                    "warmup_ratio": args.warmup_ratio,
                    "weight_decay": args.weight_decay,
                    "llrd_decay": args.llrd_decay,
                    "label_smoothing": args.label_smoothing,
                    "grad_clip": args.grad_clip,
                    "prevent_bias": args.prevent_bias,
                    "mlm_pretrain": args.mlm_pretrain,
                    "mlm_probability": args.mlm_probability,
                    "out_folder": args.out_folder,
                    "experiment_name": experiment_name
                    }
    if args.cv:
        run_cv(model_obj=BertModel,
               model_params=model_params,
               input_df=df,
               fold_col=args.fold_name,
               x_col=args.xcol,
               y_col=args.ycol,
               experiment_name=experiment_name,
               add_to_zoo=args.add_zoo,
               is_nn=True,
               prevent_bias=args.prevent_bias
               )
    else:
        model = BertModel(**model_params)
        X_train = df[args.xcol]
        y_train = df[args.ycol]

        if args.prevent_bias == 2:
            X_train, y_train = add_external_positive_data(x_series=X_train, y_series=y_train)
        X_train = preprocess_text(X_train, prevent_bias=args.prevent_bias)

        model.train(X_train,
                    y_train,
                    fold_id="none")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-model-path', type=str, default="dbmdz/bert-base-turkish-128k-uncased")
    parser.add_argument('-batch-size', type=int, default=32)
    parser.add_argument('-tokenizer-max-len', type=int, default=64)

    parser.add_argument('-learning-rate', type=float, default=7e-5)
    parser.add_argument('-epochs', type=int, default=3)
    parser.add_argument('-warmup-ratio', type=float, default=0.1)
    parser.add_argument('-weight-decay', type=float, default=0.01)
    parser.add_argument('-llrd-decay', type=float, default=0.95)
    parser.add_argument('-label-smoothing', type=float, default=0.05)
    parser.add_argument('-grad-clip', type=float, default=1.0)
    parser.add_argument('-prevent-bias', type=int, default=0)

    parser.add_argument('--mlm-pretrain', action='store_true')
    parser.add_argument('-mlm-probability', type=float, default=0.15)

    parser.add_argument('-out-folder', type=str, default="../checkpoint")
    parser.add_argument('-fold-name', type=str, default="public_fold")
    parser.add_argument('-xcol', type=str, default="text")
    parser.add_argument('-ycol', type=str, default="target_label")
    parser.add_argument('--add-zoo', action='store_true')
    parser.add_argument('--cv', action='store_true')
    args = parser.parse_args()
    main(args)
