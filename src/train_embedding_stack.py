#!/usr/bin/env python
# coding: utf-8

import argparse

from utils.data_utils import read_training_data
from utils.pipeline_utils import run_cv
from models.embedding_stack_model import EmbeddingStackModel
from models.bert_model import BertModel

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from cuml.svm import SVC


def main(args):
    df = read_training_data()

    if args.head_model == "lgbm":
        model_class = LGBMClassifier
        head_model_args = {"random_state": 1337}
    elif args.head_model == "xgb":
        model_class = XGBClassifier
        head_model_args = {"tree_method": 'gpu_hist',
                           "random_state": 1337}
    elif args.head_model == "catboost":
        model_class = CatBoostClassifier
        head_model_args = {"task_type": "GPU",
                           "random_state": 1337}
    elif args.head_model == "svc":
        model_class = SVC
        head_model_args = {}

    retrain_arg = " (Fine-Tuned)" if args.retrain_embed_model else " (Vanilla)"

    run_cv(model_obj=EmbeddingStackModel,
           model_params={"embed_model_path": args.embedding_model_path,
                         "head_model": model_class,
                         "head_model_args": head_model_args,
                         "retrain_embed_model": args.retrain_embed_model},
           input_df=df,
           fold_col=args.fold_name,
           x_col=args.xcol,
           y_col=args.ycol,
           experiment_name=f"{args.embedding_model_path}{retrain_arg} Embeddings + {args.head_model}",
           add_to_zoo=args.add_zoo,
           is_nn=True
           )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-embedding-model-path', type=str, default="dbmdz/bert-base-turkish-128k-uncased")
    parser.add_argument('-head-model', type=str, default="svc",
                        help="Choices are: 'lgbm', 'xgb', 'catboost', 'svc'")
    parser.add_argument('--retrain-embed-model', action='store_true')

    parser.add_argument('-fold-name', type=str, default="public_fold")
    parser.add_argument('-xcol', type=str, default="text")
    parser.add_argument('-ycol', type=str, default="target")
    parser.add_argument('--add-zoo', action='store_true')

    args = parser.parse_args()
    main(args)
