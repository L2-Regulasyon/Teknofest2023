#!/usr/bin/env python
# coding: utf-8

import argparse

from utils.data_utils import read_training_data
from utils.pipeline_utils import run_cv

from models.vector_stack_model import VectorStackModel

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


def main(args):
    df = read_training_data()

    if args.head_model == "lgbm":
        model_class = LGBMClassifier
        head_model_args = {"random_state": 1337}
    elif args.head_model == "xgb":
        model_class = XGBClassifier
        head_model_args = {"random_state": 1337}
    elif args.head_model == "catboost":
        model_class = CatBoostClassifier
        head_model_args = {"task_type": "GPU",
                           "random_state": 1337}

    if args.vector_model == "fasttext":
        vect_model_class = "fasttext"
        vect_model_args = {}
    elif args.vector_model == "tfidf":
        vect_model_class = TfidfVectorizer
        vect_model_args = {"analyzer": "char",
                           "ngram_range": (1, 10)}

    experiment_name = f"{args.vector_model.replace('/', '-')} Embeddings + {args.head_model}"

    run_cv(model_obj=VectorStackModel,
           model_params={"vector_model": vect_model_class,
                         "vector_model_args": vect_model_args,
                         "head_model": model_class,
                         "head_model_args": head_model_args},
           input_df=df,
           fold_col=args.fold_name,
           x_col=args.xcol,
           y_col=args.ycol,
           experiment_name=experiment_name,
           add_to_zoo=args.add_zoo
           )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-vector-model', type=str, default="tfidf",
                        help="Choices are: 'tfidf', 'fasttext'")
    parser.add_argument('-head-model', type=str, default="lgbm",
                        help="Choices are: 'lgbm', 'xgb', 'catboost'")
    parser.add_argument('-fold-name', type=str, default="public_fold")
    parser.add_argument('-xcol', type=str, default="text")
    parser.add_argument('-ycol', type=str, default="target")
    parser.add_argument('-experiment-name', type=str, default="TFIDF + LGB")
    parser.add_argument('--add-zoo', action='store_true')
    args = parser.parse_args()
    main(args)
