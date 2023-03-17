#!/usr/bin/env python
# coding: utf-8

import argparse

from utils.data_utils import read_training_data
from utils.pipeline_utils import run_cv
from models.embedding_gbdt_model import EmbeddingGBDTModel


def main(args):
    df = read_training_data()
    run_cv(model_obj=EmbeddingGBDTModel(model_path=args.embedding_model_path,
                                        batch_size=args.batch_size),
           input_df=df,
           fold_col=args.fold_name,
           x_col=args.xcol,
           y_col=args.ycol,
           experiment_name=f"{args.embedding_model_path} Embeddings + LGB",
           add_to_zoo=args.add_zoo
           )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-embedding-model-path', type=str, default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    parser.add_argument('-batch-size', type=int, default=16)
    parser.add_argument('-fold-name', type=str, default="public_fold")
    parser.add_argument('-xcol', type=str, default="text")
    parser.add_argument('-ycol', type=str, default="target")
    parser.add_argument('--add-zoo', action='store_true')
    args = parser.parse_args()
    main(args)
