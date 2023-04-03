#!/usr/bin/env python
# coding: utf-8

import argparse

from models.voting_bert_model import VotingBertModel


def main(args):
    vem = VotingBertModel(checkpoint_list=args.checkpoint_paths)
    vem.save(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint-paths', nargs='+', help='List of checkpoint paths', required=True)
    parser.add_argument('-o', '--output-path', type=str, help='List of checkpoint paths', required=True)
    args = parser.parse_args()
    main(args)
