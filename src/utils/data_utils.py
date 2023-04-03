import collections.abc
import json

import pandas as pd

from .constants import PROCESSED_DATA_PATH, ZOO_PATH


def update_zoo(zoo_dict, new_member):
    for k, v in new_member.items():
        if isinstance(v, collections.abc.Mapping):
            zoo_dict[k] = update_zoo(zoo_dict.get(k, {}), v)
        else:
            zoo_dict[k] = v
    return zoo_dict


def read_model_zoo():
    f = open(ZOO_PATH)
    zoo = json.load(f)
    return zoo


def write_model_zoo(zoo_dict):
    out_file = open(ZOO_PATH, "w")
    json.dump(zoo_dict, out_file, indent=6)
    out_file.close()
    print("The result is logged to the model zoo!")


def read_training_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["pred"] = "OTHER"
    return df
