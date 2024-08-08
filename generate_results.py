import pandas as pd
import os
import json
import numpy as np
from pathlib import Path
import re
import argparse

import pprint

pp = pprint.PrettyPrinter(indent=4)


def convert_json_folder_to_df(json_path):
    method_df = []
    index = 0
    for file in os.listdir(json_path):
        with open(os.path.join(json_path, file), "r") as f:
            data = json.load(f)
            for fold in range(10):
                method_df.append(
                    pd.DataFrame(
                        {
                            "dataset_name": data["dataset"]["name"],
                            "AUC__val": data["scorers"]["val"]["AUC"][fold],
                            "Accuracy__val": data["scorers"]["val"]["Accuracy"][fold],
                            "F1__val": data["scorers"]["val"]["F1"][fold],
                            "Accuracy__test": data["scorers"]["val"]["Accuracy"][fold],
                            "F1__test": data["scorers"]["test"]["F1"][fold],
                            "AUC__test": data["scorers"]["val"]["AUC"][fold],
                            "alg_name": data["model"]["name"],
                            "dataset_fold_id": data["dataset"]["name"]
                            + f"__fold_{fold}",
                            "hparam_source": data["hparam_source"],
                        },
                        index=[index],
                    )
                )
                index += 1
    return pd.concat(method_df, axis=0)


def ends_with_split_number(s):
    return bool(re.search(r"split[0-9]$", s))


# go through all results files and create a big df out of them
def get_local_experiment_df(
    experiments_prefix_to_load,
    method="ftknn",
    parent_path="results/tabzilla/",
    verbose=True,
):
    dfs = []
    for exp_name in os.listdir(parent_path):
        to_load = any(
            [exp_name.startswith(prefix + "_split") for prefix in experiments_prefix_to_load]
        )
        if not to_load:
            continue

        if not ends_with_split_number(exp_name):
            continue

        if verbose:
            print("loading experiment: ", exp_name)
        file_path = Path(parent_path) / exp_name / "results.csv"
        if os.path.exists(file_path):
            res_df = pd.read_csv(
                file_path, sep=",", header=None, names=["name", "acc", "f1", "auc"]
            )
            res_df = res_df[res_df["name"].str.contains("openml__")]
            res_df.rename(
                columns={
                    "name": "dataset_name",
                    "acc": "Accuracy__test",
                    "f1": "F1__test",
                    "auc": "AUC__test",
                },
                inplace=True,
            )
            res_df["alg_name"] = method
            res_df["dataset_fold_id"] = res_df["dataset_name"].apply(
                lambda x: x + "__fold_" + exp_name[-1]
            )
        else:
            raise FileNotFoundError

        dfs.append(res_df)
    return pd.concat(dfs, axis=0)


def get_best_validated_algs_for_datasets(
    df, include_datasets, include_algorithms=[], verbose=True
):
    if not include_algorithms:
        include_algorithms = df["alg_name"].unique()

    all_dfs = []
    for alg in include_algorithms:
        if verbose:
            print(f"{alg}")

        filtered_df = df.loc[
            (df["alg_name"] == alg)
            & (df["dataset_fold_id"].str.endswith("fold_0"))
            & (df["dataset_name"].isin(include_datasets))
        ]

        filtered_df = filtered_df.drop_duplicates(
            subset=["dataset_fold_id", "alg_name", "hparam_source"], keep="first"
        )
        group_df = filtered_df.groupby("hparam_source")

        num_datasets_for_alg = group_df.count().min().values[0]
        if np.isnan(num_datasets_for_alg) or num_datasets_for_alg < len(
            include_datasets
        ):
            if verbose:
                print(
                    f"Skipping {alg}: missing {len(include_datasets) - num_datasets_for_alg} datasets.."
                )
                print("-" * 50)
            continue

        if verbose:
            print(f"Number of hyper searches done: {len(group_df)}")

        best_hyper = group_df[["AUC__val"]].mean()["AUC__val"].idxmax()
        all_dfs.append(
            df.loc[
                (df["alg_name"] == alg)
                & (df["hparam_source"] == best_hyper)
                & (df["dataset_name"].isin(include_datasets))
            ].drop_duplicates(subset=["dataset_fold_id", "alg_name"], keep="first")
        )

        if verbose:
            print("-" * 50)

    return pd.concat(all_dfs).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate results for tabzilla")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--latex", action="store_true", help="Print latex output")

    args = parser.parse_args()

    experiments = {
        "ftknn": ["ftknn_test", "ftknn_test_small"],
        "ftknn_onehot": ["ftknn_onehot", "ftknn_onehot_small"],
        "ftknn_onehot_retrieval": ["ftknn_onehot_retrieval"],
        "ftknn_balance": ["ftknn_balance"],
        "xgb1k": ["xgb_1000", "xgb_1000_small"],
    }

    # load all experiments
    all_df = pd.read_csv("results/metadataset_clean.csv")[
        [
            "dataset_fold_id",
            "dataset_name",
            "alg_name",
            "AUC__val",
            "AUC__test",
            "Accuracy__val",
            "Accuracy__test",
            "F1__val",
            "F1__test",
            "hparam_source",
        ]
    ]
    for exp_name, exp_folders in experiments.items():
        local_df = get_local_experiment_df(
            exp_folders,
            method=exp_name,
            parent_path="results/tabzilla/",
            verbose=args.verbose,
        )

        all_df = pd.concat([local_df, all_df]).reset_index(drop=True).fillna(0)

    # datasets partitions
    datasets = {
        "all": [l.strip() for l in open("data_splits/tabpfn_datasets.csv", "r")],
        "small": [l.strip() for l in open("data_splits/add_datasets.csv", "r")],
        "medium": [l.strip() for l in open("data_splits/medium_datasets.csv", "r")],
    }

    for name, dataset in datasets.items():
        print(name)

        validated_df = get_best_validated_algs_for_datasets(
            all_df, dataset, verbose=args.verbose
        )
        validated_df[["Accuracy__test", "F1__test", "AUC__test"]] = validated_df[
            ["Accuracy__test", "F1__test", "AUC__test"]
        ].apply(pd.to_numeric)

        final_df = validated_df.groupby(["alg_name", "dataset_name"]).apply(
            lambda x: pd.Series(
                {
                    "acc\_mean": x["Accuracy__test"].mean(),
                    "f1\_mean": x["F1__test"].mean(),
                    "auc\_mean": x["AUC__test"].mean(),
                    "acc\_std": x["Accuracy__test"].std(),
                    "f1\_std": x["F1__test"].std(),
                    "auc\_std": x["AUC__test"].std(),
                }
            )
        )

        mean_df = (
            final_df.groupby("alg_name")
            .mean()
            .sort_values("auc\_mean", ascending=False)
        )
        median_df = (
            final_df.groupby("alg_name")
            .median()
            .sort_values("auc\_mean", ascending=False)
            .rename(
                columns={
                    "acc\_mean": "acc\_median",
                    "f1\_mean": "f1\_median",
                    "auc\_mean": "auc\_median",
                }
            )
        )

        if args.latex:
            print(mean_df.applymap(lambda x: round(x, 3)).to_latex(float_format="%.3f"))
            print(
                median_df.applymap(lambda x: round(x, 3)).to_latex(float_format="%.3f")
            )
        else:
            print(mean_df)
            print(median_df)
        print("-" * 50)

