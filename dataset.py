from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import os
import gzip
import json


def compute_one_hot_embedding_for_retrieval(X, meta_data):
    cat_idx = meta_data["cat_idx"]
    cat_dims = meta_data["cat_dims"]

    new_X = []
    for i in range(X.shape[1]):
        try:
            if i in cat_idx:
                # k is the index of i in cat_idx
                k = cat_idx.index(i)
                if cat_dims[k] < 100:
                    new_X.append(np.eye(cat_dims[k])[X[:, i].astype(int)])
                else:
                    new_X.append(X[:, i].reshape(-1, 1))
            else:
                new_X.append(X[:, i].reshape(-1, 1))
        except:
            breakpoint()
    return np.concatenate(new_X, axis=1)


def compute_one_hot_embedding(X, meta_data):
    cat_idx = meta_data["cat_idx"]
    cat_dims = meta_data["cat_dims"]
    non_cat_idx = np.setdiff1d(np.arange(X.shape[1]), cat_idx)

    new_X = []
    new_dim = X.shape[1]
    for i in range(X.shape[1]):
        if i in cat_idx:
            # k is the index of i in cat_idx
            k = cat_idx.index(i)
            if new_dim + cat_dims[k] - 1 < 100:
                new_dim += cat_dims[k] - 1
                new_X.append(np.eye(cat_dims[k])[X[:, i].astype(int)])
            else:
                new_X.append(X[:, i].reshape(-1, 1))
        else:
            new_X.append(X[:, i].reshape(-1, 1))
    return np.concatenate(new_X, axis=1)


def compute_fourier_embedding(X, meta_data):
    X_fft = np.fft.fft(X, axis=1)
    return np.concatenate([np.real(X_fft), np.imag(X_fft)], axis=1)


class PFNDataset:
    def __init__(self, args):
        self.args = args
        self.datasets = args.datasets

        if self.datasets == "toy":
            self.dataset_names = [args.toy_dataset_name]
        elif self.datasets == "tabzilla":
            if args.only_dataset:
                self.dataset_names = [args.only_dataset]
                return
            elif args.dataset_list_path:
                with open(args.dataset_list_path, "r") as f:
                    self.dataset_names = list([l.strip() for l in f])
                    return

            all_dataset_names = os.listdir(args.datasets_directory)
            self.dataset_names = []

            # filter
            for dataset_name in all_dataset_names:
                metadata_path = os.path.join(
                    args.datasets_directory, dataset_name, "metadata.json"
                )
                with open(metadata_path, "r") as f:
                    dataset_info = json.load(f)

                    if dataset_info["target_type"] == "regression":
                        continue
                    if args.numerical_only and len(dataset_info["cat_idx"]) > 0:
                        continue
                    if dataset_info["num_instances"] < 2000 and args.filter_size in [
                        "medium",
                        "medium_and_large",
                    ]:
                        continue
                    if dataset_info["num_instances"] > 200_000 and args.filter_size in [
                        "medium",
                        "small_and_medium",
                    ]:
                        continue
                    if dataset_info["num_instances"] >= 2000 and args.filter_size in [
                        "small"
                    ]:
                        continue
                    if dataset_info["num_features"] > args.filter_feature:
                        continue
                    if dataset_info["num_classes"] > args.filter_class:
                        continue
                    if not args.use_nan_datasets and dataset_info["name"] in [
                        "openml__cjs__14967",
                        "openml__higgs__146606",
                        "openml__jm1__3904",
                        "openml__sick__3021",
                    ]:
                        continue

                self.dataset_names.append(dataset_name)
            self.dataset_names = sorted(self.dataset_names)
        elif self.datasets == "cc18":
            self.dataset_names = []  # TODO
        else:
            raise ValueError(f"Unsupported dataset: {self.datasets}")

    def __len__(self):
        return len(self.dataset_names)

    @classmethod
    def generate_toy_data(cls, args):
        if args.toy_dataset_name == "blobs":
            centers = [(-1, 1), (1, -1), (-1, -1), (1, 1)]
            X, y = make_blobs(
                n_samples=args.n_samples,
                centers=centers,
                shuffle=True,
                random_state=args.seed,
                cluster_std=1,
            )
            y = y % 2  # Ensuring binary labels
        elif args.toy_dataset_name == "moons":
            X, y = make_moons(
                n_samples=args.n_samples,
                noise=0.1,
                shuffle=True,
                random_state=args.seed,
            )
        else:
            raise ValueError(f"Unsupported dataset name: {args.toy_dataset_name}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.5, random_state=args.seed
        )

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

        X_train = np.clip(X_train, -args.clipping_val, args.clipping_val)
        X_valid = np.clip(X_valid, -args.clipping_val, args.clipping_val)
        X_test = np.clip(X_test, -args.clipping_val, args.clipping_val)

        return {
            "X_train": X_train,
            "X_valid": X_valid,
            "X_test": X_test,
            "y_train": y_train,
            "y_valid": y_valid,
            "y_test": y_test,
            "dataset_info": {"name": args.toy_dataset_name},
        }

    @classmethod
    def load_tabzilla_data(cls, args, dataset_name):
        X_path = os.path.join(args.datasets_directory, dataset_name, "X.npy.gz")
        y_path = os.path.join(args.datasets_directory, dataset_name, "y.npy.gz")
        metadata_path = os.path.join(
            args.datasets_directory, dataset_name, "metadata.json"
        )
        split_indices_path = os.path.join(
            args.datasets_directory, dataset_name, "split_indeces.npy.gz"
        )

        with gzip.GzipFile(X_path, "r") as f:
            X = np.load(f, allow_pickle=True)
        with gzip.GzipFile(y_path, "r") as f:
            y = np.load(f)
        with gzip.GzipFile(split_indices_path, "rb") as f:
            split_indices = np.load(f, allow_pickle=True)

        with open(metadata_path, "r") as f:
            dataset_info = json.load(f)

        if args.add_noise_std > 0:
            if len(dataset_info["cat_idx"]) == 0:
                X += np.random.normal(0, args.add_noise_std, size=X.shape)

        # TODO
        if np.any(np.isnan(X.astype(float))):
            print("Found None: ", dataset_info["name"])
            return None

        y_unique = np.unique(y)
        y -= np.min(y_unique)
        if args.use_one_hot_emb:
            X_one_hot = compute_one_hot_embedding(X, dataset_info)
        else:
            X_one_hot = compute_one_hot_embedding_for_retrieval(X, dataset_info)

        train_indices, val_indices, test_indices = (
            split_indices[args.split]["train"],
            split_indices[args.split]["val"],
            split_indices[args.split]["test"],
        )
        X_train, X_val, X_test = (
            X[train_indices, :],
            X[val_indices, :],
            X[test_indices, :],
        )
        X_train_one_hot, X_val_one_hot, X_test_one_hot = (
            X_one_hot[train_indices, :],
            X_one_hot[val_indices, :],
            X_one_hot[test_indices, :],
        )
        y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]

        if not args.disable_normalize_data:
            scaler = StandardScaler()
            scaler.fit(X_train)

            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            # clipping
            X_train = np.clip(X_train, -args.clipping_val, args.clipping_val)
            X_val = np.clip(X_val, -args.clipping_val, args.clipping_val)
            X_test = np.clip(X_test, -args.clipping_val, args.clipping_val)

            scaler_one_hot = StandardScaler()
            scaler_one_hot.fit(X_train_one_hot)
            X_train_one_hot = scaler_one_hot.transform(X_train_one_hot)
            X_val_one_hot = scaler_one_hot.transform(X_val_one_hot)
            X_test_one_hot = scaler_one_hot.transform(X_test_one_hot)
            # clipping
            X_train_one_hot = np.clip(
                X_train_one_hot, -args.clipping_val, args.clipping_val
            )
            X_val_one_hot = np.clip(
                X_val_one_hot, -args.clipping_val, args.clipping_val
            )
            X_test_one_hot = np.clip(
                X_test_one_hot, -args.clipping_val, args.clipping_val
            )
        else:
            X = X.astype(float)

        if args.use_one_hot_emb:
            X_train = X_train_one_hot
            X_val = X_val_one_hot
            X_test = X_test_one_hot

        return {
            "X_train": X_train,
            "X_valid": X_val,
            "X_test": X_test,
            "X_train_one_hot": X_train_one_hot,
            "X_valid_one_hot": X_val_one_hot,
            "X_test_one_hot": X_test_one_hot,
            "y_train": y_train,
            "y_valid": y_val,
            "y_test": y_test,
            "dataset_info": dataset_info,
        }

    def load(self):
        if self.datasets == "toy":
            yield from [PFNDataset.generate_toy_data(self.args)]
        elif self.datasets == "tabzilla":
            for dataset_name in self.dataset_names:
                yield PFNDataset.load_tabzilla_data(self.args, dataset_name)
        elif self.datasets == "cc18":
            pass
        else:
            raise ValueError(f"Unsupported dataset: {self.datasets}")
