import numpy as np
import torch
import scipy
import os
import pandas as pd

from config import parse_args
from pfn import PFN
from methods.pfdt import eval_pfdt
from methods.distill import train_dist, eval_dist, train_dist_knn, eval_dist_knn
from methods.finetune import eval_ft, train_ft
from methods.ftknn import train_ft_knn, eval_ft_knn
from methods.pfknn import eval_pfknn, eval_pfknn_ensemble_dist
from methods.vanilla import eval_tabpfn

from dataset import PFNDataset
from utils import setup_experiment, compute_metrics, save_numpy, create_dataloaders

import xgboost as xgb

if __name__ == "__main__":
    args = parse_args()

    pfnDataset = PFNDataset(args)
    datasets = pfnDataset.load()
    writer, results_file, experiment_path = setup_experiment(args)

    model, _ = PFN.load_old(device=args.device, path=args.model_path)
    model.eval()

    num_datasets = 0
    for data in datasets:
        if not data:  # in case data contains None
            print("data contains None")
            continue

        dataset_name = data["dataset_info"]["name"]
        print(dataset_name)

        # if len(set(list(np.unique(data["y_train"]))).difference(set(list(np.unique(data["y_test"])))))!= 0 or \
        #     len(set(list(np.unique(data["y_test"]))).difference(set(list(np.unique(data["y_train"])))))!= 0:
        #     print("Error: Missing class in this split...")
        #     continue

        num_datasets += 1

        create_dataloaders(args, data)

        import time
        t0 = time.time()

        # Train and Eval
        if args.method == "vanilla":
            if args.load_model:
                model.load_state_dict(
                    torch.load(
                        f"results/tabzilla/ftknn_raw_uncondfull2_split{args.split}/data/{dataset_name}/model_best.pth"
                    )
                )
            loss, pred = eval_tabpfn(args, model, data, data["test_loader"])
            save_numpy(f"{experiment_path}/data/{dataset_name}/y_pred_best.npy", pred)

        elif args.method == "dist":
            if args.dynamic:
                args.context_length = min(
                    int(10 * np.sqrt(len(data["X_train"]))), args.context_length
                )

            if args.type == "dist":
                train_dist(args, model, data, writer, experiment_path)

                X_dist = torch.Tensor(
                    np.load(f"{experiment_path}/data/{dataset_name}/Xdist_best.npy")
                ).to(args.device)
                y_dist = torch.Tensor(
                    np.load(f"{experiment_path}/data/{dataset_name}/ydist_best.npy")
                ).to(args.device)
                loss, pred = eval_dist(args, model, data["test_loader"], X_dist, y_dist)
                save_numpy(
                    f"{experiment_path}/data/{dataset_name}/y_pred_best.npy", pred
                )
            elif args.type == "distknn":
                train_dist_knn(args, model, data, writer, experiment_path)

                X_dist = torch.Tensor(
                    np.load(f"{experiment_path}/data/{dataset_name}/Xdist_best.npy")
                ).to(args.device)
                y_dist = torch.Tensor(
                    np.load(f"{experiment_path}/data/{dataset_name}/ydist_best.npy")
                ).to(args.device)
                loss, pred = eval_dist_knn(
                    args,
                    model,
                    data["test_loader"],
                    X_dist,
                    y_dist,
                    data,
                    experiment_path,
                    dataset_name,
                    "test",
                )
                save_numpy(
                    f"{experiment_path}/data/{dataset_name}/y_pred_best.npy", pred
                )
            else:
                raise NotImplementedError

        elif args.method == "knn":
            if args.dynamic:
                args.context_length = min(
                    int(10 * np.sqrt(len(data["X_train"]))), args.context_length
                )
            elif args.dynamic_small:
                args.context_length = min(
                    int(8 * np.sqrt(len(data["X_train"]))), args.context_length
                )

            if args.type == "knn":
                loss, pred = eval_pfknn(args, model, data, data["test_loader"])
            elif args.type == "knn_with_ft":
                model.load_state_dict(
                    torch.load(
                        f"results/tabzilla/finetune2_split{args.split}/data/{dataset_name}/model_best.pth"
                    )
                )
                model.eval()
                loss, pred = eval_pfknn(
                    args, model, data, data["test_loader"], normalize_data=True
                )
            elif args.type == "knn_tt":
                loss, pred = eval_pfknn(args, model, data, data["test_loader"])
            elif args.type == "knn_kernels":
                pred = []
                for k_val in range(100, 1100, 100):
                    pred.append(
                        np.load(
                            f"results/tabzilla/knn_k{k_val}/data/{dataset_name}/y_pred_best.npy"
                        )
                    )
                pred = np.stack(pred, axis=0).mean(0)
                loss = 0
            else:
                loss, pred = eval_pfknn_ensemble_dist(
                    args,
                    model,
                    data,
                    data["test_loader"],
                    pred_folder=f"{args.ensemble_dist_folder}/{dataset_name}",
                )
            save_numpy(f"{experiment_path}/data/{dataset_name}/y_pred_best.npy", pred)

        elif args.method == "dt":
            loss, pred = eval_pfdt(args, model, data, data["test_loader"])
            save_numpy(f"{experiment_path}/data/{dataset_name}/y_pred_best.npy", pred)

        elif args.method == "ft":
            model, _ = PFN.load_old(device=args.device, path=args.model_path)
            model.train()
            if args.type == "ft":
                if args.dynamic:
                    args.context_length = min(
                        int(10 * np.sqrt(len(data["X_train"]))), args.context_length
                    )

                train_ft(args, model, data, writer, experiment_path)
                model.load_state_dict(
                    torch.load(f"{experiment_path}/data/{dataset_name}/model_best.pth")
                )
                model.eval()
                loss, pred = eval_ft(args, model, data["test_loader"], None, None, data)

            elif args.type == "ftknn":
                train_ft_knn(args, model, data, writer, experiment_path)
                model.load_state_dict(
                    torch.load(f"{experiment_path}/data/{dataset_name}/model_best.pth")
                )
                model.eval()
                loss, pred = eval_ft_knn(args, model, data["test_loader"], data)

            else:
                raise NotImplementedError
            save_numpy(f"{experiment_path}/data/{dataset_name}/y_pred_best.npy", pred)

        elif args.method == "xgb":
            params = {}
            params["verbosity"] = 0
            params["tree_method"] = "gpu_hist"
            params["gpu_id"] = "0"
            params["objective"] = "multi:softprob"
            params["num_class"] = len(np.unique(data["y_train"]))
            params["eval_metric"] = "mlogloss"

            # tabzilla default hyperparameters
            # params["max_depth"] = 5
            # params["alpha"] = 1e-4
            # params["lambda"] = 1e-4
            # params["eta"] = 0.08

            train = xgb.DMatrix(data["X_train"], label=data["y_train"])
            val = xgb.DMatrix(data["X_valid"], label=data["y_valid"])
            eval_list = [(val, "eval")]
            model = xgb.train(
                params,
                train,
                num_boost_round=1000,
                evals=eval_list,
                early_stopping_rounds=20,
                verbose_eval=True,
            )

            pred = model.predict(xgb.DMatrix(data["X_test"]))
            loss = 0

        else:
            raise ValueError("Invalid Method")

        total_time = time.time() - t0

        # Logging
        if args.method in ["vanilla", "knn", "dist", "ft", "dt"]:
            pred = scipy.special.softmax(pred, axis=1)

        acc, f1, auc = compute_metrics(data["y_test"], pred)
        if args.timing:
            print(
                "Best: Loss: {:.4f}, Accuracy: {:.4f}, F1: {:.4f}, AUC: {:.4f}, Time: {:6f}".format(
                    loss, acc, f1, auc, total_time
                )
            )
            results_file.write(f"{dataset_name},{acc},{f1},{auc},{total_time}\n")
            results_file.flush()
        else:
            print(
                "Best: Loss: {:.4f}, Accuracy: {:.4f}, F1: {:.4f}, AUC: {:.4f}".format(
                    loss, acc, f1, auc
                )
            )
            results_file.write(f"{dataset_name},{acc},{f1},{auc}\n")
            results_file.flush()

    if not args.timing:
        df = pd.read_csv(
            os.path.join(experiment_path, "results.csv"),
            sep=",",
            header=None,
            names=["name", "acc", "f1", "auc"],
        )
        medians = df.iloc[-num_datasets:, 1:].astype(float).median()
        print(medians)
        medians.to_csv(results_file, mode="a", header=True, index=False)

