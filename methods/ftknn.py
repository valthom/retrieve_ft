import os
import scipy
from collections import defaultdict
from tqdm import trange

import numpy as np
import torch

from utils import (
    get_sizes_per_class,
    compute_metrics,
    pad_x,
    MulticlassFaiss,
    SingleclassFaiss,
    clone_linear_layer,
    fix_missing,
)
from methods.pfknn import compute_embedding


def train_ft_knn(args, model, data, writer, experiment_path):
    X_train, y_train = data["X_train"], data["y_train"]
    X_train_one_hot = data["X_train_one_hot"]
    dataset_name = data["dataset_info"]["name"]

    datasets_evaluated = {
        "valid": ["valid"],
        "valid_test": ["valid", "test"],
        "all": ["valid", "test", "train"],
    }

    if args.better_selection:
        # Shuffle X_train and y_train
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        selection_counts = (
            np.ones(len(X_train), dtype=np.longlong)
            * args.num_steps
            * args.num_epochs
            * args.batch_size
            * 100
        )

    # Class balance and initialization checks
    num_classes = len(np.unique(y_train))
    num_features = X_train.shape[1]
    if any(sum(y_train == c) == 0 for c in range(num_classes)):
        raise Exception("Error: Missing class in this split...")

    if args.optim == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.opt_weight_decay
        )
    elif args.optim == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.0)
    else:
        raise Exception("Error: Optimizer not supported...")

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.num_epochs * args.num_steps, eta_min=args.lr / 10
        )

    if args.dynamic:
        num_retrievals = [
            min(int(10 * np.sqrt(len(data["X_train"]))), args.context_length),
            args.train_query_length,
        ]
    elif args.dynamic_small:
        num_retrievals = [
            min(int(8 * np.sqrt(len(data["X_train"]))), args.context_length),
            args.train_query_length,
        ]
    else:
        num_retrievals = [args.context_length, args.train_query_length]

    sizes_per_class = get_sizes_per_class(
        args, y_train, num_classes, context_length=sum(num_retrievals)
    )

    best_metric, best_epoch = -np.inf, 0
    saved_data = defaultdict(list)
    for epoch in trange(args.num_epochs):
        embedding_layer = clone_linear_layer(model.encoder, device="cpu")

        if args.knn_mode == "multiclass":
            print("Using multiclass knn")
            # faiss_knn = MulticlassFaiss(X_train, X_train, y_train)
            faiss_knn = MulticlassFaiss(
                compute_embedding(model, data, X_train_one_hot if args.onehot_retrieval else X_train, embedding_type=args.embedding, embedding_layer=embedding_layer),
                X_train,
                y_train,
            )
        else:
            print("Using singleclass knn")
            # faiss_knn = SingleclassFaiss(X_train, y_train)
            faiss_knn = SingleclassFaiss(
                compute_embedding(model, data, X_train_one_hot if args.onehot_retrieval else X_train, embedding_type=args.embedding, embedding_layer=embedding_layer),
                y_train,
            )

        # Eval
        if epoch % args.eval_interval == 0:
            prev_best_metric = best_metric

            for split in datasets_evaluated[args.splits_evaluated]:
                loss, pred = eval_ft_knn(args, model, data[split + "_loader"], data)
                acc, f1, auc = compute_metrics(
                    data["y_" + split], scipy.special.softmax(pred, axis=1)
                )

                # Calculate metric for early stopping
                if args.early_stopping_metric == "negloss":
                    metric = -loss
                elif args.early_stopping_metric == "acc":
                    metric = acc
                elif args.early_stopping_metric == "f1":
                    metric = f1
                elif args.early_stopping_metric == "auc":
                    metric = auc
                else:
                    raise ValueError("Invalid early stopping metric")

                if split == "valid" and metric > prev_best_metric:
                    best_epoch = epoch
                    best_metric = metric
                    os.makedirs(
                        experiment_path + f"/data/{dataset_name}/", exist_ok=True
                    )
                    torch.save(
                        model.state_dict(),
                        experiment_path + f"/data/{dataset_name}/model_best.pth",
                    )

                writer.add_scalar(dataset_name + "/loss/" + split, loss, epoch)
                writer.add_scalar(dataset_name + "/acc/" + split, acc, epoch)
                writer.add_scalar(dataset_name + "/f1/" + split, f1, epoch)
                writer.add_scalar(dataset_name + "/auc/" + split, auc, epoch)

                if args.save_data:
                    saved_data[f"loss_{split}"].append(loss)
                    saved_data[f"acc_{split}"].append(acc)
                    saved_data[f"f1_{split}"].append(f1)
                    saved_data[f"auc_{split}"].append(auc)

            writer.flush()
            if best_epoch + args.early_stopping_rounds < epoch:
                print("Early Stopping: ", best_epoch, epoch)
                break

        # Training
        total_loss = 0.0

        count = 0
        for X_b, y_b, X_oh_b in data["train_loader"]:
            count += 1
            if count > args.num_steps:
                break

            if args.better_selection:
                selected_index = np.random.choice(
                    len(X_train), 2, p=selection_counts / selection_counts.sum()
                )
                X_b, y_b = X_train[selected_index], y_train[selected_index]

            if args.knn_mode == "multiclass":
                # indices_X_nni, y_nni = faiss_knn.get_knn_indices(X, sizes_per_class)
                indices_X_nni, y_nni = faiss_knn.get_knn_indices(
                    compute_embedding(model, data, X_oh_b if args.onehot_retrieval else X_b.numpy(), embedding_type=args.embedding, embedding_layer=embedding_layer),
                    sizes_per_class,
                )
                X_nni = np.concatenate(
                    [
                        X_train[y_train == i][indices]
                        for i, indices in enumerate(indices_X_nni)
                    ],
                    axis=1,
                )
                X_nni = np.swapaxes(X_nni, 0, 1)
                y_nni = y_nni.reshape(-1, 1).repeat(X_b.shape[0], axis=1)
            else:
                # indices_X_nni, y_nni = faiss_knn.get_knn_indices(X, sum(sizes_per_class))
                indices_X_nni, y_nni = faiss_knn.get_knn_indices(
                    compute_embedding(model, data, X_oh_b if args.onehot_retrieval else X_b.numpy(), embedding_type=args.embedding, embedding_layer=embedding_layer),
                    sum(sizes_per_class),
                )
                X_nni = X_train[indices_X_nni]
                X_nni = np.swapaxes(X_nni, 0, 1)
                y_nni = np.swapaxes(y_nni, 0, 1)

            if args.better_selection:
                for neighbors in indices_X_nni:
                    selection_counts[neighbors.flatten()] -= 1

            if args.fix_missing:
                if args.knn_mode == "multiclass":
                    X_nni, y_nni = fix_missing(
                        X_nni, y_nni, sizes_per_class, indices_X_nni, y_train
                    )
                else:
                    size = sum(sizes_per_class)
                    missing_indices = np.where(indices_X_nni[0] == -1)[0]
                    if len(missing_indices) > 0:
                        miss_idx = missing_indices[0]
                        X_nni = X_nni[:miss_idx]
                        y_nni = y_nni[:miss_idx]
                        print("missing!!!!!!!!!!!!!!!!!!!!!", size - miss_idx)

            if args.exact_knn:
                X_nni, y_nni = X_nni[::-1].copy(), y_nni[::-1].copy()
                eval_pos = num_retrievals[0]
                assert eval_pos == len(X_nni) - 1

            else:
                rand_perm_indices = np.random.permutation(len(X_nni))
                X_nni, y_nni = X_nni[rand_perm_indices], y_nni[rand_perm_indices]

                eval_pos = num_retrievals[0]

            # assert eval_pos < len(X_nni), "not enough data for training"
            if eval_pos >= len(X_nni):
                print("eval pos is smaller than X_nni")
                eval_pos = len(X_nni) - 1

            X_nni, y_nni = (
                pad_x(torch.Tensor(X_nni)).to(args.device),
                torch.Tensor(y_nni).to(args.device),
            )
            y_label = y_nni[eval_pos:, :].flatten()

            logits = model(
                x_src=X_nni,
                y_src=y_nni,
                eval_pos=eval_pos,
                normalization=True,
                outlier_clipping=False,
                nan_replacement=False,
                used_features=num_features,
            )[..., :num_classes]

            logits = logits.reshape([-1, num_classes])
            batch_loss = torch.nn.functional.cross_entropy(
                logits, y_label.long(), reduction="none"
            )
            total_loss += batch_loss.mean().detach()

            batch_loss.mean().backward()
            opt.step()
            opt.zero_grad()
            if args.scheduler:
                scheduler.step()

        loss = total_loss / count

        writer.add_scalar(
            dataset_name + "/loss/train", loss.detach().cpu().numpy(), epoch
        )
        writer.add_scalar(dataset_name + "/lr", opt.param_groups[0]["lr"], epoch)

    if args.save_data:
        os.makedirs(experiment_path + f"/data/{dataset_name}/", exist_ok=True)
        np.savez(experiment_path + f"/data/{dataset_name}/saved_data.npz", **saved_data)


@torch.no_grad()
def eval_ft_knn(args, model, data_loader, data):
    num_features = data_loader.dataset.tensors[0].shape[-1]
    num_classes = len(np.unique(data_loader.dataset.tensors[1]))
    X_train, y_train = data["X_train"], data["y_train"]
    X_train_one_hot = data["X_train_one_hot"]

    y_pred = []
    loss = 0

    if args.dynamic:
        num_retrieval = min(
            int(10 * np.sqrt(len(data["X_train"]))), args.context_length
        )
    elif args.dynamic_small:
        num_retrieval = min(
            int(8 * np.sqrt(len(data["X_train"]))), args.context_length
        )
    else:
        num_retrieval = args.context_length

    embedding_layer = clone_linear_layer(model.encoder, device="cpu")

    if args.knn_mode == "multiclass":
        print("Using multiclass knn")
        # faiss_knn = MulticlassFaiss(X_train, X_train, y_train)
        faiss_knn = MulticlassFaiss(
            compute_embedding(model, data, X_train_one_hot if args.onehot_retrieval else X_train, embedding_type=args.embedding, embedding_layer=embedding_layer),
            X_train,
            y_train,
        )
    else:
        print("Using singleclass knn")
        # faiss_knn = SingleclassFaiss(X_train, y_train)
        faiss_knn = SingleclassFaiss(
            compute_embedding(model, data, X_train_one_hot if args.onehot_retrieval else X_train, embedding_type=args.embedding, embedding_layer=embedding_layer),
            y_train,
        )
    sizes_per_class = get_sizes_per_class(
        args, y_train, num_classes, context_length=num_retrieval
    )

    for X, y, X_oh in data_loader:
        X_cuda, y_cuda = X.to(args.device), y.to(args.device)
        cur_batch_size = X.shape[0]

        if args.knn_mode == "multiclass":
            # indices_X_nni, y_nni = faiss_knn.get_knn_indices(X, sizes_per_class)
            indices_X_nni, y_nni = faiss_knn.get_knn_indices(
                compute_embedding(model, data, X_oh if args.onehot_retrieval else X.numpy(), embedding_type=args.embedding, embedding_layer=embedding_layer),
                sizes_per_class,
            )
            X_nni = np.concatenate(
                [
                    X_train[y_train == i][indices]
                    for i, indices in enumerate(indices_X_nni)
                ],
                axis=1,
            )
            X_nni = np.swapaxes(X_nni, 0, 1)
            y_nni = y_nni.reshape(-1, 1).repeat(cur_batch_size, axis=1)
        else:
            # indices_X_nni, y_nni = faiss_knn.get_knn_indices(X, sum(sizes_per_class))
            indices_X_nni, y_nni = faiss_knn.get_knn_indices(
                compute_embedding(model, data, X_oh if args.onehot_retrieval else X.numpy(), embedding_type=args.embedding, embedding_layer=embedding_layer),
                num_retrieval,
            )
            X_nni = X_train[indices_X_nni]
            X_nni = np.swapaxes(X_nni, 0, 1)
            y_nni = np.swapaxes(y_nni, 0, 1)

        if args.fix_missing:
            if args.knn_mode == "multiclass":
                X_nni, y_nni = fix_missing(
                    X_nni, y_nni, sizes_per_class, indices_X_nni, y_train
                )
            else:
                size = num_retrieval
                missing_indices = np.where(indices_X_nni[0] == -1)[0]
                if len(missing_indices) > 0:
                    miss_idx = missing_indices[0]
                    X_nni = X_nni[:miss_idx]
                    y_nni = y_nni[:miss_idx]
                    print("missing!!!!!!!!!!!!!!!!!!!!!", size - miss_idx)

        X_nni, y_nni = (
            pad_x(torch.Tensor(X_nni).to(args.device)),
            torch.Tensor(y_nni).to(args.device),
        )
        X_cuda, y_cuda = pad_x(X_cuda.unsqueeze(0)), y_cuda.unsqueeze(0)

        logits = model(
            x_src=torch.cat([X_nni, X_cuda], dim=0),
            y_src=torch.cat([y_nni, y_cuda], dim=0),
            eval_pos=len(X_nni),
            normalization=True,
            outlier_clipping=False,
            nan_replacement=False,
            used_features=num_features,
        )[..., :num_classes]

        logits = logits.squeeze(0)
        loss += (
            torch.nn.functional.cross_entropy(
                logits, y_cuda.long().squeeze(0), reduction="none"
            )
            .detach()
            .sum()
        )
        y_pred.append(logits)

    loss = (loss / len(data_loader.dataset)).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()

    return loss, y_pred

