import math
import os
import scipy
from tqdm import trange

import numpy as np
import torch

from utils import (
    get_sizes_per_class,
    compute_metrics,
    pad_x,
    MulticlassFaiss,
    get_sizes_per_class,
)
from methods.pfknn import compute_embedding


def dist_initialization(args, data, num_classes, num_features):
    X_train, y_train = data["X_train"], data["y_train"]

    sizes_per_class = get_sizes_per_class(args, y_train, num_classes)

    # Create y_dist
    y_dist = torch.Tensor(
        np.concatenate([np.array([c] * size) for c, size in enumerate(sizes_per_class)])
    ).to(args.device)

    # Initialize X_dist
    if args.init == "random":
        X_dist = torch.cat(
            [
                torch.randn((args.context_length, 1)) / np.sqrt(num_features)
                for _ in range(num_features)
            ],
            dim=1,
        ).cuda()

    elif args.init == "random_sample":
        # Sample with replacement when using small datasets
        # sample_replacement = len(X_train) < args.context_length
        sample_replacement = False
        for c, size in enumerate(sizes_per_class):
            if size > sum(y_train == c):
                sample_replacement = True
                break

        X_dist = torch.Tensor(
            np.concatenate(
                [
                    X_train[y_train == c][
                        np.random.choice(
                            sum(y_train == c), size, replace=sample_replacement
                        ),
                        :,
                    ]
                    for c, size in enumerate(sizes_per_class)
                ],
                0,
            )
        ).to(args.device)
    else:
        raise NotImplementedError

    return X_dist, y_dist


def train_dist(args, model, data, writer, experiment_path):
    X_train, X_val, y_train, y_val = (
        data["X_train"],
        data["X_valid"],
        data["y_train"],
        data["y_valid"],
    )
    dataset_name = data["dataset_info"]["name"]

    # Shuffle X_train and y_train
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    # Class balance and initialization checks
    num_classes = len(np.unique(y_train))
    num_features = X_train.shape[1]
    if any(sum(y_train == c) == 0 for c in range(num_classes)):
        raise Exception("Error: Missing class in this split...")

    # Initialization
    X_dist, y_dist = dist_initialization(args, data, num_classes, num_features)
    X_dist.requires_grad_(True)

    opt = torch.optim.AdamW([X_dist], lr=args.lr, weight_decay=args.opt_weight_decay)
    n_batches = math.ceil(len(X_train) / args.batch_size)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs * n_batches, eta_min=0)

    best_metric, best_epoch = -np.inf, 0
    for epoch in trange(args.num_epochs):
        # Eval
        if epoch % args.eval_interval == 0:
            prev_best_metric = best_metric

            for split in ["valid"]:
                loss, pred = eval_dist(
                    args, model, data[split + "_loader"], X_dist, y_dist
                )
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
                    np.save(
                        experiment_path + f"/data/{dataset_name}/" + "Xdist_best.npy",
                        X_dist.detach().cpu().numpy(),
                    )
                    np.save(
                        experiment_path + f"/data/{dataset_name}/" + "ydist_best.npy",
                        y_dist.cpu().numpy(),
                    )

                writer.add_scalar(dataset_name + "/loss/" + split, loss, epoch)
                writer.add_scalar(dataset_name + "/acc/" + split, acc, epoch)
                writer.add_scalar(dataset_name + "/f1/" + split, f1, epoch)
                writer.add_scalar(dataset_name + "/auc/" + split, auc, epoch)

            writer.flush()
            if best_epoch + args.early_stopping_rounds < epoch:
                print("Early Stopping: ", best_epoch, epoch)
                break

        # Saving
        if epoch % args.save_interval == 0:
            os.makedirs(experiment_path + f"/data/{dataset_name}/", exist_ok=True)
            np.save(
                experiment_path
                + f"/data/{dataset_name}/"
                + "Xdist_"
                + str(epoch)
                + ".npy",
                X_dist.detach().cpu().numpy(),
            )
            np.save(
                experiment_path
                + f"/data/{dataset_name}/"
                + "ydist_"
                + str(epoch)
                + ".npy",
                y_dist.cpu().numpy(),
            )

        # Training
        total_loss = 0.0
        total_energy = 0.0
        total_grad_norm = 0.0

        for X_b, y_b in data["train_loader"]:
            X_b, y_b = X_b.to(args.device), y_b.to(args.device)

            cur_num_features = X_b.shape[1]

            X_b, y_b = (
                pad_x(X_b.unsqueeze(1)).to(args.device),
                y_b.unsqueeze(1).to(args.device),
            )
            X_dist_paded, y_dist_padded = (
                pad_x(X_dist.unsqueeze(1)),
                y_dist.unsqueeze(1),
            )

            logits = model(
                x_src=torch.cat([X_dist_paded, X_b], dim=0),
                y_src=torch.cat([y_dist_padded, y_b], dim=0),
                eval_pos=len(X_dist),
                normalization=False,
                outlier_clipping=False,
                nan_replacement=False,
                used_features=cur_num_features,
            )[..., :num_classes]

            batch_loss = torch.nn.functional.cross_entropy(
                logits.squeeze(1), y_b.long().squeeze(1), reduction="none"
            )
            total_loss += batch_loss.mean().detach()

            if args.energy_coeff > 0:
                logits = model(
                    x_src=torch.cat([X_b, X_dist_paded], dim=0)
                    / (cur_num_features / 100),
                    y_src=torch.cat([y_b, y_dist_padded], dim=0),
                    eval_pos=len(X_b),
                    normalization=False,
                    outlier_clipping=False,
                    nan_replacement=False,
                )[..., :num_classes]
                batch_energy = -logits[torch.arange(len(y_dist)), 0, y_dist.long()]
                total_energy += batch_energy.mean().detach()
            else:
                batch_energy = torch.zeros_like(batch_loss)

            (batch_loss.mean() + args.energy_coeff * batch_energy.mean()).backward()
            total_grad_norm += X_dist.grad.detach().norm(2)
            opt.step()
            opt.zero_grad()
            # scheduler.step()

        loss = total_loss / len(data["train_loader"])
        energy = total_energy / len(data["train_loader"])
        grad_norm = total_grad_norm / len(data["train_loader"])

        writer.add_scalar(
            dataset_name + "/loss/train", loss.detach().cpu().numpy(), epoch
        )
        writer.add_scalar(dataset_name + "/loss/energy", energy, epoch)
        writer.add_scalar(dataset_name + "/lr", opt.param_groups[0]["lr"], epoch)
        writer.add_scalar(dataset_name + "/grad_norm", grad_norm, epoch)


@torch.no_grad()
def eval_dist(args, model, data_loader, X_dist, y_dist):
    num_features = data_loader.dataset.tensors[0].shape[-1]

    y_pred = []
    loss = 0

    for X, y in data_loader:
        X, y = X.to(args.device), y.to(args.device)

        logits = model.predict(
            device=args.device,
            nan_replacement=None,
            normalization=False,
            outlier_clipping=False,
            return_logits=True,
            temperature=args.inf_temperature,
            test_x=X / (num_features / 100),
            train_x=X_dist / (num_features / 100),
            train_y=y_dist,
        )

        loss += (
            torch.nn.functional.cross_entropy(logits, y.long(), reduction="none")
            .detach()
            .sum()
        )
        y_pred.append(logits)

    loss = (loss / len(data_loader.dataset)).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()

    return loss, y_pred


def train_dist_knn(args, model, data, writer, experiment_path):
    X_train, X_val, y_train, y_val = (
        data["X_train"],
        data["X_valid"],
        data["y_train"],
        data["y_valid"],
    )
    dataset_name = data["dataset_info"]["name"]

    # Shuffle X_train and y_train
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    # Class balance and initialization checks
    num_classes = len(np.unique(y_train))
    num_features = X_train.shape[1]
    if any(sum(y_train == c) == 0 for c in range(num_classes)):
        raise Exception("Error: Missing class in this split...")

    # Initialization
    X_dist, y_dist = dist_initialization(args, data, num_classes, num_features)
    X_dist.requires_grad_(True)

    opt = torch.optim.AdamW([X_dist], lr=args.lr, weight_decay=args.opt_weight_decay)
    n_batches = math.ceil(len(X_train) / args.batch_size)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs * n_batches, eta_min=0)

    cache_path = os.path.join("cache", "results/tabzilla/distknn/", dataset_name)
    if args.compute_knn:
        os.makedirs(cache_path, exist_ok=True)

        faiss_knn = MulticlassFaiss(
            compute_embedding(model, data, X_train, embedding_type=args.embedding),
            X_train,
            y_train,
        )
        sizes_per_class = get_sizes_per_class(args, y_train, num_classes)

    else:
        all_X_nni, all_y_nni = [], []
        orders = []
        for f in os.listdir(cache_path):
            if not f.startswith("train_data"):
                continue
            data_saved = np.load(os.path.join(cache_path, f))
            X_nni_f, y_nni_f, idx_b_f = (
                data_saved["X_nni"],
                data_saved["y_nni"],
                data_saved["idx_b"],
            )
            all_X_nni.append(X_nni_f)
            all_y_nni.append(y_nni_f)
            orders.append(idx_b_f)
        all_X_nni = torch.Tensor(np.concatenate(all_X_nni, axis=1))
        all_y_nni = torch.Tensor(np.concatenate(all_y_nni, axis=1))
        orders = np.concatenate(orders, axis=0)
        all_X_nni = all_X_nni[:, orders.argsort(), :]
        all_y_nni = all_y_nni[:, orders.argsort()]

    best_metric, best_epoch = -np.inf, 0
    for epoch in trange(args.num_epochs):
        # Eval
        if epoch % args.eval_interval == 0:
            prev_best_metric = best_metric

            for split in ["valid", "test"]:
                loss, pred = eval_dist_knn(
                    args,
                    model,
                    data[split + "_loader"],
                    X_dist,
                    y_dist,
                    data,
                    experiment_path,
                    dataset_name,
                    split,
                )
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
                    np.save(
                        experiment_path + f"/data/{dataset_name}/" + "Xdist_best.npy",
                        X_dist.detach().cpu().numpy(),
                    )
                    np.save(
                        experiment_path + f"/data/{dataset_name}/" + "ydist_best.npy",
                        y_dist.cpu().numpy(),
                    )

                writer.add_scalar(dataset_name + "/loss/" + split, loss, epoch)
                writer.add_scalar(dataset_name + "/acc/" + split, acc, epoch)
                writer.add_scalar(dataset_name + "/f1/" + split, f1, epoch)
                writer.add_scalar(dataset_name + "/auc/" + split, auc, epoch)

            writer.flush()
            if best_epoch + args.early_stopping_rounds < epoch:
                print("Early Stopping: ", best_epoch, epoch)
                break

        # Saving
        if epoch % args.save_interval == 0:
            os.makedirs(experiment_path + f"/data/{dataset_name}/", exist_ok=True)
            np.save(
                experiment_path
                + f"/data/{dataset_name}/"
                + "Xdist_"
                + str(epoch)
                + ".npy",
                X_dist.detach().cpu().numpy(),
            )
            np.save(
                experiment_path
                + f"/data/{dataset_name}/"
                + "ydist_"
                + str(epoch)
                + ".npy",
                y_dist.cpu().numpy(),
            )

        # Training
        total_loss = 0.0
        total_energy = 0.0
        total_grad_norm = 0.0

        for b, (X_b, y_b, idx_b) in enumerate(data["train_loader"]):
            X_b, y_b = X_b.to(args.device), y_b.to(args.device)

            cur_batch_size, cur_num_features = X_b.shape[0], X_b.shape[1]

            if args.compute_knn:
                X_nni, y_nni, dist_nni, indices_X_nni = faiss_knn.get_knn(
                    compute_embedding(model, data, X_b, embedding_type=args.embedding),
                    sizes_per_class,
                )
                self_indices = dist_nni.argmin(0)

                mask = torch.ones(
                    (args.context_length, cur_batch_size), dtype=torch.bool
                )
                mask[self_indices, torch.arange(cur_batch_size)] = False
                X_nni = X_nni[mask].reshape(
                    [args.context_length - 1, cur_batch_size, cur_num_features]
                )
                y_nni = y_nni[mask].reshape([args.context_length - 1, cur_batch_size])

                np.savez(
                    cache_path + f"/train_data{b:03d}.npz",
                    X_nni=X_nni,
                    y_nni=y_nni,
                    idx_b=idx_b.numpy(),
                )
            else:
                X_nni, y_nni = all_X_nni[:, idx_b, :], all_y_nni[:, idx_b]

            X_nni, y_nni = (
                pad_x(torch.Tensor(X_nni)).to(args.device),
                torch.Tensor(y_nni).to(args.device),
            )
            X_b, y_b = (
                pad_x(X_b.unsqueeze(0)).to(args.device),
                y_b.unsqueeze(0).to(args.device),
            )

            X_dist_paded, y_dist_padded = (
                pad_x(X_dist.unsqueeze(1)).repeat_interleave(cur_batch_size, dim=1),
                y_dist.unsqueeze(1).repeat_interleave(cur_batch_size, dim=1),
            )

            if args.fix_missing:
                new_X_nni, new_y_nni = [], []
                offset = 0
                for c, size in enumerate(sizes_per_class):
                    missing_indices = np.where(indices_X_nni[c][0] == -1)[0]
                    if len(missing_indices) > 0:
                        miss_idx = missing_indices[0]
                        print("missing!!!!!!!!!!!!!!!!!!!!!", size - miss_idx)
                    else:
                        miss_idx = size
                    new_X_nni.append(X_nni[offset : offset + miss_idx])
                    new_y_nni.append(y_nni[offset : offset + miss_idx])
                    offset += size
                X_nni = torch.cat(new_X_nni, dim=0)
                y_nni = torch.cat(new_y_nni)

            logits = model(
                x_src=torch.cat([X_dist_paded, X_nni, X_b], dim=0),
                y_src=torch.cat([y_dist_padded, y_nni, y_b], dim=0),
                eval_pos=len(X_dist) + len(X_nni),
                normalization=True,
                outlier_clipping=False,
                nan_replacement=False,
                used_features=cur_num_features,
            )[..., :num_classes]

            logits = logits.squeeze(0)
            batch_loss = torch.nn.functional.cross_entropy(
                logits, y_b.long().squeeze(0), reduction="none"
            )
            total_loss += batch_loss.mean().detach()

            if args.energy_coeff > 0:
                X_b_energy, y_b_energy = (
                    X_b.squeeze(0).unsqueeze(1),
                    y_b.squeeze(0).unsqueeze(1),
                )
                X_dist_paded_energy, y_dist_padded_energy = (
                    X_dist_paded[:, 0, :].unsqueeze(1),
                    y_dist_padded[:, 0].unsqueeze(1),
                )
                logits = model(
                    x_src=torch.cat([X_b_energy, X_dist_paded_energy], dim=0)
                    / (cur_num_features / 100),
                    y_src=torch.cat([y_b_energy, y_dist_padded_energy], dim=0),
                    eval_pos=len(X_b_energy),
                    normalization=True,
                    outlier_clipping=False,
                    nan_replacement=False,
                )[..., :num_classes]
                batch_energy = -logits[
                    torch.arange(len(y_dist_padded_energy)),
                    0,
                    y_dist_padded_energy.long(),
                ]
                total_energy += batch_energy.mean().detach()
            else:
                batch_energy = torch.zeros_like(batch_loss)

            (batch_loss.mean() + args.energy_coeff * batch_energy.mean()).backward()
            total_grad_norm += X_dist.grad.detach().norm(2)
            opt.step()
            opt.zero_grad()
            # scheduler.step()

        loss = total_loss / len(data["train_loader"].dataset)
        energy = total_energy / len(data["train_loader"].dataset)
        grad_norm = total_grad_norm / len(data["train_loader"].dataset)

        writer.add_scalar(
            dataset_name + "/loss/train", loss.detach().cpu().numpy(), epoch
        )
        writer.add_scalar(dataset_name + "/loss/energy", energy, epoch)
        writer.add_scalar(dataset_name + "/lr", opt.param_groups[0]["lr"], epoch)
        writer.add_scalar(dataset_name + "/grad_norm", grad_norm, epoch)


# TODO: need to store the neighbors so they don't have to be recomputed every time
@torch.no_grad()
def eval_dist_knn(
    args,
    model,
    data_loader,
    X_dist,
    y_dist,
    data,
    experiment_path,
    dataset_name,
    val_or_test,
):
    num_features = data_loader.dataset.tensors[0].shape[-1]
    num_classes = len(np.unique(data_loader.dataset.tensors[1]))
    X_train, y_train = data["X_train"], data["y_train"]

    y_pred = []
    loss = 0

    cache_path = os.path.join("cache", "results/tabzilla/distknn/", dataset_name)
    if args.compute_knn:
        os.makedirs(cache_path, exist_ok=True)

        faiss_knn = MulticlassFaiss(
            compute_embedding(model, data, X_train, embedding_type=args.embedding),
            X_train,
            y_train,
        )
        sizes_per_class = get_sizes_per_class(args, y_train, num_classes)

    else:
        all_X_nni, all_y_nni = [], []
        orders = []
        for f in os.listdir(cache_path):
            if not f.startswith(val_or_test + "_data"):
                continue
            data_saved = np.load(os.path.join(cache_path, f))
            X_nni_f, y_nni_f, idx_b_f = (
                data_saved["X_nni"],
                data_saved["y_nni"],
                data_saved["idx_b"],
            )
            orders.append(idx_b_f)
            all_X_nni.append(X_nni_f)
            all_y_nni.append(y_nni_f)
        all_X_nni = torch.Tensor(np.concatenate(all_X_nni, axis=1))
        all_y_nni = torch.Tensor(np.concatenate(all_y_nni, axis=1))
        orders = np.concatenate(orders, axis=0)
        all_X_nni = all_X_nni[:, orders.argsort(), :]
        all_y_nni = all_y_nni[:, orders.argsort()]

    faiss_knn = MulticlassFaiss(
        compute_embedding(model, data, X_train, embedding_type=args.embedding),
        X_train,
        y_train,
    )
    sizes_per_class = get_sizes_per_class(args, y_train, num_classes)

    for b, (X, y, idx_b) in enumerate(data_loader):
        cur_batch_size = X.shape[0]
        X, y = X.to(args.device), y.to(args.device)

        if args.compute_knn:
            indices_X_nni, y_nni = faiss_knn.get_knn_indices(
                compute_embedding(model, data, X, embedding_type=args.embedding),
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

            np.savez(
                cache_path + f"/{val_or_test}_data{b:03d}.npz",
                X_nni=X_nni,
                y_nni=y_nni,
                idx_b=idx_b.numpy(),
            )
        else:
            X_nni, y_nni = all_X_nni[:, idx_b, :], all_y_nni[:, idx_b]

        X_nni, y_nni = (
            pad_x(torch.Tensor(X_nni)).to(args.device),
            torch.Tensor(y_nni).to(args.device),
        )
        X, y = pad_x(X.unsqueeze(0)).to(args.device), y.unsqueeze(0).to(args.device)

        if args.fix_missing:
            new_X_nni, new_y_nni = [], []
            offset = 0
            for c, size in enumerate(sizes_per_class):
                missing_indices = np.where(indices_X_nni[c][0] == -1)[0]
                if len(missing_indices) > 0:
                    miss_idx = missing_indices[0]
                    print("missing!!!!!!!!!!!!!!!!!!!!!", size - miss_idx)
                else:
                    miss_idx = size
                new_X_nni.append(X_nni[offset : offset + miss_idx])
                new_y_nni.append(y_nni[offset : offset + miss_idx])
                offset += size
            X_nni = torch.cat(new_X_nni, dim=0)
            y_nni = torch.cat(new_y_nni)

        X_dist_paded, y_dist_padded = (
            pad_x(X_dist.unsqueeze(1)).repeat_interleave(cur_batch_size, dim=1),
            y_dist.unsqueeze(1).repeat_interleave(cur_batch_size, dim=1),
        )

        logits = model(
            x_src=torch.cat([X_dist_paded, X_nni, X], dim=0),
            y_src=torch.cat([y_dist_padded, y_nni, y], dim=0),
            eval_pos=len(X_dist) + len(X_nni),
            normalization=True,
            outlier_clipping=False,
            nan_replacement=False,
            used_features=num_features,
        )[..., :num_classes]

        logits = logits.squeeze(0)
        loss += (
            torch.nn.functional.cross_entropy(
                logits, y.long().squeeze(0), reduction="none"
            )
            .detach()
            .sum()
        )
        y_pred.append(logits)

    loss = (loss / len(data_loader.dataset)).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()

    return loss, y_pred

