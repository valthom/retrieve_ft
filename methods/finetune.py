import math
import os
import scipy

import numpy as np
import torch

from utils import get_sizes_per_class, compute_metrics, pad_x


def data_initialization(args, data, num_classes, num_features=100):
    X_train, y_train = data["X_train"], data["y_train"]

    sizes_per_class = get_sizes_per_class(args, y_train, num_classes)

    # Create y_ctx
    y_ctx = torch.Tensor(
        np.concatenate([np.array([c] * size) for c, size in enumerate(sizes_per_class)])
    ).to(args.device)

    # Initialize X_ctx

    # Sample with replacement when using small datasets
    # sample_replacement = len(X_train) < args.context_length
    sample_replacement = False
    for c, size in enumerate(sizes_per_class):
        if size > sum(y_train == c):
            sample_replacement = True
            break

    X_ctx = torch.Tensor(
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
    return X_ctx, y_ctx


def train_ft(args, model, data, writer, experiment_path):
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
    X_ctx_init, y_ctx_init = data_initialization(args, data, num_classes, num_features)
    X_ctx, y_ctx = X_ctx_init.clone(), y_ctx_init.clone()

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.opt_weight_decay
    )
    # opt = torch.optim.AdamW([model.encoder.parmae()], lr=args.lr, weight_decay=args.opt_weight_decay)
    n_batches = math.ceil(len(X_train) / args.batch_size)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs * n_batches, eta_min=0)

    best_metric, best_epoch = -np.inf, 0
    for epoch in trange(args.num_epochs):
        # Eval
        if epoch % args.eval_interval == 0:
            prev_best_metric = best_metric

            for split in ["valid", "test"]:
                loss, pred = eval_ft(
                    args, model, data[split + "_loader"], X_ctx_init, y_ctx_init, data
                )
                # loss, pred = eval_pfknn(args, model, data, data[split+"_loader"])
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
                    # best_epoch = epoch
                    best_metric = metric
                    os.makedirs(
                        experiment_path + f"/data/{dataset_name}/", exist_ok=True
                    )
                    # np.save(experiment_path + f"/data/{dataset_name}/" + "Xdist_best.npy", X_ctx_init.detach().cpu().numpy())
                    # np.save(experiment_path + f"/data/{dataset_name}/" + "ydist_best.npy", y_ctx_init.cpu().numpy())
                    torch.save(
                        model.state_dict(),
                        experiment_path + f"/data/{dataset_name}/model_best.pth",
                    )

                writer.add_scalar(dataset_name + "/loss/" + split, loss, epoch)
                writer.add_scalar(dataset_name + "/acc/" + split, acc, epoch)
                writer.add_scalar(dataset_name + "/f1/" + split, f1, epoch)
                writer.add_scalar(dataset_name + "/auc/" + split, auc, epoch)

            writer.flush()
            # if best_epoch + args.early_stopping_rounds < epoch:
            #     print("Early Stopping: ", best_epoch, epoch)
            #     break

        # Saving
        # if epoch % args.save_interval == 0:
        #     os.makedirs(experiment_path + f"/data/{dataset_name}/", exist_ok=True)
        #     np.save(experiment_path + f"/data/{dataset_name}/" + "Xdist_" + str(epoch) + ".npy", X_ctx.detach().cpu().numpy())
        #     np.save(experiment_path + f"/data/{dataset_name}/" + "ydist_" + str(epoch) + ".npy", y_ctx.cpu().numpy())

        # Training
        total_loss = 0.0

        for X_b, y_b in data["train_loader"]:
            X_ctx, y_ctx = data_initialization(args, data, num_classes, num_features)
            X_b, y_b = X_b.to(args.device), y_b.to(args.device)

            cur_num_features = X_b.shape[1]

            X_b, y_b = (
                pad_x(X_b.unsqueeze(1)).to(args.device),
                y_b.unsqueeze(1).to(args.device),
            )
            X_dist_paded, y_dist_padded = pad_x(X_ctx.unsqueeze(1)), y_ctx.unsqueeze(1)

            logits = model(
                x_src=torch.cat([X_dist_paded, X_b], dim=0),
                y_src=torch.cat([y_dist_padded, y_b], dim=0),
                eval_pos=len(X_ctx),
                normalization=False,
                outlier_clipping=False,
                nan_replacement=False,
                used_features=cur_num_features,
            )[..., :num_classes]

            batch_loss = torch.nn.functional.cross_entropy(
                logits.squeeze(1), y_b.long().squeeze(1), reduction="none"
            )
            total_loss += batch_loss.sum().detach()

            batch_loss.mean().backward()
            opt.step()
            opt.zero_grad()
            # scheduler.step()

        loss = total_loss / len(data["train_loader"].dataset)

        writer.add_scalar(
            dataset_name + "/loss/train", loss.detach().cpu().numpy(), epoch
        )
        writer.add_scalar(dataset_name + "/lr", opt.param_groups[0]["lr"], epoch)
    return X_ctx_init, y_ctx_init


@torch.no_grad()
def eval_ft(args, model, data_loader, X_ctx, y_ctx, data):
    num_features = data_loader.dataset.tensors[0].shape[-1]
    num_classes = len(np.unique(data_loader.dataset.tensors[1]))

    # all_y_pred = []
    # for _ in range(5):
    #     y_pred = []
    #     loss = 0

    X_ctx, y_ctx = data_initialization(args, data, num_classes, num_features)

    #     for X, y in data_loader:
    #         X, y = X.to(args.device), y.to(args.device)

    #         logits = model.predict(device=args.device, nan_replacement=None, normalization=False,
    #                             outlier_clipping=False, return_logits=True, temperature=args.inf_temperature,
    #                             test_x=X / (num_features / 100), train_x=X_ctx / (num_features / 100), train_y=y_ctx)

    #         loss += torch.nn.functional.cross_entropy(logits, y.long(), reduction='none').detach().sum()
    #         y_pred.append(logits)

    #     loss = (loss / len(data_loader.dataset)).cpu().numpy()
    #     y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()

    #     all_y_pred.append(y_pred)

    # y_pred = np.stack(all_y_pred, axis=0).mean(0)

    y_pred = []
    loss = 0

    X_ctx, y_ctx = data_initialization(args, data, num_classes, num_features)

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
            train_x=X_ctx / (num_features / 100),
            train_y=y_ctx,
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
