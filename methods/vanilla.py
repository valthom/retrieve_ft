from tqdm import tqdm
import numpy as np
import torch

from utils import MulticlassFaiss, get_sizes_per_class, pad_x


def random_initialization(args, data, num_classes, num_features):
    X_train, y_train = data["X_train"], data["y_train"]

    sizes_per_class = get_sizes_per_class(args, y_train, num_classes)

    # Create y_dist
    y_init = torch.Tensor(
        np.concatenate([np.array([c] * size) for c, size in enumerate(sizes_per_class)])
    ).to(args.device)

    # Sample with replacement when using small datasets
    # sample_replacement = len(X_train) < args.context_length
    sample_replacement = False
    for c, size in enumerate(sizes_per_class):
        if size > sum(y_train == c):
            sample_replacement = True
            break

    X_init = torch.Tensor(
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
    return X_init, y_init


@torch.no_grad()
def eval_tabpfn(args, model, data, data_loader):
    X_train = data["X_train_one_hot"] if args.use_one_hot_emb else data["X_train"]
    y_train = data["y_train"]

    # checks
    num_classes = len(np.unique(y_train))
    if any(sum(y_train == c) == 0 for c in range(num_classes)):
        raise Exception("Error: Missing class in this split...")

    sizes_per_class = get_sizes_per_class(args, y_train, num_classes)
    X_init, y_init = random_initialization(args, data, num_classes, X_train.shape[1])

    # eval
    y_pred = []
    loss = 0

    for X, y in tqdm(data_loader):
        cur_batch_size, cur_num_features = X.shape[0], X.shape[1]

        # X_nni = np.swapaxes(X_nni, 0, 1)
        X_ctx = X_init.unsqueeze(0).repeat(cur_batch_size, 1, 1)
        X_ctx = np.swapaxes(X_ctx, 0, 1)
        y_ctx = y_init.unsqueeze(0).repeat(cur_batch_size, 1)
        y_ctx = np.swapaxes(y_ctx, 0, 1)
        X_ctx, y_ctx = pad_x(X_ctx).to(args.device), y_ctx.to(args.device)
        # y_nni = y_nni.reshape(-1, 1).repeat(cur_batch_size, axis=1)

        # X_nni, y_nni = pad_x(torch.Tensor(X_nni)).to(args.device), torch.Tensor(y_nni).to(args.device)
        X, y = pad_x(X.unsqueeze(0)).to(args.device), y.unsqueeze(0).to(args.device)

        logits = model(
            x_src=torch.cat([X_ctx, X], dim=0) / (cur_num_features / 100),
            y_src=torch.cat([y_ctx, y], dim=0),
            eval_pos=len(X_ctx),
            normalization=True,
            outlier_clipping=False,
            nan_replacement=False,
        )

        logits = logits.squeeze(0)[:, :num_classes]
        loss += (
            torch.nn.functional.cross_entropy(
                logits, y.long().squeeze(0), reduction="none"
            )
            .detach()
            .sum()
        )
        y_pred.append(logits)

    loss = (loss / len(data_loader)).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()

    return loss, y_pred

@torch.no_grad()
def eval_tabpfn_original(args, data, data_loader):
    from tabpfn import TabPFNClassifier

    classifier = TabPFNClassifier(device="cuda:0", N_ensemble_configurations=32)

    X_train = data["X_train_one_hot"] if args.use_one_hot_emb else data["X_train"]
    y_train = data["y_train"]

    # checks
    num_classes = len(np.unique(y_train))
    if any(sum(y_train == c) == 0 for c in range(num_classes)):
        raise Exception("Error: Missing class in this split...")

    sizes_per_class = get_sizes_per_class(args, y_train, num_classes)
    X_init, y_init = random_initialization(args, data, num_classes, X_train.shape[1])

    # eval
    y_pred = []
    loss = 0

    for X, y in tqdm(data_loader):
        cur_batch_size, cur_num_features = X.shape[0], X.shape[1]

        # X_nni = np.swapaxes(X_nni, 0, 1)
        X_ctx = X_init.unsqueeze(0).repeat(cur_batch_size, 1, 1)
        X_ctx = np.swapaxes(X_ctx, 0, 1)
        y_ctx = y_init.unsqueeze(0).repeat(cur_batch_size, 1)
        y_ctx = np.swapaxes(y_ctx, 0, 1)
        X_ctx, y_ctx = pad_x(X_ctx).to(args.device), y_ctx.to(args.device)
        # y_nni = y_nni.reshape(-1, 1).repeat(cur_batch_size, axis=1)

        # X_nni, y_nni = pad_x(torch.Tensor(X_nni)).to(args.device), torch.Tensor(y_nni).to(args.device)
        X, y = pad_x(X.unsqueeze(0)).to(args.device), y.unsqueeze(0).to(args.device)

        logits = model(
            x_src=torch.cat([X_ctx, X], dim=0) / (cur_num_features / 100),
            y_src=torch.cat([y_ctx, y], dim=0),
            eval_pos=len(X_ctx),
            normalization=True,
            outlier_clipping=False,
            nan_replacement=False,
        )

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict_proba(X_test)

        logits = logits.squeeze(0)[:, :num_classes]
        loss += (
            torch.nn.functional.cross_entropy(
                logits, y.long().squeeze(0), reduction="none"
            )
            .detach()
            .sum()
        )
        y_pred.append(logits)

    loss = (loss / len(data_loader)).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()

    return loss, y_pred


@torch.no_grad()
def eval_pfknn_ensemble_dist(args, model, data, data_loader, pred_folder=""):
    X_train = data["X_train_one_hot"] if args.use_one_hot_emb else data["X_train"]
    y_train = data["y_train"]

    # checks
    num_classes = len(np.unique(y_train))
    if any(sum(y_train == c) == 0 for c in range(num_classes)):
        raise Exception("Error: Missing class in this split...")

    faiss_knn = MulticlassFaiss(X_train, y_train)
    sizes_per_class = get_sizes_per_class(args, y_train, num_classes)

    # eval
    y_pred = []
    loss = 0

    for X, y in tqdm(data_loader):
        cur_batch_size, cur_num_features = X.shape[0], X.shape[1]

        indices_X_nni, y_nni = faiss_knn.get_knn_indices(X, sizes_per_class)
        X_nni = np.concatenate(
            [X_train[y_train == i][indices] for i, indices in enumerate(indices_X_nni)],
            axis=1,
        )
        X_nni = np.swapaxes(X_nni, 0, 1)
        y_nni = y_nni.reshape(-1, 1).repeat(cur_batch_size, axis=1)

        X_nni, y_nni = (
            pad_x(torch.Tensor(X_nni)).to(args.device),
            torch.Tensor(y_nni).to(args.device),
        )
        X, y = pad_x(X.unsqueeze(0)).to(args.device), y.unsqueeze(0).to(args.device)

        dataset_name = data["dataset_info"]["name"]
        X_dist = torch.Tensor(np.load(f"{pred_folder}/Xdist_best.npy")).to(args.device)
        y_dist = torch.Tensor(np.load(f"{pred_folder}/ydist_best.npy")).to(args.device)
        X_dist = X_dist.unsqueeze(1).repeat(1, cur_batch_size, 1)
        y_dist = y_dist.unsqueeze(1).repeat(1, cur_batch_size)
        X_dist = pad_x(X_dist)

        logits = model(
            x_src=torch.cat([X_dist, X_nni, X], dim=0) / (cur_num_features / 100),
            y_src=torch.cat([y_dist, y_nni, y], dim=0),
            eval_pos=len(X_dist) + len(X_nni),
            normalization=True,
            outlier_clipping=False,
            nan_replacement=False,
        )

        logits = logits.squeeze(0)[:, :num_classes]
        loss += (
            torch.nn.functional.cross_entropy(
                logits, y.long().squeeze(0), reduction="none"
            )
            .detach()
            .sum()
        )
        y_pred.append(logits)

    loss = (loss / len(data_loader)).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()

    return loss, y_pred

