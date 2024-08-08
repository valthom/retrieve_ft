from tqdm import tqdm
import numpy as np
import torch
import time

from utils import MulticlassFaiss, get_sizes_per_class, pad_x, SingleclassFaiss


@torch.no_grad()
def compute_embedding(
    model,
    data,
    X,
    y=None,
    is_context=False,
    embedding_type="raw",
    device="cuda:0",
    embedding_layer=None,
):
    if embedding_type == "raw":
        return X
    elif embedding_type == "logits":
        num_features = X.shape[-1]
        num_classes = len(np.unique(data["y_train"]))
        sample_size = 1000

        sample_indices = np.random.choice(
            data["X_train"].shape[0], sample_size, replace=False
        )
        X_train_sampled = data["X_train"][sample_indices, :]
        y_train_sampled = (
            torch.Tensor(data["y_train"][sample_indices]).unsqueeze(1).to(device)
        )
        X_train_sampled, X = (
            pad_x(torch.Tensor(X_train_sampled).unsqueeze(1)).to(device),
            pad_x(torch.Tensor(X).unsqueeze(1)).to(device),
        )

        logits = model(
            x_src=torch.cat([X_train_sampled, X], dim=0),
            y_src=torch.cat(
                [y_train_sampled, torch.zeros((len(X), 1)).to(device)], dim=0
            ),
            eval_pos=len(X_train_sampled),
            normalization=True,
            outlier_clipping=False,
            nan_replacement=False,
            used_features=num_features,
        )
        embedding = logits.squeeze(1)[:, :num_classes].cpu().numpy()
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding
    elif embedding_type == "layer1":
        X_padded = pad_x(torch.Tensor(X).unsqueeze(1))
        if embedding_layer:
            embedding = embedding_layer(X_padded).squeeze(1).cpu().numpy()
        else:
            embedding = model.encoder(X_padded).squeeze(1).cpu().numpy()
        if y is not None:
            y_embedding = model.y_encoder(torch.Tensor(y).unsqueeze(1).to(device))
            embedding += y_embedding.squeeze(1).cpu().numpy()
        # embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding
    elif embedding_type == "layer2":
        X_padded = pad_x(torch.Tensor(X).unsqueeze(1)).to(device)
        embedding = model.encoder(X_padded)
        w_q, w_k, w_v = model.transformer_encoder[0].self_attn.in_proj_weight.chunk(3)
        b_q, b_k, b_v = model.transformer_encoder[0].self_attn.in_proj_bias.chunk(3)

        # TODO need to be fixed
        if not is_context:
            embedding = torch.nn.functional.linear(embedding, w_q, b_q)
            # embedding = torch.stack(embedding.chunk(4, -1), -1)
        else:
            embedding = torch.nn.functional.linear(embedding, w_k, b_k)
            # embedding = torch.stack(embedding.chunk(4, -1), -1)
        embedding = embedding.squeeze(1).cpu().numpy()
        # embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding
    else:
        raise Exception("Error: Invalid embedding type...")


@torch.no_grad()
def eval_pfknn(args, model, data, data_loader, normalize_data=True):
    X_train = data["X_train_one_hot"] if args.use_one_hot_emb else data["X_train"]
    y_train = data["y_train"]
    X_train_one_hot = data["X_train_one_hot"]

    # checks
    num_classes = len(np.unique(y_train))
    if any(sum(y_train == c) == 0 for c in range(num_classes)):
        raise Exception("Error: Missing class in this split...")

    # total_time = 0

    if args.knn_mode == "multiclass":
        print("Using multiclass knn")
        # faiss_knn = MulticlassFaiss(X_train, X_train, y_train)
        faiss_knn = MulticlassFaiss(
            compute_embedding(model, data, X_train_one_hot if args.onehot_retrieval else X_train, embedding_type=args.embedding),
            X_train,
            y_train,
        )
    else:
        print("Using singleclass knn")
        # faiss_knn = SingleclassFaiss(X_train, y_train)
        faiss_knn = SingleclassFaiss(
            compute_embedding(model, data, X_train_one_hot if args.onehot_retrieval else X_train, embedding_type=args.embedding),
            y_train,
        )
    sizes_per_class = get_sizes_per_class(args, y_train, num_classes)

    # eval
    y_pred = []
    loss = 0

    for X, y, X_oh in tqdm(data_loader):
        cur_batch_size, cur_num_features = X.shape[0], X.shape[1]

        # t0 = time.time()
        if args.knn_mode == "multiclass":
            # indices_X_nni, y_nni = faiss_knn.get_knn_indices(X, sizes_per_class)
            indices_X_nni, y_nni = faiss_knn.get_knn_indices(
                compute_embedding(model, data, X_oh if args.onehot_retrieval else X.numpy(), embedding_type=args.embedding),
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
                compute_embedding(model, data, X_oh if args.onehot_retrieval else X.numpy(), embedding_type=args.embedding),
                sum(sizes_per_class),
            )
            X_nni = X_train[indices_X_nni]
            X_nni = np.swapaxes(X_nni, 0, 1)
            y_nni = np.swapaxes(y_nni, 0, 1)

        # total_time += time.time() - t0

        X_nni, y_nni = (
            pad_x(torch.Tensor(X_nni)).to(args.device),
            torch.Tensor(y_nni).to(args.device),
        )
        X, y = pad_x(X.unsqueeze(0)).to(args.device), y.unsqueeze(0).to(args.device)

        if args.fix_missing:
            if args.knn_mode == "singleclass":
                size = sum(sizes_per_class)
                missing_indices = np.where(indices_X_nni[0] == -1)[0]
                if len(missing_indices) > 0:
                    miss_idx = missing_indices[0]
                    X_nni = X_nni[:miss_idx]
                    y_nni = y_nni[:miss_idx]
                    print("missing!!!!!!!!!!!!!!!!!!!!!", size - miss_idx)
            else:
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

        # t2 = time.time()
        logits = model(
            x_src=torch.cat([X_nni, X], dim=0),
            y_src=torch.cat([y_nni, y], dim=0),
            eval_pos=len(X_nni),
            normalization=normalize_data,
            outlier_clipping=False,
            nan_replacement=False,
            used_features=cur_num_features,
        )

        # total_time += time.time() - t2

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
            normalization=False,
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
