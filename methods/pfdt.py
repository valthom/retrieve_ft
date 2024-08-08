import numpy as np
import torch
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier


@torch.no_grad()
def eval_pfdt(args, model, data, data_loader):
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]

    # checks
    num_classes = len(np.unique(y_train))
    if any(sum(y_train == c) == 0 for c in range(num_classes)):
        raise Exception("Error: Missing class in this split...")

    num_features = X_train.shape[1]

    clf = DecisionTreeClassifier(random_state=42, max_depth=3)
    clf.fit(X_train, y_train)

    # y_pred = clf.predict_proba(X_test)

    leaf_indices_train = clf.apply(X_train)
    leaf_to_train_samples = defaultdict(list)
    leaf_to_train_labels = defaultdict(list)
    for sample_index, leaf_index in enumerate(leaf_indices_train):
        leaf_to_train_samples[leaf_index].append(X_train[sample_index])
        leaf_to_train_labels[leaf_index].append(y_train[sample_index])

    leaf_indices_test = clf.apply(X_test)
    leaf_to_test_samples = defaultdict(list)
    for sample_index, leaf_index in enumerate(leaf_indices_test):
        leaf_to_test_samples[leaf_index].append(X_test[sample_index])

    y_pred = []
    for l in leaf_to_test_samples.keys():
        X_bl, y_bl = (
            np.array(leaf_to_train_samples[l]),
            np.array(leaf_to_train_labels[l]),
        )

        if len(X_bl) > 2000:
            indices = np.random.choice(len(X_bl), 2000)
            X_bl, y_bl = X_bl[indices], y_bl[indices]

        X_t = np.array(leaf_to_test_samples[l])

        X_bl, y_bl = (
            torch.Tensor(X_bl).to(args.device),
            torch.Tensor(y_bl).to(args.device),
        )
        X_bl_paded, y_bl_padded = pad_x(X_bl.unsqueeze(1)), y_bl.unsqueeze(1)
        X_t = pad_x(torch.Tensor(X_t).unsqueeze(1)).to(args.device)
        y_t = torch.zeros((len(X_t), 1)).to(args.device)

        logits = model(
            x_src=torch.cat([X_bl_paded, X_t], dim=0),
            y_src=torch.cat([y_bl_padded, y_t], dim=0),
            eval_pos=len(X_bl),
            normalization=True,
            outlier_clipping=False,
            nan_replacement=False,
            used_features=num_features,
        )[..., :num_classes]

        pred_i = []
        for c in range(num_classes):
            pred_i.append(np.mean(np.array(leaf_to_train_labels[l]) == c))

        if len(set(list(y_bl.cpu().numpy()))) != num_classes:
            y_temp = []
            for _ in range(len(X_t)):
                y_temp.append(pred_i)
            y_temp = np.array(y_temp)

            y_pred.append(y_temp)
        else:
            print("here")
            y_pred.append(logits.squeeze(1).detach().cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)

    return 0.0, y_pred
