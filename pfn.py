import torch
import numpy as np
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F


def maskmean(x, mask, dim):
    x = torch.where(mask, x, 0)
    return x.sum(dim=dim, keepdim=True) / mask.sum(dim=dim, keepdim=True)


def maskstd(x, mask, dim=0):
    num = mask.sum(dim=dim, keepdim=True)
    mean = maskmean(x, mask, dim=0)
    diffs = torch.where(mask, mean - x, 0)
    return ((diffs**2).sum(dim=0, keepdim=True) / (num - 1)) ** 0.5


def normalize_data(data, eval_pos):
    mask = ~torch.isnan(data[:eval_pos])
    mean = maskmean(data[:eval_pos], mask, dim=0)
    std = maskstd(data[:eval_pos], mask, dim=0) + 1e-6
    data = (data - mean) / std
    return torch.clip(data, min=-100, max=100)


def clip_outliers(X, eval_pos, n_sigma=4):
    assert len(X.shape) == 3, "X must be T,B,H"
    mask = ~torch.isnan(X[:eval_pos])
    mean = maskmean(X[:eval_pos], mask, dim=0)
    cutoff = n_sigma * maskstd(X[:eval_pos], ~torch.isnan(X[:eval_pos]), dim=0)
    mask = mask & (X[:eval_pos] >= mean - cutoff) & (X[:eval_pos] <= mean + cutoff)
    cutoff = n_sigma * maskstd(X[:eval_pos], ~torch.isnan(X[:eval_pos]), dim=0)
    return torch.clip(X, mean - cutoff, mean + cutoff)


def convert_to_torch_tensor(input):
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    elif torch.is_tensor(input):
        return input
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")


class PFN(nn.Module):
    def __init__(
        self,
        dropout,
        embedding_normalization,
        n_out,
        nhead,
        nhid,
        ninp,
        nlayers,
        norm_first,
        num_features,
    ):
        super().__init__()
        self.n_out = n_out
        self.embedding_normalization = embedding_normalization
        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    activation="gelu",
                    d_model=ninp,
                    dim_feedforward=nhid,
                    dropout=dropout,
                    nhead=nhead,
                    norm_first=norm_first,
                )
                for _ in range(nlayers)
            ]
        )
        self.num_features = num_features
        self.encoder = nn.Linear(num_features, ninp)
        self.y_encoder = nn.Linear(1, ninp)
        self.decoder = nn.Sequential(
            nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, n_out)
        )

    def forward(
        self,
        x_src,
        y_src,
        eval_pos,
        normalization,
        outlier_clipping,
        nan_replacement,
        used_features=None,
    ):
        if normalization:
            x_src = normalize_data(x_src, eval_pos)
        if outlier_clipping:
            x_src = clip_outliers(x_src, eval_pos, n_sigma=4)
        # check if this is after or before the clip outliers
        if used_features is not None:
            x_src = x_src / (used_features / 100)
        if nan_replacement is not None:
            x_src = torch.nan_to_num(x_src, nan=nan_replacement)
        x_src = self.encoder(x_src)
        if self.embedding_normalization:
            x_src = F.normalize(x_src, p=2, dim=-1)
        y_src = self.y_encoder(y_src.unsqueeze(-1))
        train_x = x_src[:eval_pos] + y_src[:eval_pos]
        src = torch.cat([train_x, x_src[eval_pos:]], 0)
        condition = torch.arange(src.shape[0]).to(src.device) >= eval_pos
        attention_mask = condition.repeat(src.shape[0], 1)
        for layer in self.transformer_encoder:
            src = layer(src, attention_mask)
        src = self.decoder(src)
        return src[eval_pos:]

    def predict(
        self,
        device,
        nan_replacement,
        normalization,
        outlier_clipping,
        return_logits,
        temperature,
        test_x,
        train_x,
        train_y,
    ):
        to_numpy = not torch.is_tensor(train_x)
        train_x, train_y, test_x = (
            convert_to_torch_tensor(train_x).to(device).float(),
            convert_to_torch_tensor(train_y).to(device).float(),
            convert_to_torch_tensor(test_x).to(device).float(),
        )
        dim = train_x.dim()
        if dim == 2:
            train_x = train_x.unsqueeze(1)
            test_x = test_x.unsqueeze(1)
            train_y = train_y.unsqueeze(1)
        else:
            assert False, "Predict only supports one dataset at a time"

        seq_len, batch_size, n_features = train_x.shape
        num_classes = len(torch.unique(train_y))
        test_seq_len = test_x.shape[0]
        full_x = torch.cat((train_x, test_x), dim=0)
        zero_feature_padding = torch.zeros(
            (
                seq_len + test_seq_len,
                batch_size,
                self.num_features - n_features,
            )
        ).to(device)
        full_x = torch.cat([full_x, zero_feature_padding], -1)

        y_padding = torch.zeros((test_seq_len, batch_size)).to(device)
        full_y = torch.cat([train_y, y_padding], dim=0)

        # forward
        output = self.forward(
            eval_pos=seq_len,
            nan_replacement=nan_replacement,
            normalization=normalization,
            outlier_clipping=outlier_clipping,
            x_src=full_x,
            y_src=full_y,
        )
        output = output[..., :num_classes] / temperature
        if not return_logits:
            output = torch.nn.functional.softmax(output, dim=-1)
        if dim == 2:
            output.squeeze_(1)
        if to_numpy:
            output = output.detach().cpu().numpy()
        return output

    @classmethod
    def load(cls, path, device):
        model_state, config = torch.load(path, map_location=device)
        assert config["max_num_classes"] > 2
        model = PFN(
            dropout=config["dropout"],
            embedding_normalization=config["embedding_normalization"],
            n_out=config["max_num_classes"],
            nhead=config["nhead"],
            nhid=config["emsize"] * config["nhid_factor"],
            ninp=config["emsize"],
            nlayers=config["nlayers"],
            norm_first=config["norm_first"],
            num_features=config["max_num_features"],
        )

        module_prefix = "module."
        model_state = {k.replace(module_prefix, ""): v for k, v in model_state.items()}
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        return model, config

    @classmethod
    def load_old(cls, path, device):
        model_state, _, config = torch.load(path, map_location=device)
        assert config["max_num_classes"] > 2
        model = PFN(
            dropout=config["dropout"],
            embedding_normalization=False,
            n_out=config["max_num_classes"],
            nhead=config["nhead"],
            nhid=config["emsize"] * config["nhid_factor"],
            ninp=config["emsize"],
            nlayers=config["nlayers"],
            norm_first=False,
            num_features=config["num_features"],
        )

        module_prefix = "module."
        model_state = {
            k.replace(module_prefix, "").replace("layers.", ""): v
            for k, v in model_state.items()
        }
        del model_state["criterion.weight"]
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        return model, config

