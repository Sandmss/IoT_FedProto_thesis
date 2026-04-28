#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d, MaxPool1d


def _flatten_batch(x):
    return x.view(x.size(0), -1)


class _FeatureOnlyModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.extract_features(x)


class BaseHeadSplit(nn.Module):
    def __init__(self, args, cid):
        super().__init__()

        if hasattr(args, "model_builders"):
            raw_model = args.model_builders[cid % len(args.model_builders)]()
        else:
            raw_model = eval(args.models[cid % len(args.models)])
        if not hasattr(raw_model, "extract_features"):
            raise NotImplementedError(
                f"Unsupported tabular model for BaseHeadSplit: {raw_model.__class__.__name__}"
            )

        self.base = _FeatureOnlyModel(raw_model)

        if hasattr(args, "head_builders"):
            self.head = args.head_builders[cid % len(args.head_builders)]()
        elif hasattr(args, 'heads'):
            self.head = eval(args.heads[cid % len(args.heads)])
        elif hasattr(raw_model, 'classifier'):
            self.head = raw_model.classifier
        elif hasattr(raw_model, 'layer_hidden'):
            self.head = raw_model.layer_hidden
        elif hasattr(raw_model, 'fc'):
            self.head = raw_model.fc
        else:
            self.head = nn.Linear(args.feature_dim, args.num_classes)

    def forward(self, x):
        rep = self.base(x)
        out = self.head(rep)
        return out


class Head(nn.Module):
    def __init__(self, num_classes=10, hidden_dims=None):
        super().__init__()
        dims = list(hidden_dims or [512]) + [num_classes]

        layers = []
        for idx in range(1, len(dims)):
            layers.append(nn.Linear(dims[idx - 1], dims[idx]))
            if idx < len(dims) - 1:
                layers.append(nn.ReLU(inplace=True))

        self.fc = nn.Sequential(*layers)

    def forward(self, rep):
        return self.fc(rep)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def extract_features(self, x):
        x = _flatten_batch(x)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class MLP_IoT(nn.Module):
    def __init__(self, dim_in=77, dim_hidden=128, dim_out=64, num_classes=15):
        super(MLP_IoT, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.classifier = nn.Linear(dim_out, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def extract_features(self, x):
        x = _flatten_batch(x)
        x = self.relu(self.layer_input(x))
        x = self.dropout(x)
        x = self.relu(self.layer_hidden(x))
        return x

    def forward(self, x):
        x1 = self.extract_features(x)
        x = self.classifier(x1)
        return F.log_softmax(x, dim=1), x1


class CNN1D_IoT(nn.Module):
    def __init__(self, dim_in=77, dim_out=64, num_classes=15):
        super(CNN1D_IoT, self).__init__()
        self.conv1 = Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * (dim_in // 4), 128)
        self.fc_proto = nn.Linear(128, dim_out)
        self.classifier = nn.Linear(dim_out, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def extract_features(self, x):
        x = _flatten_batch(x)
        x = x.unsqueeze(1)
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = _flatten_batch(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc_proto(x))
        return x

    def forward(self, x):
        x1 = self.extract_features(x)
        x = self.classifier(x1)
        return F.log_softmax(x, dim=1), x1


class Transformer1D_IoT(nn.Module):
    def __init__(
        self,
        dim_in=77,
        dim_out=64,
        num_classes=15,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dropout=0.2,
        dim_feedforward=None,
    ):
        super(Transformer1D_IoT, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.dim_in = dim_in
        self.token_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, dim_in, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward or d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.projector = nn.Sequential(
            nn.Linear(d_model, dim_out),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(dim_out, num_classes)

    def extract_features(self, x):
        x = _flatten_batch(x)
        if x.size(1) != self.dim_in:
            raise ValueError(f"Expected input dimension {self.dim_in}, got {x.size(1)}")
        x = x.unsqueeze(-1)
        x = self.token_embed(x) + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.projector(x)

    def forward(self, x):
        x1 = self.extract_features(x)
        x = self.classifier(x1)
        return F.log_softmax(x, dim=1), x1


class FedAvgMLP(nn.Module):
    def __init__(self, in_features=77, num_classes=10, hidden_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU(inplace=True)

    def extract_features(self, x):
        x = _flatten_batch(x)
        x = self.act(self.fc1(x))
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.fc2(x)
        return x


class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim=77, num_classes=10):
        super(Mclr_Logistic, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def extract_features(self, x):
        return _flatten_batch(x)

    def forward(self, x):
        x = self.extract_features(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


class DNN(nn.Module):
    def __init__(self, input_dim=77, mid_dim=100, num_classes=10):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)

    def extract_features(self, x):
        x = _flatten_batch(x)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
