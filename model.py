import copy
import functools
import math
from typing import Any, Union, Sequence, Type

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_conv1d_layer(
    in_channels: int, out_channels: int, ln_shape: tuple[int], kernel_size: int, pool_size: int, use_pooling: bool, dropout_rate: float
) -> nn.Module:
    layers = [
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size),
        nn.LayerNorm(ln_shape),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
    ]
    if use_pooling:
        layers.append(nn.MaxPool1d(kernel_size=pool_size))
    return nn.Sequential(*layers)

def _get_time_shape(
    time_dim: int, ksize: int, dilation: int = 1, stride: int = 1, padding: int = 0, pool_size: int = 1,
) -> int:
    return int(((time_dim // pool_size + 2 * padding - dilation * (ksize-1) - 1) / stride + 1))

@gin.configurable
class ProjectionHead(nn.Module):

    def __init__(
            self,
            in_shape: tuple[int],
            conv_channels: Sequence[int] = (32, 32),
            dense_neurons: Sequence[int] = (64, 32, 1),
            use_poolings: Sequence[bool] = (True, True),
            kernel_size: int = 5,
            pool_size: int = 5,
            dropout_rate: float = 0.3,
            ):
        super(ProjectionHead, self).__init__()
        assert len(dense_neurons) == 3
        assert len(conv_channels) == len(use_poolings)

        conv_layer = functools.partial(_get_conv1d_layer, kernel_size=kernel_size, pool_size=pool_size, dropout_rate=dropout_rate)
        projection_channels = 128
        ln_shape = (projection_channels, _get_time_shape(in_shape[1], ksize=5))
        layers = [_get_conv1d_layer(in_shape[0], projection_channels, ln_shape=ln_shape, kernel_size=5, pool_size=5, use_pooling=False, dropout_rate=0.3)]
        conv_channels = [projection_channels] + list(conv_channels)
        use_poolings_shape = (False,) + use_poolings[:-1]

        for in_channels, out_channels, use_pooling, use_pooling_shape in zip(conv_channels[:-1], conv_channels[1:], use_poolings, use_poolings_shape):
            ln_shape = (out_channels, _get_time_shape(ln_shape[-1], ksize=kernel_size, pool_size=1 if not use_pooling_shape else pool_size))
            layers.append(conv_layer(in_channels, out_channels, ln_shape=ln_shape, use_pooling=use_pooling))

        self._encoder = nn.Sequential(*layers)
        self._flatten = nn.Flatten()
        in_dense = self._get_in_dense(in_shape)
        hidden_dense1, hidden_dense2, out_dense = dense_neurons
        self._head = nn.Sequential(
            nn.Linear(in_dense, hidden_dense1),
            nn.ReLU(),
            nn.Linear(hidden_dense1, hidden_dense2),
            nn.ReLU(),
            nn.Linear(hidden_dense2, out_dense),
        )

    def _get_in_dense(self, in_shape: tuple[int]) -> int:
        x = torch.zeros((1,)+in_shape)
        x = self._encoder(x)
        return self._flatten(x).shape[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._encoder(x)
        x = self._flatten(x)
        return 2*self._head(x)+3
