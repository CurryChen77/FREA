#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：Lagrange.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/3/9
"""

from typing import Optional, Tuple
from safebench.util.torch_util import CUDA
from torch import nn
import torch.nn.init as init

import torch


class LagrangeMultiplier(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: [int], hidden_activation: nn = nn.ReLU, output_activation: nn = nn.Softplus):
        super(LagrangeMultiplier, self).__init__()
        dims = [state_dim, *hidden_dims, 1]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # add
                self.layers.append(hidden_activation())
        self.layers.append(output_activation())
        # init
        self.apply(self.init_weights)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0.1)


class Lagrange:
    """
    Modified from: https://github.com/PKU-Alignment/omnisafe/blob/main/omnisafe/common/lagrange.py

    Base class for Lagrangian-base Algorithms.

    This class implements the State-wise Lagrange multiplier update and the Lagrange loss.


    Args:
        cost_limit (float): The cost limit.
        lagrangian_multiplier_init (float): The initial value of the Lagrange multiplier.
        lambda_lr (float): The learning rate of the Lagrange multiplier.
        lambda_optimizer (str): The optimizer for the Lagrange multiplier.
        lagrangian_upper_bound (float or None, optional): The upper bound of the Lagrange multiplier.
            Defaults to None.

    Attributes:
        cost_limit (float): The cost limit.
        lambda_lr (float): The learning rate of the Lagrange multiplier.
        lagrangian_upper_bound (float, optional): The upper bound of the Lagrange multiplier.
            Defaults to None.
        lagrangian_multiplier (torch.nn.Parameter): The Lagrange multiplier.
        lambda_range_projection (torch.nn.ReLU): The projection function for the Lagrange multiplier.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        state_dim: int,
        hidden_dims: [int],
        constraint_limit: float,
        multiplier_upper_bound: int,
        lambda_lr: float,
        lambda_optimizer: str,
    ) -> None:
        """Initialize an instance of :class:`Lagrange`."""
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.constraint_limit: float = constraint_limit
        self.lambda_lr: float = lambda_lr
        self.multiplier_upper_bound: Optional[float] = multiplier_upper_bound
        self.criterion = nn.MSELoss()
        self.lagrangian_multiplier = CUDA(LagrangeMultiplier(state_dim=state_dim, hidden_dims=hidden_dims, hidden_activation=nn.ReLU, output_activation=nn.Softplus))

        # fetch optimizer from PyTorch optimizer package
        assert hasattr(
            torch.optim,
            lambda_optimizer,
        ), f'Optimizer={lambda_optimizer} not found in torch.'
        torch_opt = getattr(torch.optim, lambda_optimizer)
        self.lambda_optimizer: torch.optim.Optimizer = torch_opt(
            self.lagrangian_multiplier.parameters(),
            lr=lambda_lr,
        )

    def get_lagrangian_multiplier(self, state: torch.Tensor) -> torch.Tensor:
        # get lagrangian multiplier
        lagrangian_multiplier = self.lagrangian_multiplier(state)  # [B, 1]
        # set upper bound if necessary
        lagrangian_multiplier = lagrangian_multiplier.squeeze(1).clamp_(min=0.0, max=self.multiplier_upper_bound)
        return lagrangian_multiplier

    def compute_lambda_loss(self, state: torch.Tensor, constraint: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Special lambda loss
            For safe states, learn their lambdas: 0 or finite values
            For unsafe states, want their lambdas to be close to the upper bound, a large penalty
            why (upper_bound - 1): because the upperbound of lambdas is upper_bound, to avoid grad vanishing
        """
        penalty = CUDA(constraint - torch.ones_like(constraint) * self.constraint_limit)
        lagrangian_multiplier = self.get_lagrangian_multiplier(state)
        lagrangian_multiplier_safe = torch.mul(constraint <= 0, lagrangian_multiplier)
        lagrangian_multiplier_unsafe = torch.mul(constraint > 0, lagrangian_multiplier)
        # special lambda multiplier loss
        loss = - 0.5 * torch.mean(torch.mul(lagrangian_multiplier_safe, penalty)) + \
            self.criterion(lagrangian_multiplier_unsafe, (constraint > 0) * (self.multiplier_upper_bound - 1))
        return loss.mean(), lagrangian_multiplier

    def update_lagrange_multiplier(self, state: torch.Tensor, constraint: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Update Lagrange multiplier (lambda).
        """
        self.lambda_optimizer.zero_grad()
        lambda_loss, lagrange_multiplier = self.compute_lambda_loss(state, constraint)
        lambda_loss.backward()
        nn.utils.clip_grad_norm_(self.lagrangian_multiplier.parameters(), 0.5)
        self.lambda_optimizer.step()
        return lambda_loss, lagrange_multiplier
