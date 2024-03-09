#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：Lagrange.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/3/9
"""

from typing import Optional
from safebench.util.torch_util import CUDA
from .net import build_mlp, layer_init_with_orthogonal
import torch


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
        lambda_lr: float,
        lambda_optimizer: str,
        lagrangian_upper_bound: Optional[float] = None,
    ) -> None:
        """Initialize an instance of :class:`Lagrange`."""
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.constraint_limit: float = constraint_limit
        self.lambda_lr: float = lambda_lr
        self.lagrangian_upper_bound: Optional[float] = lagrangian_upper_bound

        self.lagrangian_multiplier = CUDA(build_mlp(dims=[state_dim, *hidden_dims, 1]))
        # initialize
        layer_init_with_orthogonal(self.lagrangian_multiplier[-1], std=0.1)

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
        lagrangian_multiplier = lagrangian_multiplier.squeeze(1).clamp_(min=-1., max=self.lagrangian_upper_bound)
        return lagrangian_multiplier

    def compute_lambda_loss(self, state: torch.Tensor, constraint: torch.Tensor) -> torch.Tensor:
        """
            Penalty loss for Lagrange multiplier.
        """
        constraint = CUDA(constraint)
        constraint_limit = CUDA(torch.ones_like(constraint) * self.constraint_limit)
        loss = -self.lagrangian_multiplier(state) * (constraint - constraint_limit)
        return loss.mean()

    def update_lagrange_multiplier(self, state: torch.Tensor, constraint: torch.Tensor) -> None:
        """
            Update Lagrange multiplier (lambda).
        """
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(state, constraint)
        lambda_loss.backward()
        self.lambda_optimizer.step()
