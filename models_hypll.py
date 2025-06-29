# Model definitions of the Poincare architectures (not used in experiments)

import torch

from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.manifolds.euclidean import Euclidean
from hypll.tensors import TangentTensor
from typing import Optional

from torch import nn

from hypll import nn as hnn
from hypll.tensors import ManifoldTensor

class PoincareResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        manifold: PoincareBall,
        stride: int = 1,
        downsample: Optional[nn.Sequential] = None,
    ):
        # We can replace each operation in the usual ResidualBlock by a manifold-agnostic
        # operation and supply the PoincareBall object to these operations.
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.manifold = manifold
        self.stride = stride
        self.downsample = downsample

        self.conv1 = hnn.HConvolution2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            manifold=manifold,
            stride=stride,
            padding=1,
        )
        self.bn1 = hnn.HBatchNorm2d(features=out_channels, manifold=manifold)
        self.relu = hnn.HReLU(manifold=self.manifold)
        self.conv2 = hnn.HConvolution2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            manifold=manifold,
            padding=1,
        )
        self.bn2 = hnn.HBatchNorm2d(features=out_channels, manifold=manifold)

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        # We replace the addition operation inside the skip connection by a Mobius addition.
        x = self.manifold.mobius_add(x, residual)
        x = self.relu(x)

        return x


class PoincareResNet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        base_dim: int,
        channel_sizes: list[int],
        group_depths: list[int],
        manifold: PoincareBall,
    ):
        # For the Poincare ResNet itself we again replace each layer by a manifold-agnostic one
        # and supply the PoincareBall to each of these. We also replace the ResidualBlocks by
        # the manifold-agnostic one defined above.
        super().__init__()
        self.channel_sizes = channel_sizes
        self.group_depths = group_depths
        self.manifold = manifold

        self.base_dim = base_dim

        self.conv = hnn.HConvolution2d(
            in_channels=base_dim,
            out_channels=channel_sizes[0],
            kernel_size=3,
            manifold=manifold,
            padding=1,
        )
        self.bn = hnn.HBatchNorm2d(features=channel_sizes[0], manifold=manifold)
        self.relu = hnn.HReLU(manifold=manifold)
        self.group1 = self._make_group(
            in_channels=channel_sizes[0],
            out_channels=channel_sizes[0],
            depth=group_depths[0],
        )
        self.group2 = self._make_group(
            in_channels=channel_sizes[0],
            out_channels=channel_sizes[1],
            depth=group_depths[1],
            stride=2,
        )
        self.group3 = self._make_group(
            in_channels=channel_sizes[1],
            out_channels=channel_sizes[2],
            depth=group_depths[2],
            stride=2,
        )

        self.avg_pool = hnn.HAvgPool2d(kernel_size=8, manifold=manifold)
        self.avg_pool2 = hnn.HAvgPool2d(kernel_size=5, manifold=manifold)
        self.fc = hnn.HLinear(in_features=channel_sizes[2], out_features=n_classes, manifold=manifold)

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.avg_pool(x)
        if self.base_dim != 3: x = self.avg_pool2(x)
        x = x.squeeze()
        x = self.fc(x)
        return x

    def _make_group(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int = 1,
    ) -> nn.Sequential:
        if stride == 1:
            downsample = None
        else:
            downsample = hnn.HConvolution2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                manifold=self.manifold,
                stride=stride,
            )

        layers = [
            PoincareResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                manifold=self.manifold,
                stride=stride,
                downsample=downsample,
            )
        ]

        for _ in range(1, depth):
            layers.append(
                PoincareResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    manifold=self.manifold,
                )
            )

        return nn.Sequential(*layers)


class PoincareResNetModel(nn.Module):
    def __init__(
        self,
        args,
        n_classes: int,
        base_dim: int,
        channel_sizes: list[int],
        group_depths: list[int],
        manifold_type: str,
    ):
        # For the Poincare ResNet itself we again replace each layer by a manifold-agnostic one
        # and supply the PoincareBall to each of these. We also replace the ResidualBlocks by
        # the manifold-agnostic one defined above.
        super().__init__()
        if manifold_type == 'poincare':
            self.manifold = PoincareBall(c=Curvature(value=0.1, requires_grad=True))
        elif manifold_type == 'euclidean':
            self.manifold = Euclidean()
        self.resnet = PoincareResNet(args,
                                     n_classes=n_classes,
                                     base_dim=base_dim,
                                     channel_sizes=channel_sizes,
                                     group_depths=group_depths,
                                     manifold=self.manifold)
    
    def forward(self, x):
        tangents = TangentTensor(data=x, man_dim=1, manifold=self.manifold)
        manifold_inputs = self.manifold.expmap(tangents)
        outputs = self.resnet(manifold_inputs)
        return outputs
