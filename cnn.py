from __future__ import annotations
from typing import Sequence

from tinygrad import Tensor, nn
from tinygrad.nn import Conv1d, BatchNorm1d, Dropout

def _make_ntuple(x: int | Sequence[int], n: int) -> tuple[int, ...]:
    return (x,) * n if isinstance(x, int) else tuple(x)

class ReLU:
    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()

class MaxPool1d:
    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int] = 0,
    ):
        self.kernel = _make_ntuple(kernel_size, 2)
        self.stride = _make_ntuple(stride if stride is not None else kernel_size, 2)
        self.pad = _make_ntuple(padding, 2)

    def __call__(self, x: Tensor) -> Tensor:
        # (N,C,L) -> (N,C,1,L)  ;  pool  -> (N,C,1,L')  ;  -> (N,C,L')
        return (
            x.reshape(*x.shape[:2], 1, x.shape[2])
            .max_pool2d(self.kernel, self.stride, self.pad)
            .reshape(x.shape[0], x.shape[1], -1)
        )

class AvgPool1d:
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0):
        self.kernel = _make_ntuple(kernel_size, 2)
        self.stride = _make_ntuple(stride if stride is not None else kernel_size, 2)
        self.pad = _make_ntuple(padding, 2)

    def __call__(self, x: Tensor) -> Tensor:
        return (
            x.reshape(*x.shape[:2], 1, x.shape[2])
            .avg_pool2d(self.kernel, self.stride, self.pad)
            .reshape(x.shape[0], x.shape[1], -1)
        )

class InceptionBlock:
    def __init__(self, in_ch: int, out_ch: int):
        assert out_ch % 4 == 0, "out_ch must be divisible by 4 for equal split"
        b = out_ch // 4
        # branch 1: 1×1
        self.branch1 = Conv1d(in_ch, b, 1, bias=False)
        self.bn1 = BatchNorm1d(b)
        # branch 2: 1×3
        self.branch2 = Conv1d(in_ch, b, 3, padding=1, bias=False)
        self.bn2 = BatchNorm1d(b)
        # branch 3: 1×5
        self.branch3 = Conv1d(in_ch, b, 5, padding=2, bias=False)
        self.bn3 = BatchNorm1d(b)
        # branch 4: pool → 1×1
        self.pool = MaxPool1d(3, stride=1, padding=1)
        self.branch4 = Conv1d(in_ch, b, 1, bias=False)
        self.bn4 = BatchNorm1d(b)
        self.relu = ReLU()

    def __call__(self, x: Tensor) -> Tensor:
        o1 = self.relu(self.bn1(self.branch1(x)))
        o2 = self.relu(self.bn2(self.branch2(x)))
        o3 = self.relu(self.bn3(self.branch3(x)))
        o4 = self.relu(self.bn4(self.branch4(self.pool(x))))
        return o1.cat(o2, o3, o4, dim=1)

class DeepSignalCNN:
    """
    Input : (batch, 1, 360) raw Nanopore signal
    Output: (batch, 256) feature vector
    """

    def __init__(self, dropout_p: float = 0.2):
        self.stem:list = [
            Conv1d(1, 64, 7, stride=2, bias=False),
            BatchNorm1d(64),
            ReLU(),
            MaxPool1d(3, stride=2, padding=1),
        ]
        self.inc3 = [InceptionBlock(64 if i == 0 else 128, 128) for i in range(3)]
        self.pool1 = MaxPool1d(3, stride=2, padding=1)
        self.pool2 = MaxPool1d(3, stride=2, padding=1)
        self.inc5 = [InceptionBlock(128 if i == 0 else 256, 256) for i in range(5)]
        self.inc3_2 = [InceptionBlock(256, 256) for _ in range(3)]
        self.global_pool = AvgPool1d(kernel_size=7, stride=1, padding=0)
        self.dropout = Dropout(dropout_p)
        for idx, layer in enumerate(self.stem):
            setattr(self, f"stem_{idx}", layer)
        for idx, layer in enumerate(self.inc3):
            setattr(self, f"inc3_{idx}", layer)
        for idx, layer in enumerate(self.inc5):
            setattr(self, f"inc5_{idx}", layer)
        for idx, layer in enumerate(self.inc3_2):
            setattr(self, f"inc3_2_{idx}", layer)

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.stem:
            x = layer(x)
        for layer in self.inc3:
            x = layer(x)
        x = self.pool1(x)
        x = self.pool2(x)
        for layer in self.inc5:
            x = layer(x)
        for layer in self.inc3_2:
            x = layer(x)
        x = self.global_pool(x)
        x = x.reshape(x.shape[0], -1)  # (B, C)
        x = self.dropout(x)
        return x

if __name__ == "__main__":
    model = DeepSignalCNN(dropout_p=0.2)
    sig = Tensor.randn(16, 1, 360)  # batch=16, length=360
    feat = model(sig)
    print("Input :", sig.shape)
    print("Output:", feat.shape)  # -> (16, 256)
