import torch
from torch import nn
import torch.nn.functional as F
from typing import Literal, Tuple, Union, Iterable
import math

activation_fn_dict = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh
}

torch_device = "cuda" if torch.cuda.is_available() else "gpu"

class MLP(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        *,
        activation_fn: Literal["relu", "gelu", "sigmoid", "tanh"] = "relu",
        normalise_output: bool = False,
        normalise_before_activation: bool = False,
    ):
        super().__init__()
        net = [nn.Linear(input_dim, output_dim)]
        if normalise_before_activation and normalise_output:
            net.append(nn.BatchNorm1d(output_dim))
        net.append(activation_fn_dict[activation_fn]())
        if normalise_output:
            net.append(nn.BatchNorm1d(output_dim))
        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class LinearNet(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        layer_sizes: Tuple[int, ...],
        *,
        activation_fn: Union[Literal["relu", "gelu", "sigmoid", "tanh"], Tuple[Literal["relu", "gelu", "sigmoid", "tanh"], ...]] = "relu",
        normalise_output: Union[bool, Tuple[bool, ...]],
        normalise_before_activation: Union[bool, Tuple[bool, ...]],
        device: Literal["cuda", "gpu"] = torch_device,
    ):
        self.device = device
        if isinstance(normalise_output, bool):
            normalise_output = tuple(normalise_output for _ in layer_sizes)
        else:
            assert len(normalise_output) == len(layer_sizes)
        if isinstance(normalise_before_activation, bool):
            normalise_before_activation = tuple(normalise_before_activation for _ in layer_sizes)
        else:
            assert len(normalise_before_activation) == len(layer_sizes)
        if isinstance(activation_fn, str):
            activation_fn = tuple(activation_fn for _ in layer_sizes)
        else:
            assert len(activation_fn) == len(layer_sizes)
        
        modules = [
            MLP(
                input_dim, 
                layer_sizes[0], 
                activation_fn=activation_fn[0],
                normalise_output=normalise_output[0],
                normalise_before_activation=normalise_before_activation[0],
            ).to(self.device)
        ]
        for k in range(1, len(layer_sizes)):
            modules.append(
                MLP(
                    layer_sizes[k - 1], 
                    layer_sizes[k], 
                    activation_fn=activation_fn[k],
                    normalise_output=normalise_output[k],
                    normalise_before_activation=normalise_before_activation[k],
                    device=self.device,
                ).to(self.device)
            )
        self.net = nn.Sequential(*modules).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.net(x)
        
class MMLP(nn.Module):
    def __init__(self, input_dim: Tuple[int, ...], equation: str, bias_shape: Tuple[int, ...], dtype: torch.dtype = torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(bias_shape, dtype=dtype))
        self.equation = equation
        for param in self.parameters():
            nn.init.normal_(param)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.einsum(self.equation, input, self.weight) + self.bias
    

class xLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, embedding_dim, device=torch.device('cuda')):
        super().__init__()
        self.device = device
        # output, value, key, query
        self.multilinear_gates = MMLP(
            (hidden_size*4, input_dim, embedding_dim), 
            equation='bi, lij -> blj', 
            bias_shape=(hidden_size*4, embedding_dim)
        ).to(self.device)
        # input, forget
        self.linear_gates = nn.Linear(input_dim, hidden_size * 2).to(self.device)
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.scale = math.sqrt(embedding_dim)
        
        self.hidden_state, self.cell_state = (
            torch.zeros((1, self.hidden_size, self.embedding_dim), dtype=torch.float32).to(self.device), 
            torch.zeros((1, self.hidden_size, self.embedding_dim, self.embedding_dim), dtype=torch.float32).to(self.device)
        )
        self.m, self.n = (
            torch.zeros((1, self.hidden_size), dtype=torch.float32).to(self.device), 
            torch.zeros((1, self.hidden_size, self.embedding_dim), dtype=torch.float32).to(self.device)
        )
        
    @torch.jit.export
    def reset_states(self, batch_size: int) -> None:
        batch_size = int(batch_size)
        self.hidden_state, self.cell_state = (
            torch.zeros((batch_size, self.hidden_size, self.embedding_dim), dtype=torch.float32).to(self.device), 
            torch.zeros((batch_size, self.hidden_size, self.embedding_dim, self.embedding_dim), dtype=torch.float32).to(self.device)
        )
        self.m, self.n = (
            torch.zeros((batch_size, self.hidden_size), dtype=torch.float32).to(self.device), 
            torch.zeros((batch_size, self.hidden_size, self.embedding_dim), dtype=torch.float32).to(self.device)
        )
        
    @torch.jit.export
    def _forward_step(self, input: torch.Tensor) -> None:
        linear_gates = self.linear_gates(input)
        multilinear_gates = self.multilinear_gates(input)
        o_, v, k, q = torch.chunk(multilinear_gates, 4, dim=1)
        o = F.sigmoid(o_)
        i_, f_ = torch.chunk(linear_gates, 2, dim=1)
        m = torch.amax(torch.stack((f_ + self.m, i_), dim=2), dim=2)
        i = torch.exp(i_ - m)
        f = torch.exp(f_ + self.m - m)
        self.m = m
        self.cell_state = (f.unsqueeze(-1).unsqueeze(-1) * self.cell_state) + i.unsqueeze(-1).unsqueeze(-1)*torch.einsum('bli, blj -> blij', v, k)
        self.n = f.unsqueeze(-1) * self.n + i.unsqueeze(-1) * k
        self.hidden_state = o * torch.einsum('blij, bli -> blj', self.cell_state, q) / torch.clamp(torch.sum(self.n*q, dim=-1), 1, torch.inf).unsqueeze(-1)
    
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input = input.to(self.device)
        self.reset_states(len(input))
        for t in range(input.shape[1]):
            self._forward_step(input[:, t, :])
        return self.hidden_state, self.cell_state
    

class TimeseriesConv2D(nn.Module):
    def __init__(self, columns: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.conv = nn.Conv2d(
                in_channels=1, 
                out_channels=out_channels,
                kernel_size=(kernel_size, columns),
                stride=stride,
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(-1)
        return x


class ResidualConv1D(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=out_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.connection = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1d(x) + self.connection(x)


class ResidualTimeseriesConv2D(nn.Module):
    def __init__(
        self, 
        input_dim: Tuple[int, int],
        *,
        out_channels: Tuple[int, ...], 
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]],
        activation_fn: Union[Literal["relu", "gelu", "sigmoid", "tanh"], None] = None,
        normalise_output: bool = False,
    ):
        super().__init__()
        rows, columns = input_dim
        modules = [
            TimeseriesConv2D(
                columns=columns,
                out_channels=out_channels[0],
                kernel_size=kernel_size[0],
                stride=stride[0],
            )
        ]
        if activation_fn is not None:
            modules.append(activation_fn_dict[activation_fn]())
        if normalise_output:
            modules.append(nn.BatchNorm1d(num_features=out_channels[0]))
        rows = math.ceil((rows - kernel_size[0] + 1) / stride[0])
        assert rows > 0
        for k in range(1, len(out_channels)):
            modules.append(
                ResidualConv1D(
                    in_channels=out_channels[k - 1], 
                    out_channels=out_channels[k],
                    kernel_size=kernel_size[k],
                    stride=stride[k],
                )
            )
            if activation_fn is not None:
                modules.append(activation_fn_dict[activation_fn]())
            if normalise_output:
                modules.append(nn.BatchNorm1d(num_features=out_channels[k]))
            rows = math.ceil((rows - kernel_size[k] + 1) / stride[k])
            assert rows > 0
        modules.append(
            MMLP(
                input_dim=(rows, out_channels[-1]),
                equation="bcr, ri -> bc",
                bias_shape=(out_channels[-1],)
            )
        )
        if activation_fn is not None:
            modules.append(activation_fn_dict[activation_fn]())
        if normalise_output:
            modules.append(nn.BatchNorm1d(num_features=out_channels[-1]))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    input_dim = (12, 6)
    out_channels = (4, 4)
    kernel_size = (3, 2)
    stride = (1, 2)
    activation_fn = "relu"
    normalise_output = True

    test = ResidualTimeseriesConv2D(
        input_dim=input_dim,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        activation_fn=activation_fn,
        normalise_output=normalise_output,
    )

    print(test(torch.randn(7, 12, 6)).shape)