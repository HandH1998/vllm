"""Custom activation functions."""
import torch
import torch.nn as nn

from vllm import activation_ops


class SiluAndMul(nn.Module):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[1] // 2.

    Shapes:
        x: (num_tokens, 2 * d)
        return: (num_tokens, d)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]
        d = x.shape[1] // 2
        out = torch.empty(num_tokens, d, dtype=x.dtype, device=x.device)
        activation_ops.silu_and_mul(out, x)
        return out
    
class DequantSiluAndMulQuant(nn.Module):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[1] // 2.

    Shapes:
        x: (num_tokens, 2 * d)
        return: (num_tokens, d)
    """
    def __init__(self, scale_in: float = 1.0, scale_out: float = 1.0) -> None:
        super().__init__()
        self.register_buffer('a', torch.tensor(scale_in, dtype=torch.float32, requires_grad=False))
        self.register_buffer('inscale', torch.tensor(scale_out, dtype=torch.float32, requires_grad=False))
        
    def _apply(self, fn):
        super()._apply(fn)
        self.a = self.a.cpu()
        self.inscale = self.inscale.cpu()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.a = self.a.to(*args, **kwargs)
        self.a = self.a.to(torch.float32)
        self.inscale = self.inscale.to(*args, **kwargs)
        self.inscale = self.inscale.to(torch.float32)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]
        d = x.shape[1] // 2
        out = torch.empty(num_tokens, d, dtype=torch.int8, device=x.device)
        activation_ops.invoke_dequant_silu_and_mul_quant(out, x, self.a.item(), self.a.item(), self.inscale.item())
        return out  

class DequantSiluAndMul(nn.Module):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[1] // 2.

    Shapes:
        x: (num_tokens, 2 * d)
        return: (num_tokens, d)
    """
    def __init__(self, scale_in: float = 1.0) -> None:
        super().__init__()
        self.register_buffer('a', torch.tensor(scale_in, dtype=torch.float32, requires_grad=False))
        
    def _apply(self, fn):
        super()._apply(fn)
        self.a = self.a.cpu()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.a = self.a.to(*args, **kwargs)
        self.a = self.a.to(torch.float32)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]
        d = x.shape[1] // 2
        # to do: dytpe
        out = torch.empty(num_tokens, d, dtype=torch.float16, device=x.device)
        activation_ops.invoke_dequant_silu_and_mul(out, x, self.a.item(), self.a.item())
        return out

class NewGELU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]
        d = x.shape[1]
        out = torch.empty(num_tokens, d, dtype=x.dtype, device=x.device)
        activation_ops.gelu_new(out, x)
        return out


class FastGELU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]
        d = x.shape[1]
        out = torch.empty(num_tokens, d, dtype=x.dtype, device=x.device)
        activation_ops.gelu_fast(out, x)
        return out


_ACTIVATION_REGISTRY = {
    "gelu": nn.GELU(),
    "gelu_fast": FastGELU(),
    "gelu_new": NewGELU(),
    "gelu_pytorch_tanh": nn.GELU(approximate="tanh"),
    "relu": nn.ReLU(),
}


def get_act_fn(act_fn: str) -> nn.Module:
    """Get an activation function by name."""
    act_fn = act_fn.lower()
    if act_fn in _ACTIVATION_REGISTRY:
        return _ACTIVATION_REGISTRY[act_fn]
    raise ValueError(f"Activation function {act_fn!r} is not supported.")
