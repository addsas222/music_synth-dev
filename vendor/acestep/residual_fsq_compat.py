import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None



def round_ste(z: Tensor) -> Tensor:
    zhat = z.round()
    return z + (zhat - z).detach()


class FSQ(Module):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks: int = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        channel_first: bool = False,
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer('_levels', _levels, persistent=False)
        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer('_basis', _basis, persistent=False)
        self.scale = scale
        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim
        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim
        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        if num_codebooks > 1 and not keep_num_codebooks_dim:
            raise ValueError('keep_num_codebooks_dim must be true when num_codebooks > 1')
        self.keep_num_codebooks_dim = keep_num_codebooks_dim
        self.dim = default(dim, effective_codebook_dim)
        self.channel_first = channel_first
        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections
        self.codebook_size = int(self._levels.prod().item())
        implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
        self.register_buffer('implicit_codebook', implicit_codebook, persistent=False)

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, torch.tensor(0.5, device=z.device), torch.tensor(0.0, device=z.device))
        shift = torch.atanh(offset / half_l)
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices: Tensor):
        indices = indices.reshape(*indices.shape, 1)
        codes_non_centered = (indices // self._basis) % self._levels
        return self._scale_and_shift_inverse(codes_non_centered)

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        if zhat.shape[-1] != self.codebook_dim:
            raise ValueError(f'expected last dim {self.codebook_dim}, got {zhat.shape[-1]}')
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        codes = self._indices_to_codes(indices)
        if self.keep_num_codebooks_dim:
            codes = codes.reshape(*codes.shape[:-2], -1)
        codes = self.project_out(codes)
        return codes

    def forward(self, z: Tensor):
        orig_dtype = z.dtype
        if self.channel_first and z.ndim >= 3:
            z = z.movedim(1, -1)
            moved = True
        else:
            moved = False

        if z.shape[-1] != self.dim:
            raise ValueError(f'expected dimension of {self.dim} but found {z.shape[-1]}')

        leading_shape = z.shape[:-1]
        z = self.project_in(z)
        z = z.reshape(*leading_shape, self.num_codebooks, self.codebook_dim)
        if z.dtype not in (torch.float32, torch.float64):
            z = z.float()
        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)
        codes = codes.reshape(*leading_shape, self.num_codebooks * self.codebook_dim)
        if codes.dtype != orig_dtype:
            codes = codes.to(orig_dtype)
        out = self.project_out(codes)
        if moved:
            out = out.movedim(-1, 1)
        if not self.keep_num_codebooks_dim:
            indices = indices.reshape(*indices.shape[:-1])
        return out, indices


class ResidualFSQ(Module):
    def __init__(
        self,
        *,
        dim,
        levels: List[int],
        num_quantizers,
        **kwargs,
    ):
        super().__init__()
        codebook_dim = len(levels)
        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection
        self.num_quantizers = num_quantizers
        self.levels = levels
        self.layers = nn.ModuleList([])
        levels_tensor = torch.tensor(levels, dtype=torch.float32)
        scales = []
        for ind in range(num_quantizers):
            scales.append((levels_tensor - 1) ** (-ind))
            fsq = FSQ(levels=levels, dim=codebook_dim, **kwargs)
            self.layers.append(fsq)
        self.codebook_size = self.layers[0].codebook_size
        self.register_buffer('scales', torch.stack(scales), persistent=False)

    @property
    def codebooks(self):
        codebooks = [layer.implicit_codebook for layer in self.layers]
        return torch.stack(codebooks, dim=0)

    def get_codes_from_indices(self, indices):
        if indices.shape[-1] < self.num_quantizers:
            pad_shape = list(indices.shape)
            pad_shape[-1] = self.num_quantizers - indices.shape[-1]
            pad = torch.full(pad_shape, -1, device=indices.device, dtype=indices.dtype)
            indices = torch.cat([indices, pad], dim=-1)

        all_codes = []
        for quantizer_index, layer in enumerate(self.layers):
            index_slice = indices[..., quantizer_index]
            mask = index_slice == -1
            safe_indices = index_slice.masked_fill(mask, 0)
            codes = layer.indices_to_codes(safe_indices)
            codes = codes * self.scales[quantizer_index].to(codes.device, codes.dtype)
            if mask.any():
                codes = codes.masked_fill(mask.unsqueeze(-1), 0.0)
            all_codes.append(codes)
        return torch.stack(all_codes, dim=0)

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = codes.sum(dim=0)
        return self.project_out(codes_summed)

    def forward(self, x, return_all_codes=False, rand_quantize_dropout_fixed_seed=None):
        x = self.project_in(x)
        quantized_out = torch.zeros_like(x)
        residual = self.layers[0].bound(x)
        all_indices = []
        for layer, scale in zip(self.layers, self.scales):
            quantized, indices = layer(residual / scale)
            quantized = quantized * scale.to(quantized.device, quantized.dtype)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
        quantized_out = self.project_out(quantized_out)
        all_indices = torch.stack(all_indices, dim=-1)
        if not return_all_codes:
            return quantized_out, all_indices
        all_codes = self.get_codes_from_indices(all_indices)
        return quantized_out, all_indices, all_codes
