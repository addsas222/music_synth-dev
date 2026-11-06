from __future__ import annotations

import json
import math
import os
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

try:
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:  # pragma: no cover
    load_safetensors_file = None



def _randn_tensor(shape, generator=None, device=None, dtype=None):
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


class Snake1d(nn.Module):
    def __init__(self, hidden_dim: int, logscale: bool = True):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.logscale = logscale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        alpha = self.alpha if not self.logscale else torch.exp(self.alpha)
        beta = self.beta if not self.logscale else torch.exp(self.beta)
        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
        hidden_states = hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)
        return hidden_states.reshape(shape)


class OobleckResidualUnit(nn.Module):
    def __init__(self, dimension: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dimension)
        self.conv1 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=7, dilation=dilation, padding=pad))
        self.snake2 = Snake1d(dimension)
        self.conv2 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=1))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        output_tensor = self.conv1(self.snake1(hidden_state))
        output_tensor = self.conv2(self.snake2(output_tensor))
        padding = (hidden_state.shape[-1] - output_tensor.shape[-1]) // 2
        if padding > 0:
            hidden_state = hidden_state[..., padding:-padding]
        return hidden_state + output_tensor


class OobleckEncoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int = 1):
        super().__init__()
        self.res_unit1 = OobleckResidualUnit(input_dim, dilation=1)
        self.res_unit2 = OobleckResidualUnit(input_dim, dilation=3)
        self.res_unit3 = OobleckResidualUnit(input_dim, dilation=9)
        self.snake1 = Snake1d(input_dim)
        self.conv1 = weight_norm(
            nn.Conv1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2))
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        hidden_state = self.snake1(self.res_unit3(hidden_state))
        return self.conv1(hidden_state)


class OobleckDecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int = 1):
        super().__init__()
        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = weight_norm(
            nn.ConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )
        )
        self.res_unit1 = OobleckResidualUnit(output_dim, dilation=1)
        self.res_unit2 = OobleckResidualUnit(output_dim, dilation=3)
        self.res_unit3 = OobleckResidualUnit(output_dim, dilation=9)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv_t1(hidden_state)
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        return self.res_unit3(hidden_state)


class OobleckDiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.scale = parameters.chunk(2, dim=1)
        self.std = nn.functional.softplus(self.scale) + 1e-4
        self.var = self.std * self.std
        self.logvar = torch.log(self.var)
        self.deterministic = deterministic

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        sample = _randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std * sample

    def mode(self) -> torch.Tensor:
        return self.mean


@dataclass
class AutoencoderOobleckOutput:
    latent_dist: OobleckDiagonalGaussianDistribution


@dataclass
class OobleckDecoderOutput:
    sample: torch.Tensor


class OobleckEncoder(nn.Module):
    def __init__(self, encoder_hidden_size, audio_channels, downsampling_ratios, channel_multiples):
        super().__init__()
        channel_multiples = [1] + list(channel_multiples)
        self.conv1 = weight_norm(nn.Conv1d(audio_channels, encoder_hidden_size, kernel_size=7, padding=3))
        blocks = []
        for stride_index, stride in enumerate(downsampling_ratios):
            blocks.append(
                OobleckEncoderBlock(
                    input_dim=encoder_hidden_size * channel_multiples[stride_index],
                    output_dim=encoder_hidden_size * channel_multiples[stride_index + 1],
                    stride=stride,
                )
            )
        self.block = nn.ModuleList(blocks)
        d_model = encoder_hidden_size * channel_multiples[-1]
        self.snake1 = Snake1d(d_model)
        self.conv2 = weight_norm(nn.Conv1d(d_model, encoder_hidden_size, kernel_size=3, padding=1))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv1(hidden_state)
        for module in self.block:
            hidden_state = module(hidden_state)
        hidden_state = self.snake1(hidden_state)
        return self.conv2(hidden_state)


class OobleckDecoder(nn.Module):
    def __init__(self, channels, input_channels, audio_channels, upsampling_ratios, channel_multiples):
        super().__init__()
        channel_multiples = [1] + list(channel_multiples)
        self.conv1 = weight_norm(nn.Conv1d(input_channels, channels * channel_multiples[-1], kernel_size=7, padding=3))
        blocks = []
        for stride_index, stride in enumerate(upsampling_ratios):
            blocks.append(
                OobleckDecoderBlock(
                    input_dim=channels * channel_multiples[len(upsampling_ratios) - stride_index],
                    output_dim=channels * channel_multiples[len(upsampling_ratios) - stride_index - 1],
                    stride=stride,
                )
            )
        self.block = nn.ModuleList(blocks)
        self.snake1 = Snake1d(channels)
        self.conv2 = weight_norm(nn.Conv1d(channels, audio_channels, kernel_size=7, padding=3, bias=False))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv1(hidden_state)
        for layer in self.block:
            hidden_state = layer(hidden_state)
        hidden_state = self.snake1(hidden_state)
        return self.conv2(hidden_state)


class AutoencoderOobleck(nn.Module):
    _supports_gradient_checkpointing = False
    _supports_group_offloading = False

    def __init__(
        self,
        encoder_hidden_size: int = 128,
        downsampling_ratios=None,
        channel_multiples=None,
        decoder_channels: int = 128,
        decoder_input_channels: int = 64,
        audio_channels: int = 2,
        sampling_rate: int = 44100,
    ):
        super().__init__()
        downsampling_ratios = downsampling_ratios or [2, 4, 4, 8, 8]
        channel_multiples = channel_multiples or [1, 2, 4, 8, 16]

        self.encoder_hidden_size = encoder_hidden_size
        self.downsampling_ratios = list(downsampling_ratios)
        self.decoder_channels = decoder_channels
        self.upsampling_ratios = list(downsampling_ratios)[::-1]
        self.hop_length = int(math.prod(downsampling_ratios))
        self.sampling_rate = sampling_rate
        self.audio_channels = audio_channels
        self.decoder_input_channels = decoder_input_channels
        self.channel_multiples = list(channel_multiples)
        self.config = SimpleNamespace(
            encoder_hidden_size=encoder_hidden_size,
            downsampling_ratios=list(downsampling_ratios),
            channel_multiples=list(channel_multiples),
            decoder_channels=decoder_channels,
            decoder_input_channels=decoder_input_channels,
            audio_channels=audio_channels,
            sampling_rate=sampling_rate,
        )

        self.encoder = OobleckEncoder(
            encoder_hidden_size=encoder_hidden_size,
            audio_channels=audio_channels,
            downsampling_ratios=downsampling_ratios,
            channel_multiples=channel_multiples,
        )
        self.decoder = OobleckDecoder(
            channels=decoder_channels,
            input_channels=decoder_input_channels,
            audio_channels=audio_channels,
            upsampling_ratios=self.upsampling_ratios,
            channel_multiples=channel_multiples,
        )
        self.use_slicing = False

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike]):
        model_dir = Path(pretrained_model_name_or_path)
        config_path = model_dir / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError(f'VAE config not found at {config_path}')
        config = json.loads(config_path.read_text(encoding='utf-8'))
        init_kwargs = {
            'encoder_hidden_size': config['encoder_hidden_size'],
            'downsampling_ratios': config['downsampling_ratios'],
            'channel_multiples': config['channel_multiples'],
            'decoder_channels': config['decoder_channels'],
            'decoder_input_channels': config['decoder_input_channels'],
            'audio_channels': config['audio_channels'],
            'sampling_rate': config['sampling_rate'],
        }
        model = cls(**init_kwargs)

        safetensors_path = model_dir / 'diffusion_pytorch_model.safetensors'
        bin_path = model_dir / 'diffusion_pytorch_model.bin'
        if safetensors_path.exists():
            if load_safetensors_file is None:
                raise ImportError('safetensors is required to load the ACE VAE weights')
            state_dict = load_safetensors_file(str(safetensors_path))
        elif bin_path.exists():
            state_dict = torch.load(str(bin_path), map_location='cpu')
        else:
            raise FileNotFoundError(f'No VAE weights found under {model_dir}')

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(f'Failed to load VAE weights cleanly. missing={missing}, unexpected={unexpected}')
        return model

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            hidden = torch.cat(encoded_slices)
        else:
            hidden = self.encoder(x)
        posterior = OobleckDiagonalGaussianDistribution(hidden)
        if not return_dict:
            return (posterior,)
        return AutoencoderOobleckOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor, return_dict: bool = True):
        decoded = self.decoder(z)
        if not return_dict:
            return (decoded,)
        return OobleckDecoderOutput(sample=decoded)

    def decode(self, z: torch.Tensor, return_dict: bool = True, generator=None):
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample
        if not return_dict:
            return (decoded,)
        return OobleckDecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        posterior = self.encode(sample).latent_dist
        latents = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        decoded = self.decode(latents).sample
        if not return_dict:
            return (decoded,)
        return OobleckDecoderOutput(sample=decoded)
