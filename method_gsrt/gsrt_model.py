from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal, Type
import torch

from nerfstudio.models.base_model import Model, ModelConfig

from ..extension import GaussiansTracer

class GSRTModelConfig(ModelConfig):
    _target:Type = field(default_factory=lambda: GSRTModel)

class GSRTModel(Model):
    config: GSRTModelConfig

    def __init__(
        self,
        config: GSRTModelConfig,
        dataparser_transform=None,
        dataparser_scale=None,
        metadata=None,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            **kwargs,
        )
        self.dataparser_transform = dataparser_transform
        self.dataparser_scale = dataparser_scale
        self._gs_tracer = None
        self._gs_initialized = False

    def get_gs_tracer(self):
        if self._gs_tracer is not None:
            return self._gs_tracer
        if not self._gs_initialized:
            self._init_gaussians()
        self._gs_tracer = GaussiansTracer()
        self._gs_tracer.load_gaussians(...)
    
    def _init_gaussians(self):
        gaussians,it = torch.load("data/drums.pth")
        self._gs_initialized = True