from pyparsing import Optional
import torch 
import torch.nn as nn
from utils.general_utils import safe_state, build_rotation, get_expon_lr_func, inverse_sigmoid  # type: ignore
from .trace_function import TraceFunction
from . import Rays
from .initialization import initialize
from .checkpoint import load_checkpoint

class Activations:
    def __init__(self, cfg):
        self.scaling = torch.exp
        self.scaling_inverse = torch.log
        self.opacity = torch.sigmoid
        self.opacity_inverse = inverse_sigmoid
        self.rotation = torch.nn.functional.normalize

class GaussianModel(nn.Module):

    def __init__(self, *,
                activations: Activations,
                xyz: torch.Tensor,
                scaling: torch.Tensor,
                rotation: torch.Tensor,
                opacity: torch.Tensor,
                features_dc: torch.Tensor,
                features_rest: torch.Tensor,
                active_sh_degree: int = 0,
                max_sh_degree: int,
                scene_extent: float):
        super().__init__()

        self._act = activations
        self._max_sh_degree = max_sh_degree
        self._active_sh_degree = active_sh_degree
        self._scene_extent = scene_extent

        self._xyz = nn.Parameter(xyz)
        self._scaling = nn.Parameter(scaling)
        self._rotation = nn.Parameter(rotation)
        self._opacity = nn.Parameter(opacity)
        self._features_dc = nn.Parameter(features_dc)
        self._features_rest = nn.Parameter(features_rest)
        
        self.optimizer = None

    @staticmethod
    def random_init(cfg) -> 'GaussianModel':
        ...

    @staticmethod
    def from_checkpoint(cfg, checkpoint) -> 'GaussianModel':
        ...

    def get_parameter_groups(self, lr_config):
        return [
            {'params': [self._xyz], 'lr': lr_config.position_lr_init, "name": "xyz"},
            {'params': [self._features_dc], 'lr': lr_config.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': lr_config.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': lr_config.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': lr_config.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': lr_config.rotation_lr, "name": "rotation"},
        ]

    def oneupSHdegree(self):
        if self.active_sh_degree < self._max_sh_degree:
            self._active_sh_degree += 1

    @property
    def num_gaussians(self):
        return self._xyz.shape[0]

    @property
    def active_sh_degree(self):
        assert(self._active_sh_degree <= self._max_sh_degree)
        return self._active_sh_degree

    @property
    def scene_extent(self):
        return self._scene_extent

    @property
    def opacity(self):
        return self._act.opacity(self._opacity)

    @property
    def scaling(self):
        return self._act.scaling(self._scaling)

    @property
    def rotation(self):
        return self._act.rotation(self._rotation)

    @property
    def xyz(self):
        return self._xyz 

    @property
    def features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1) 

    def forward(self, tracer, rays: Rays, white_background=False):
        return TraceFunction.apply(self.opacity, self.xyz, self.scaling, self.rotation, self.features,
                tracer, self.active_sh_degree, rays, white_background)

GaussianModel.random_init = staticmethod(initialize)
GaussianModel.from_checkpoint = staticmethod(load_checkpoint)
