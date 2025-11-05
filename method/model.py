from typing import Optional
import os
import torch 
import torch.nn as nn
from utils.general_utils import get_expon_lr_func, inverse_sigmoid 
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

class Optimizer:
    def __init__ (self, cfg, state_dict=None):
        self.optim_par = cfg["parameters"]["optimizer"]
        self.type = self.optim_par.get("type", "adam")
        self.lr = self.optim_par.get("lr", 0.01)
        self.eps = self.optim_par.get("eps", 1e-15)
        self.opt = None
        self._param_groups = None
        self._state_dict = state_dict

    def setup(self, model: 'GaussianModel'):
        if self.type != "adam":
            raise ValueError(f"Unknown optimizer type: {self.type}")
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=self.optim_par.position_lr_init*model.scene_extent, 
            lr_final=self.optim_par.position_lr_final*model.scene_extent,
            lr_delay_mult=self.optim_par.position_lr_delay_mult,
            max_steps=self.optim_par.position_lr_max_steps
        )
        model_params = model.get_opt_params()
        self._param_groups =  [
            {
                'params': [model_params["xyz"]],
                'lr': self.optim_par.position_lr_init,
                "name": "xyz"
            },
            {
                'params': [model_params["features_dc"]],
                'lr': self.optim_par.feature_lr,
                "name": "f_dc"
            },
            {
                'params': [model_params["features_rest"]],
                'lr': self.optim_par.feature_lr / 20.0,
                "name": "f_rest"
            },
            {
                'params': [model_params["opacity"]],
                'lr': self.optim_par.opacity_lr,
                "name": "opacity"
            },
            {
                'params': [model_params["scaling"]],
                'lr': self.optim_par.scaling_lr,
                "name": "scaling"
            },
            {
                'params': [model_params["rotation"]],
                'lr': self.optim_par.rotation_lr,
                "name": "rotation"
            }
        ]
        self.opt = torch.optim.Adam(self._param_groups, lr=self.lr, eps=self.eps)
        if self._state_dict is not None:
            self.opt.load_state_dict(self._state_dict)
        self.step = self.opt.step
        self.zero_grad = self.opt.zero_grad
        self.state_dict = self.opt.state_dict
        self.state = self.opt.state

    def update_learning_rate(self, iteration):
        for param_group in self._param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr

    @property
    def param_groups(self):
        return self._param_groups


class GaussianModel(nn.Module):

    def __init__(self, cfg, *,
                activations: Optional[Activations] = None,
                optimizer: Optional[Optimizer] = None,
                iteration: int = 0,
                xyz: torch.Tensor,
                scaling: torch.Tensor,
                rotation: torch.Tensor,
                opacity: torch.Tensor,
                features_dc: torch.Tensor,
                features_rest: torch.Tensor,
                active_sh_degree: int = 0,
                max_sh_degree: int,
                scene_extent: float,
                white_bg: bool):
        super().__init__()

        if not activations:
            activations = Activations(cfg)

        if not optimizer:
            optimizer = Optimizer(cfg)

        self._act = activations
        self._max_sh_degree = max_sh_degree
        self._active_sh_degree = active_sh_degree
        self._scene_extent = scene_extent
        self.iteration = iteration

        self._xyz = nn.Parameter(xyz.cuda())
        self._scaling = nn.Parameter(scaling.cuda())
        self._rotation = nn.Parameter(rotation.cuda())
        self._opacity = nn.Parameter(opacity.cuda())
        self._features_dc = nn.Parameter(features_dc.cuda())
        self._features_rest = nn.Parameter(features_rest.cuda())

        self._white_background = white_bg
        self.tracer = None
        self._optimizer = optimizer 


    @staticmethod
    def from_dataset(cfg) -> 'GaussianModel':
        ...

    @staticmethod
    def from_checkpoint(cfg, checkpoint) -> 'GaussianModel':
        ...

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'xyz':self._xyz.detach(),
                    'f_dc':self._features_dc.detach(),
                    'f_rest':self._features_rest.detach(),
                    'opacity':self._opacity.detach(),
                    'scaling':self._scaling.detach(),
                    'rotation':self._rotation.detach(),
                    'sh_deg':self._active_sh_degree,
                    'max_sh_deg':self._max_sh_degree,
                    'scene_extent':self._scene_extent,
                    'iteration':self.iteration,
                    'optimizer': self.optimizer.state_dict()\
                          if self.optimizer is not None else None,
                    'white_bg': self._white_background
                    }, path)  

    def oneupSHdegree(self):
        if self.active_sh_degree < self._max_sh_degree:
            self._active_sh_degree += 1

    def setup_training(self):
        if self._optimizer is not None:
            self._optimizer.setup(self)
        for param in self.parameters():
            param.requires_grad = True

    def get_opt_params(self) -> dict:
        return {
            "xyz": self._xyz,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
            "features_dc": self._features_dc,
            "features_rest": self._features_rest
        }

    @property
    def white_background(self):
        return self._white_background

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
    def optimizer(self):
        return self._optimizer

    @property
    def features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1) 

    def forward(self, rays: Rays):
        return TraceFunction.apply(self.opacity, self.xyz, self.scaling, self.rotation, self.features,
                self.tracer, self.active_sh_degree, rays, self._white_background)

GaussianModel.from_dataset = staticmethod(initialize)
GaussianModel.from_checkpoint = staticmethod(load_checkpoint)
