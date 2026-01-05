from typing import Optional
import os
import torch 
import torch.nn as nn
from utils.general_utils import get_expon_lr_func, inverse_sigmoid 
from .trace_function import TraceFunction
from . import Rays
from .initialization import initialize
from .checkpoint import load_checkpoint
from extension import GaussiansTracer

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
        self._optimizer = optimizer 
        self._tracer = GaussiansTracer()
        self.as_params = {"type": cfg.tracer_type}

        #self.xyz_2d = torch.zeros_like(xyz, requires_grad=True).cuda()
        self.fps = 0
        self.gas_size = 0
        self.num_its = 0

    @staticmethod
    def from_dataset(cfg) -> 'GaussianModel':
        ...

    @staticmethod
    def from_checkpoint(cfg, checkpoint) -> 'GaussianModel':
        ...

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.isdir(path):
            path = os.path.join(path,"chpt.pt")
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
                          if self.optimizer is not None and "state_dict" in self.optimizer.__dict__ else None,
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
        #self.xyz_2d.grad = None
        col,stats = TraceFunction.apply(self.opacity, self.xyz, self.scaling, self.rotation, self.features, None,#self.xyz_2d,
                self._tracer, self.active_sh_degree, rays, self._white_background, self.training, self.as_params)
        self.fps = 1e3/stats["time_ms"] if stats["time_ms"]!=0 else 0
        self.gas_size = stats.get("gas_size",0)/1024/1024
        self.num_its = stats["num_its"]
        return col

    def gradcheck(self, rays: Rays, gt_image, fn_loss, n_points=100, pad_x = None, pad_y = None, **kwargs) -> bool:
        def func(opacity, xyz, scaling, rotation, features):
            img = TraceFunction.apply(opacity, xyz, scaling, rotation, features,
                self._tracer, self.active_sh_degree, rays, self._white_background, True, self.as_params)
            return fn_loss(img, gt_image)
        from torch.autograd import gradcheck
        inputs = (self.opacity.detach(),
                  self.xyz.detach(),
                  self.scaling.detach(),
                  self.rotation.detach(),
                  self.features.detach())
        for inp in inputs:
            inp.requires_grad = True
        # run pass with padded rays
        all_rays = rays
        full_gt_image = gt_image
        u,v = torch.meshgrid(torch.arange(rays.res_x),torch.arange(rays.res_y),indexing='xy')
        if pad_x is None and pad_y is None:
            pad_x,pad_y = rays.res_x//4, rays.res_y//4
        assert pad_x < rays.res_x//2 and pad_y < rays.res_y//2
        mask = (u.flatten() >= pad_x) & (u.flatten() < (rays.res_x-pad_x)) \
               & (v.flatten() >= pad_y) & (v.flatten() < (rays.res_y-pad_y))
        sub_rays = Rays(origins=rays.origins[mask], directions=rays.directions[mask],
                         res_x=rays.res_x-2*pad_x, res_y=rays.res_y-2*pad_y)
        gt_image = gt_image[mask]
        assert sub_rays.origins.shape[0] == (sub_rays.res_x*sub_rays.res_y)
        rays = sub_rays
        func(*inputs).backward()
        rays = all_rays
        gt_image = full_gt_image
        # sample gaussians that received gradients
        any_grad = any(torch.count_nonzero(inp.grad) != 0 for inp in inputs)
        assert any_grad, "All grads are zero!"
        mask = torch.zeros(self.num_gaussians, dtype=torch.bool, device=self._xyz.device)
        for inp in inputs:
            mask |= torch.any(inp.grad.view(self.num_gaussians,-1),dim=1)
        ids = torch.nonzero(mask, as_tuple=True)[0][:n_points]
        print(f"Running gradcheck on {ids.shape[0]} points...")
        del inputs
        torch.cuda.empty_cache()
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        inputs = (self.opacity.detach()[ids],
                  self.xyz.detach()[ids],
                  self.scaling.detach()[ids],
                  self.rotation.detach()[ids],
                  self.features.detach()[ids])
        for inp in inputs:
            inp.requires_grad = True
        return gradcheck(func, inputs, **kwargs)

    def get_wandb_log(self):
        import wandb
        pos_grad_norm = torch.norm(self._xyz.grad,dim=1) if self._xyz.grad is not None else torch.zeros(self._xyz.shape[0])
        #pos_grad_2d_norm = torch.norm(self.xyz_2d.grad,dim=1) if self.xyz_2d.grad is not None else torch.zeros(self._xyz.shape[0])
        return {
            'opacities': wandb.Histogram(self.opacity.detach().cpu().numpy(),num_bins=100),
            'scales_max': wandb.Histogram(self.scaling.detach().cpu().numpy().max(axis=-1),num_bins=100),
            'pos_grad_norm': wandb.Histogram(pos_grad_norm.cpu(),num_bins=100),
            #'pos_grad_2d_norm': wandb.Histogram(pos_grad_2d_norm.cpu(),num_bins=100),
            'pos_grad_norm_max': pos_grad_norm.cpu().max(),
            #'pos_grad_2d_norm_max': pos_grad_2d_norm.cpu().max(),
            'GAS size': self.gas_size,
            'FPS': self.fps
            }

GaussianModel.from_dataset = staticmethod(initialize)
GaussianModel.from_checkpoint = staticmethod(load_checkpoint)
