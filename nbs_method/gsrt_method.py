from typing import Any, Dict
from nerfbaselines import Method
from nerfbaselines._types import Dataset, ModelInfo
from nerfbaselines._types import Cameras, CameraModel
import torch

from extension import GaussiansTracer

class GSRTMethod(Method):
    def __init__(self, *,
                checkpoint: str = None, 
                train_dataset: Dataset = None,
                config_overrides: Dict[str, Any] = None):
        super().__init__()

        self.device = torch.device("cuda:0")
        self.tracer = GaussiansTracer(self.device)

        self.checkpoint = checkpoint

        if checkpoint is None:
            print("No checkpoint provided!")
            return
        print("Loaded checkpoint ", checkpoint)

        gaussians,it = torch.load(checkpoint)
        gs_xyz = gaussians[1].detach().cpu()
        gs_scaling = torch.exp(gaussians[4].detach().cpu())
        gs_rotation = gaussians[5].detach().cpu()

        #gs_opacity = torch.ones_like(gaussians[6].detach().cpu()) 
        gs_opacity = torch.sigmoid(gaussians[6].detach().cpu())

        gs_features_dc = gaussians[2].detach().cpu()
        gs_features_rest = gaussians[3].detach().cpu()
        gs_active_sh_degree = gaussians[0]

        gs_sh = torch.cat((gs_features_dc,gs_features_rest),dim=1).contiguous()

        self.tracer.load_gaussians(gs_xyz,gs_rotation,gs_scaling,gs_opacity,gs_sh,gs_active_sh_degree)

    def save(self, path):
        pass

    @classmethod
    def get_method_info(cls):
        return {
            # Method ID is provided by the registry
            "method_id": "",  

            # Supported camera models (e.g., pinhole, opencv, ...)
            "supported_camera_models": frozenset(("pinhole",)),

            # Features required for training (e.g., color, points3D_xyz, ...)
            "required_features": frozenset(("color",)),

            # Declare supported outputs
            "supported_outputs": ("color","transmittance","debug_map_0","debug_map_1"),
        }
    
    def get_info(self) -> ModelInfo:
        return {
            **self.get_method_info(),
            "loaded_checkpoint": self.checkpoint
        }
    
    def train_iteration(self, step: int) -> Dict[str, float]:
        return super().train_iteration(step)

    @torch.no_grad()
    def render(self, camera : Cameras, *, options=None):
        w,h = camera.image_sizes[0]
        res_x,res_y = w,h
        num_rays = res_x*res_y
        x,y=torch.meshgrid(torch.linspace(-w/2,w/2,res_x),torch.linspace(-h/2,h/2,res_y))
        focus = 1250.
        T = torch.from_numpy(camera.poses[0]).to(dtype=torch.float32)
        R = T[:,:3]
        t = T[:,3]
        c_im = torch.stack([x.flatten(),y.flatten(),-torch.ones(num_rays)*focus],dim=1)
        c_im /= c_im.norm(dim=1)[:,None]
        fx,fy,cx,cy = camera.intrinsics[0]
        c_im = c_im@R.T
        origin = -t
        ray_origins = origin.repeat(num_rays,1).to(self.device,dtype=torch.float32)
        ray_directions = c_im.to(self.device,dtype=torch.float32)
        res = self.tracer.trace_rays(ray_origins,ray_directions)
        color = res["radiance"].cpu().reshape(res_x,res_y,3).numpy()
        transmittance = res["transmittance"].cpu().reshape(res_x,res_y)[:,:,None].repeat(1,1,3).numpy()
        debug_map_0 = res["debug_map_0"].cpu().reshape(res_x,res_y,3).numpy()
        debug_map_1 = res["debug_map_1"].cpu().reshape(res_x,res_y,3).numpy()
        return {
            "color": color,
            "transmittance": transmittance,
            "debug_map_0": debug_map_0,
            "debug_map_1": debug_map_1,
        }