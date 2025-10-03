from typing import Any, Dict
from nerfbaselines import Method
from nerfbaselines import Dataset, ModelInfo
from nerfbaselines import Cameras, CameraModel
from nerfbaselines import cameras
import torch

from extension import GaussiansTracer,TracerCustom

class GSRTMethod(Method):
    def __init__(self, *,
                checkpoint: str = None, 
                train_dataset: Dataset = None,
                config_overrides: Dict[str, Any] = None):
        super().__init__()

        self.device = torch.device("cuda:0")
        self.use_custom = True
        if self.use_custom:
            self.tracer = TracerCustom(self.device)
            cf_type = 2
            K_I = 2.
            K_T = 3.
            k1 = 3.
            k2 = 1.25
            self.tracer.set_parameters(cf_type,K_I,K_T,k1,k2)
        else:
            self.tracer = GaussiansTracer(self.device)

        self.checkpoint = checkpoint

        if checkpoint is None:
            print("No checkpoint provided!")
            return
        print("Loaded checkpoint ", checkpoint)

        gaussians,it = torch.load(checkpoint)
        gs_xyz = gaussians[1].detach().cpu()
        gs_scaling = torch.exp(gaussians[4].detach().cpu())*1.5
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
        custom = False
        T = torch.tensor([ 0.01610, -0.89595,  0.44386, -1.09488,
                            -0.99987, -0.01443,  0.00715, -0.01763,
                            0.00000, -0.44392, -0.89607,  2.21032]).reshape(3,4)

        def f(x,s):
            if s == "poses" and custom:
                return T.contiguous().to(self.device)
            return torch.from_numpy(x).contiguous().to(self.device)
        #camera_th = camera.apply(lambda x, _: torch.from_numpy(x).contiguous().to(self.device))
        #camera_th = camera.apply(lambda x, _: T.contiguous().to(self.device))
        camera_th = camera.apply(f)
        xy = cameras.get_image_pixels(camera_th.image_sizes)
        ray_origins, ray_directions = cameras.get_rays(camera_th, xy[None])
        res_x, res_y = camera.item().image_sizes
        print(res_x,res_y)


        # import math
        # ntiles = 2
        # tilew = math.ceil(res_x/ntiles)
        # tileh = math.ceil(res_y/ntiles)
        # ray_origins_sq = ray_origins.reshape((1,res_y,res_x,3))
        # ray_directions_sq = ray_directions.reshape((1,res_y,res_x,3))
        # color = torch.zeros((res_y,res_x,3))
        # transmittance = torch.zeros((res_y,res_x,1))   
        # time_ms = 0
        # nit = 100
        # for _ in range(nit):
        #     for i in range(ntiles):
        #         for j in range(ntiles):
        #             xs = min(tilew*i,res_x)
        #             xe = min(tilew*(i+1),res_x)
        #             sx = xe-xs
        #             ys = min(tileh*j,res_y)
        #             ye = min(tileh*(j+1),res_y)
        #             sy = ye-ys
        #             tile_ray_orig = ray_origins_sq[:,ys:ye,xs:xe].reshape(1,sx*sy,3)
        #             tile_ray_dir = ray_directions_sq[:,ys:ye,xs:xe].reshape(1,sx*sy,3)
        #             res = self.tracer.trace_rays(tile_ray_orig.float().squeeze(0).contiguous(),tile_ray_dir.float().squeeze(0).contiguous())
        #             color[ys:ye,xs:xe] = res["radiance"].cpu().reshape(sy,sx,3)
        #             transmittance[ys:ye,xs:xe] = res["transmittance"].cpu().reshape(sy,sx,1)
        #             time_ms += res["time_ms"]
        # time_ms /= nit
        # print(1000/time_ms, res_x, res_y)
        # color = color.numpy()
        # transmittance = transmittance.numpy()
        # debug_map_0 = color
        # debug_map_1 = color
        

        time_ms = 0
        nit = 1
        for i in range(nit):
            if not self.use_custom:
                res = self.tracer.trace_rays(ray_origins.float().squeeze(0).contiguous(),
                                             ray_directions.float().squeeze(0).contiguous(),
                                             res_x, res_y,
                                             False)
            else:
                draw_kd = False
                tracer_type = 6
                res = self.tracer.trace_rays(ray_origins.float().squeeze(0).contiguous(),
                                            ray_directions.float().squeeze(0).contiguous(),
                                            res_x,res_y,
                                            tracer_type,draw_kd,False,torch.tensor(0.))
            time_ms += res["time_ms"]
            #print(i,time_ms)
            #print(res["num_its"])
        time_ms /= nit
        
        color = res["radiance"].cpu().reshape(res_y,res_x,3).numpy()
        transmittance = res["transmittance"].cpu().reshape(res_y,res_x)[:,:,None].repeat(1,1,3).numpy()
        debug_map_0 = res["debug_map_0"].cpu().reshape(res_x,res_y,3).numpy()
        debug_map_1 = res["debug_map_1"].cpu().reshape(res_x,res_y,3).numpy()
        time_ms = res["time_ms"]
        num_its = res["num_its"]
        #print(num_its)
        #print(1000/time_ms, num_its/(res_x*res_y), res_x,res_y)
        
        return {
            "color": color,# + transmittance,
            "transmittance": transmittance,
            "debug_map_0": debug_map_0,
            "debug_map_1": debug_map_1,
        }