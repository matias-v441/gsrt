from typing import Any, Dict

import torch
from torch import nn
import numpy as np

import random
import warnings

from nerfbaselines import (
    cameras,Method, MethodInfo, ModelInfo, RenderOutput, Cameras, camera_model_to_int, Dataset
)

from random import randint

#from scene import GaussianModel # type: ignore
from scene.dataset_readers import storePly, fetchPly  # type: ignore
from utils.loss_utils import l1_loss, ssim  # type: ignore
from utils.image_utils import psnr

from lpipsPyTorch import lpips  # type: ignore

from method import Rays
from method.training import Training
from method.initialization import initialize
from method.checkpoint import load_checkpoint

import lovely_tensors as lt
lt.monkey_patch()


class GSRTMethod(Method):

    def __init__(self, *,
                checkpoint: str = None, 
                train_dataset: Dataset = None,
                test_dataset: Dataset = None,
                config_overrides: Dict[str, Any] = None):
        super().__init__()

        self.cfg = config_overrides
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        if checkpoint is not None:
            self.model = load_checkpoint(self.cfg)
        else:
            self.model = initialize(self.cfg, self.train_dataset)
        if train_dataset is not None:
            self.training = Training(self.cfg, self.model)
            self.viewpoint_ids = torch.arange(len(self.train_dataset['cameras']))

    def transform_gt_image(self, gt_image: torch.Tensor) -> torch.Tensor:
        gt_image = gt_image.float() / 255
        bg = np.array([1, 1, 1]) if self.model.white_background else np.array([0, 0, 0])
        if gt_image.shape[2] == 4:
            #warnings.warn("Alpha channel found in ground truth image, applying alpha compositing with background color.") 
            gt_image_alpha = gt_image[:, :, 3:4]
            gt_image = gt_image[:, :, :3] * gt_image_alpha + (1 - gt_image_alpha) * bg
        gt_image = gt_image[:, :, :3]
        return gt_image.float().reshape(-1, 3).cuda()

    def train_iteration(self, step: int) -> Dict[str, float]:

        # if not self._viewpoint_stack:
        #     loadCam.was_called = False  # type: ignore
        #     self._viewpoint_stack = self.scene.getTrainCameras().copy()
        #     if any(not getattr(cam, "_patched", False) for cam in self._viewpoint_stack):
        #         raise RuntimeError("could not patch loadCam!")
        # viewpoint_cam = self._viewpoint_stack.pop(randint(0, len(self._viewpoint_stack) - 1))
        random.seed(step)
        vp_id = self.viewpoint_ids[randint(0, self.viewpoint_ids.shape[0] - 1)] 
        train_cameras = self.train_dataset['cameras']
        #cameras_th = train_cameras.apply(lambda x, _: torch.from_numpy(x).contiguous().cuda())
        #camera_th = cameras_th.__getitem__(vp_id)
        def fit_distortion(x,name):
            if x is not None and name == "distortion_parameters":
                x = x[:,None,None,:]
            return torch.from_numpy(x).contiguous().cuda()
        cameras_th = train_cameras.apply(fit_distortion)
        camera_th = cameras_th.__getitem__(vp_id)
        xy = cameras.get_image_pixels(camera_th.image_sizes)

        ray_origins, ray_directions = cameras.get_rays(camera_th, xy[None])
        # ray_origins = torch.from_numpy(ray_origins).contiguous().cuda()
        # ray_directions = torch.from_numpy(ray_directions).contiguous().cuda()
        res_x, res_y = camera_th.image_sizes
        ray_origins = ray_origins.float().squeeze()
        ray_directions = ray_directions.float().squeeze()

        gt_image = self.transform_gt_image(torch.from_numpy(self.train_dataset['images'][vp_id]))

        if self.cfg.use_batched_rays:
            batch = torch.randint(0,ray_origins.shape[1],(2**16,),device="cuda")
            batch_res = 2**8
            rays = Rays(origins=ray_origins[batch].contiguous(), directions=ray_directions[batch].contiguous(),
                         res_x=batch_res.item(), res_y=batch_res.item())
            gt_image=gt_image[batch]
        else:  
            rays = Rays(origins=ray_origins.contiguous(), directions=ray_directions.contiguous(),
                         res_x=res_x.item(), res_y=res_y.item())

        out = self.training.step(t_step=step, rays=rays, gt_image=gt_image)

        image = out["image"]

        psnr = 10 * torch.log10(1 / torch.mean((image - gt_image) ** 2))
        
        return {"loss":out["loss"],"vp_id":vp_id,
                "out_image": image.detach().cpu().reshape(res_y,res_x,3).numpy(),
                "densif_stats":self.training.densif_strategy.densif_stats,
                "psnr":psnr}
    

    def test_iteration(self, step: int) -> Dict[str, float]:
        test_cameras = self.test_dataset['cameras']
        
        image = torch.from_numpy(self.render(test_cameras, options={"vp_id":step})["color"]).cuda()
        res_y, res_x = image.shape[:2]
        
        gt_image = self.transform_gt_image(torch.from_numpy(self.test_dataset['images'][step]))

        image  = image.reshape(1,res_y,res_x,3).permute(0,3,1,2)
        gt_image = gt_image.reshape(1,res_y,res_x,3).permute(0,3,1,2)

        ssim_value = ssim(image, gt_image)
        psnr_value = 10 * torch.log10(1 / torch.mean((image - gt_image) ** 2))
        lpips_value = lpips(image, gt_image, net_type='vgg')
        return ssim_value.item(), psnr_value.item(), lpips_value.item()

    
    def gradcheck(self, vp_id=0) -> bool:
        test_cameras = self.test_dataset['cameras']
        def fit_distortion(x,name):
            if x is not None and name == "distortion_parameters":
                x = x[:,None,None,:]
            return torch.from_numpy(x).contiguous().cuda()
        cameras_th = test_cameras.apply(fit_distortion)
        camera_th = cameras_th.__getitem__(vp_id)
        xy = cameras.get_image_pixels(camera_th.image_sizes)

        ray_origins, ray_directions = cameras.get_rays(camera_th, xy[None])
        res_x, res_y = camera_th.image_sizes
        ray_origins = ray_origins.float().squeeze()
        ray_directions = ray_directions.float().squeeze()

        gt_image = self.transform_gt_image(torch.from_numpy(self.test_dataset['images'][vp_id]))

        rays = Rays(origins=ray_origins.contiguous(), directions=ray_directions.contiguous(),
                         res_x=res_x.item(), res_y=res_y.item())

        def loss_fn(img, gt_image):
            return l1_loss(img, gt_image)
        return self.model.gradcheck(rays, gt_image, loss_fn, eps=1e-3, atol=3e-4, rtol=5e-2, nondet_tol=1e-2, fast_mode=True, raise_exception=True)


    @torch.no_grad()
    def render(self, camera : Cameras, *, options=None):
        def fit_distortion(x,name):
            #if x is not None and name == "distortion_parameters":
            #    x = x[:,None,None,:]
            return torch.from_numpy(x).contiguous().cuda()
        camera_th = camera.apply(fit_distortion)
        vp_id = 0 if options is None else options.get('vp_id',0)
        if len(camera_th) != 1:
            camera_th = camera_th.__getitem__(vp_id)
        xy = cameras.get_image_pixels(camera_th.image_sizes)
        ray_origins, ray_directions = cameras.get_rays(camera_th, xy[None])
        res_x, res_y = camera_th.image_sizes
        rays = Rays(origins=ray_origins.contiguous(), directions=ray_directions.contiguous(),
                         res_x=res_x, res_y=res_y)
        self.model.eval()
        img = self.model.forward(rays)
        return {"color":img.detach().cpu().reshape(res_y,res_x,3).numpy()}

    def save(self, path):
        self.model.save(path)

    @classmethod
    def get_method_info(cls):
        return {
            # Method ID is provided by the registry
            "method_id": "",  

            # Supported camera models (e.g., pinhole, opencv, ...)
            "supported_camera_models": frozenset(("pinhole","opencv_fisheye")),

            # Features required for training (e.g., color, points3D_xyz, ...)
            "required_features": frozenset(("color","points3D_xyz")),

            # Declare supported outputs
            "supported_outputs": ("color","transmittance","debug_map_0","debug_map_1"),
        }
    

    def get_info(self) -> ModelInfo:
        return {
            **self.get_method_info()
        }
