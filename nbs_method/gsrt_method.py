from typing import Any, Dict

import torch
from torch import nn
import numpy as np

import dataclasses
import random
import itertools
import shlex
import copy
import os
import tempfile
import numpy as np

from nerfbaselines import (
    cameras,Method, MethodInfo, ModelInfo, RenderOutput, Cameras, camera_model_to_int, Dataset
)
import shlex

from argparse import ArgumentParser

from random import randint

from arguments import ModelParams, PipelineParams, OptimizationParams #  type: ignore
#from scene import GaussianModel # type: ignore
from scene.dataset_readers import storePly, fetchPly  # type: ignore
from utils.loss_utils import l1_loss, ssim  # type: ignore
from utils.image_utils import psnr

from lpipsPyTorch import lpips  # type: ignore

from extension import GaussiansTracer
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
            self.training = Training(self.cfg, self.model, train_dataset)
            self.viewpoint_ids = torch.arange(len(self.train_dataset['cameras']))
        self.tracer = GaussiansTracer()


    def train_iteration(self, step: int) -> Dict[str, float]:

        iteration = step
        self.iteration = iteration

        # if not self._viewpoint_stack:
        #     loadCam.was_called = False  # type: ignore
        #     self._viewpoint_stack = self.scene.getTrainCameras().copy()
        #     if any(not getattr(cam, "_patched", False) for cam in self._viewpoint_stack):
        #         raise RuntimeError("could not patch loadCam!")
        # viewpoint_cam = self._viewpoint_stack.pop(randint(0, len(self._viewpoint_stack) - 1))
        random.seed(iteration)
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
    
        gt_image = torch.from_numpy(self.train_dataset['images'][vp_id])/255
        bg = np.array([1, 1, 1]) if self.cfg.white_bg else np.array([0, 0, 0])
        if gt_image.shape[2] == 4: # has alpha
            gt_image_alpha = gt_image[:, :, 3:4]
            gt_image = gt_image[:, :, :3]*gt_image_alpha + (1 - gt_image_alpha)*bg
        gt_image = gt_image[:,:,:3]
        gt_image = gt_image.float().reshape(res_x*res_y,3).cuda()

        if self.cfg.use_batched_rays:
            batch = torch.randint(0,ray_origins.shape[1],(2**16,),device="cuda")
            batch_res = 2**8
            rays = Rays(origins=ray_origins[batch].contiguous(), directions=ray_directions[batch].contiguous(),
                         res_x=batch_res, res_y=batch_res)
            gt_image=gt_image[batch]
        else:  
            rays = Rays(origins=ray_origins.contiguous(), directions=ray_directions.contiguous(),
                         res_x=res_x, res_y=res_y)

        out = self.training.step(tracer=self.tracer, t_step=step, rays=rays, gt_image=gt_image)

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
        
        gt_image = torch.from_numpy(self.test_dataset['images'][step])/255
        bg = np.array([1, 1, 1]) if self.white_background else np.array([0, 0, 0])
        if gt_image.shape[2] == 4: # has alpha
            gt_image_alpha = gt_image[:, :, 3:4]
            gt_image = gt_image[:, :, :3]*gt_image_alpha + (1 - gt_image_alpha)*bg
        gt_image = gt_image[:,:,:3]
        gt_image = gt_image.float().reshape(res_x*res_y,3).cuda()

        image  = image.reshape(1,res_y,res_x,3).permute(0,3,1,2)
        gt_image = gt_image.reshape(1,res_y,res_x,3).permute(0,3,1,2)

        ssim_value = ssim(image, gt_image)
        psnr_value = 10 * torch.log10(1 / torch.mean((image - gt_image) ** 2))
        lpips_value = lpips(image, gt_image)
        return ssim_value.item(), psnr_value.item(), lpips_value.item()


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

        if not self.tracer.has_gaussians():
            self.tracer.load_gaussians(self.model.xyz, self.model.rotation, self.model.scaling, self.model.opacity,
                                    self.model.features, self.model.active_sh_degree, {"type":"optix"})
        out = self.tracer.trace_fwd(ray_origins.contiguous(),ray_directions.contiguous(),
                               res_x, res_y, self.cfg.white_bg)
        
        color = out["radiance"].detach().cpu().reshape(res_y,res_x,3).numpy()
        return {"color":color}

    
    def save(self, path, it):
        torch.save({'xyz':self.model.xyz.detach(),
                    'f_dc':self.model.features_dc.detach(),
                    'f_rest':self.model.features_rest.detach(),
                    'opacity':self.model.opacity.detach(),
                    'scaling':self.model.scaling.detach(),
                    'rotation':self.model.rotation.detach(),
                    'color':self.model.color.detach(),
                    'sh_deg':self.model.active_sh_degree
                    },f'{path}/checkpoint_{it}.pt')
        
    def get_data(self):
        return (self._xyz.detach(),
                self._features_dc.detach(),
                self._features_rest.detach(),
                self._opacity.detach(),
                self._scaling.detach(),
                self._rotation.detach(),
                self._color.detach(),
                self.active_sh_degree)

    def set_data(self,data):
            (self._xyz,
            self._features_dc,
            self._features_rest,
            self._opacity,
            self._scaling,
            self._rotation,
            self._color,
            self.active_sh_degree) = data


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
