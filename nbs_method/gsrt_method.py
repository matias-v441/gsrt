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
#lt.monkey_patch()

import method.cam_grut as camgrut


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
        def fit_distortion(x,name):
            if x is not None and name == "distortion_parameters":
                x = x[:,None,None,:]
            return torch.from_numpy(x).contiguous().cuda()
        if self.train_dataset:
            self.train_cameras_th = self.train_dataset['cameras'].apply(fit_distortion)
        if self.test_dataset:
            self.test_cameras_th = self.test_dataset['cameras'].apply(fit_distortion)
        if checkpoint is not None:
            self.model = load_checkpoint(self.cfg)
        else:
            self.model = initialize(self.cfg, self.train_dataset)
        if train_dataset is not None:
            self.training = Training(self.cfg, self.model)
            self.viewpoint_ids = None

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
        iter = self.training.start_iter+step
        random.seed(iter)
        torch.manual_seed(iter)
        batch_id = (iter-1)%len(self.train_cameras_th)
        if batch_id == 0 or step == 0:
            self.viewpoint_ids = torch.randperm(len(self.train_cameras_th))
        vp_id = self.viewpoint_ids[batch_id] 
        camera_th = self.train_cameras_th.__getitem__(vp_id)
        xy = cameras.get_image_pixels(camera_th.image_sizes)

        # fx,fy,cx,cy = camera_th.intrinsics
        # k1,k2,p1,p2,k3,k4 = camera_th.distortion_parameters.squeeze()
        # w,h = camera_th.image_sizes
        # ray_directions = camgrut.create_fisheye_camera(np.array([cx.item(),cy.item()]).astype(np.float32),
        #                                                np.array([fx.item(),fy.item()]).astype(np.float32),
        #                                                np.array([k1.item(),k2.item(),k3.item(),k4.item()]).astype(np.float32),
        #                                                 xy, xy.device, w.item(),h.item())
        # rotation = camera_th.poses[..., :3, :3]  # (..., 3, 3)
        # ray_directions = (ray_directions[..., None, :] * rotation).sum(-1)
        # ray_origins = torch.broadcast_to(camera_th.poses[..., :3, 3], ray_directions.shape)

        ray_origins, ray_directions = cameras.get_rays(camera_th, xy[None])

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

        with torch.no_grad():
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(out["image"].detach().reshape(1,rays.res_y,rays.res_x,3).squeeze().cpu().numpy())
            # plt.figure()
            # plt.imshow(gt_image.detach().reshape(1,rays.res_y,rays.res_x,3).squeeze().cpu().numpy())
            # plt.show()
            image = out["image"].detach()
            psnr = 10 * torch.log10(1 / torch.mean((image - gt_image) ** 2))
            metrics = {"loss":out["loss"],"vp_id":vp_id,
                "out_image": image.detach().cpu().reshape(res_y,res_x,3).numpy(),
                "densif_stats":self.training.densif_strategy.densif_stats,
                "psnr":psnr}
        return metrics
    

    def test_iteration(self, step: int) -> Dict[str, float]:

        cam_th = self.test_cameras_th.__getitem__(step%len(self.test_cameras_th))
        image_np = self.render(cam_th,patch_camera=False)["color"]
        image = torch.from_numpy(image_np).cuda()
        res_y, res_x = image.shape[:2]
        
        gt_image = self.transform_gt_image(torch.from_numpy(self.test_dataset['images'][step]))

        image  = image.reshape(1,res_y,res_x,3).permute(0,3,1,2)
        gt_image = gt_image.reshape(1,res_y,res_x,3).permute(0,3,1,2)

        ssim_value = ssim(image, gt_image)
        psnr_value = 10 * torch.log10(1 / torch.mean((image - gt_image) ** 2))
        lpips_value = lpips(image, gt_image, net_type='vgg')
        return ssim_value.item(), psnr_value.item(), lpips_value.item()

    
    def gradcheck(self, vp_id=0) -> bool:
        
        camera_th = self.test_cameras_th.__getitem__(vp_id)
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
    def render(self, camera : Cameras, *, options=None, patch_camera=True):
        def fit_distortion(x,name):
            if name == "distortion_parameters" and x is not None:
                x = x[...,None,None,:]
            if type(x) is not np.ndarray:
                return torch.tensor(x)
            return torch.from_numpy(x).contiguous().cuda()
        camera_th = camera.apply(fit_distortion) if patch_camera else camera
        if camera_th.distortion_parameters.numel()==0 and self.test_dataset is not None:
            tcam = self.test_cameras_th.__getitem__(0)
            if tcam.distortion_parameters.numel()!=0:
                print("adding distortion")
                from nerfbaselines._types import new_cameras
                camera_th = new_cameras(poses=camera_th.poses, 
                                        intrinsics=camera_th.intrinsics,
                                        camera_models=tcam.camera_models,
                                        image_sizes=camera_th.image_sizes,
                                        distortion_parameters=tcam.distortion_parameters,
                                        nears_fars=camera_th.nears_fars,
                                        metadata=camera_th.metadata)
                print(camera_th.distortion_parameters)
        xy = cameras.get_image_pixels(camera_th.image_sizes)

        # k1,k2,p1,p2,k3,k4 = camera_th.distortion_parameters.squeeze()
        # fx,fy,cx,cy = camera_th.intrinsics
        # w,h = camera_th.image_sizes
        # ray_directions = camgrut.create_fisheye_camera(np.array([cx.item(),cy.item()]).astype(np.float32),
        #                                                np.array([fx.item(),fy.item()]).astype(np.float32),
        #                                                np.array([k1.item(),k2.item(),k3.item(),k4.item()]).astype(np.float32),
        #                                                 xy, xy.device, w.item(),h.item())
        # ray_origins = torch.broadcast_to(camera_th.poses[..., :3, 3], ray_directions.shape)
        # rotation = camera_th.poses[..., :3, :3]  # (..., 3, 3)
        # ray_directions = (ray_directions[..., None, :] * rotation).sum(-1)
        ray_origins, ray_directions = cameras.get_rays(camera_th, xy[None])

        res_x, res_y = camera_th.image_sizes
        rays = Rays(origins=ray_origins.contiguous(), directions=ray_directions.contiguous(),
                         res_x=res_x, res_y=res_y)
        self.model.eval()
        img = torch.clamp(self.model.forward(rays),0.,1.)
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
