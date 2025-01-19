from typing import Any, Dict

import torch
from torch import nn
import numpy as np

import dataclasses
import warnings
import random
import itertools
import shlex
import logging
import copy
from typing import Optional
import os
import tempfile
import numpy as np
from PIL import Image

from nerfbaselines import (
    cameras,Method, MethodInfo, ModelInfo, RenderOutput, Cameras, camera_model_to_int, Dataset
)
import shlex

from argparse import ArgumentParser

from random import randint

from utils.general_utils import PILtoTorch  # type: ignore
from arguments import ModelParams, PipelineParams, OptimizationParams #  type: ignore
from scene import GaussianModel # type: ignore
import scene.dataset_readers  # type: ignore
from scene.dataset_readers import SceneInfo, getNerfppNorm, focal2fov  # type: ignore
from scene.dataset_readers import CameraInfo as _old_CameraInfo  # type: ignore
from scene.dataset_readers import storePly, fetchPly  # type: ignore
from utils.general_utils import safe_state, build_rotation, get_expon_lr_func, inverse_sigmoid  # type: ignore
from utils.graphics_utils import fov2focal  # type: ignore
from utils.loss_utils import l1_loss, ssim  # type: ignore
from utils.sh_utils import SH2RGB  # type: ignore
from scene import Scene, sceneLoadTypeCallbacks  # type: ignore
from utils import camera_utils  # type: ignore

from extension import GaussiansTracer

def flatten_hparams(hparams, *, separator: str = "/", _prefix: str = ""):
    flat = {}
    if dataclasses.is_dataclass(hparams):
        hparams = {f.name: getattr(hparams, f.name) for f in dataclasses.fields(hparams)}
    for k, v in hparams.items():
        if _prefix:
            k = f"{_prefix}{separator}{k}"
        if isinstance(v, dict) or dataclasses.is_dataclass(v):
            flat.update(flatten_hparams(v, _prefix=k, separator=separator).items())
        else:
            flat[k] = v
    return flat


def getProjectionMatrixFromOpenCV(w, h, fx, fy, cx, cy, znear, zfar):
    z_sign = 1.0
    P = torch.zeros((4, 4))
    P[0, 0] = 2.0 * fx / w
    P[1, 1] = 2.0 * fy / h
    P[0, 2] = (2.0 * cx - w) / w
    P[1, 2] = (2.0 * cy - h) / h
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

#
# Patch Gaussian Splatting to include sampling masks
# Also, fix cx, cy (ignored in gaussian-splatting)
#
# Patch loadCam to include sampling mask
_old_loadCam = camera_utils.loadCam
def loadCam(args, id, cam_info, resolution_scale):
    camera = _old_loadCam(args, id, cam_info, resolution_scale)

    sampling_mask = None
    if cam_info.sampling_mask is not None:
        sampling_mask = PILtoTorch(cam_info.sampling_mask, (camera.image_width, camera.image_height))
    setattr(camera, "sampling_mask", sampling_mask)
    setattr(camera, "_patched", True)

    # Fix cx, cy (ignored in gaussian-splatting)
    camera.focal_x = fov2focal(cam_info.FovX, camera.image_width)
    camera.focal_y = fov2focal(cam_info.FovY, camera.image_height)
    camera.cx = cam_info.cx
    camera.cy = cam_info.cy
    camera.projection_matrix = getProjectionMatrixFromOpenCV(
        camera.image_width, 
        camera.image_height, 
        camera.focal_x, 
        camera.focal_y, 
        camera.cx, 
        camera.cy, 
        camera.znear, 
        camera.zfar).transpose(0, 1).cuda()
    camera.full_proj_transform = (camera.world_view_transform.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0))).squeeze(0)

    return camera
camera_utils.loadCam = loadCam


# Patch CameraInfo to add sampling mask
class CameraInfo(_old_CameraInfo):
    def __new__(cls, *args, sampling_mask=None, cx, cy, **kwargs):
        self = super(CameraInfo, cls).__new__(cls, *args, **kwargs)
        self.sampling_mask = sampling_mask
        self.cx = cx
        self.cy = cy
        return self
scene.dataset_readers.CameraInfo = CameraInfo


def _load_caminfo(idx, pose, intrinsics, image_name, image_size, image=None, image_path=None, sampling_mask=None, scale_coords=None):
    pose = np.copy(pose)
    pose = np.concatenate([pose, np.array([[0, 0, 0, 1]], dtype=pose.dtype)], axis=0)
    pose = np.linalg.inv(pose)
    R = pose[:3, :3]
    T = pose[:3, 3]
    if scale_coords is not None:
        T = T * scale_coords
    R = np.transpose(R)

    width, height = image_size
    fx, fy, cx, cy = intrinsics
    if image is None:
        image = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    return CameraInfo(
        uid=idx, R=R, T=T, 
        FovX=focal2fov(float(fx), float(width)),
        FovY=focal2fov(float(fy), float(height)),
        image=image, image_path=image_path, image_name=image_name, 
        width=int(width), height=int(height),
        sampling_mask=sampling_mask,
        cx=cx, cy=cy)


def _config_overrides_to_args_list(args_list, config_overrides):
    for k, v in config_overrides.items():
        if str(v).lower() == "true":
            v = True
        if str(v).lower() == "false":
            v = False
        if isinstance(v, bool):
            if v:
                if f'--no-{k}' in args_list:
                    args_list.remove(f'--no-{k}')
                if f'--{k}' not in args_list:
                    args_list.append(f'--{k}')
            else:
                if f'--{k}' in args_list:
                    args_list.remove(f'--{k}')
                else:
                    args_list.append(f"--no-{k}")
        elif f'--{k}' in args_list:
            args_list[args_list.index(f'--{k}') + 1] = str(v)
        else:
            args_list.append(f"--{k}")
            args_list.append(str(v))


def _convert_dataset_to_gaussian_splatting(dataset: Optional[Dataset], tempdir: str, white_background: bool = False, scale_coords=None):
    if dataset is None:
        return SceneInfo(None, [], [], nerf_normalization=dict(radius=None, translate=None), ply_path=None)
    assert np.all(dataset["cameras"].camera_models == camera_model_to_int("pinhole")), "Only pinhole cameras supported"

    cam_infos = []
    for idx, extr in enumerate(dataset["cameras"].poses):
        del extr
        intrinsics = dataset["cameras"].intrinsics[idx]
        pose = dataset["cameras"].poses[idx]
        image_path = dataset["image_paths"][idx] if dataset["image_paths"] is not None else f"{idx:06d}.png"
        image_name = (
            os.path.relpath(str(dataset["image_paths"][idx]), str(dataset["image_paths_root"])) if dataset["image_paths"] is not None and dataset["image_paths_root"] is not None else os.path.basename(image_path)
        )

        w, h = dataset["cameras"].image_sizes[idx]
        im_data = dataset["images"][idx][:h, :w]
        assert im_data.dtype == np.uint8, "Gaussian Splatting supports images as uint8"
        if white_background and im_data.shape[-1] == 4:
            bg = np.array([1, 1, 1])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + (1 - norm_data[:, :, 3:4]) * bg
            im_data = np.array(arr * 255.0, dtype=np.uint8)
        if not white_background and dataset["metadata"].get("id") == "blender":
            warnings.warn("Blender scenes are expected to have white background. If the background is not white, please set white_background=True in the dataset loader.")
        elif white_background and dataset["metadata"].get("id") != "blender":
            warnings.warn("white_background=True is set, but the dataset is not a blender scene. The background may not be white.")
        image = Image.fromarray(im_data)
        sampling_mask = None
        if dataset["sampling_masks"] is not None:
            sampling_mask = Image.fromarray((dataset["sampling_masks"][idx] * 255).astype(np.uint8))

        cam_info = _load_caminfo(
            idx, pose, intrinsics, 
            image_name=image_name, 
            image_path=image_path,
            image_size=(w, h),
            image=image,
            sampling_mask=sampling_mask,
            scale_coords=scale_coords,
        )
        cam_infos.append(cam_info)

    cam_infos = sorted(cam_infos.copy(), key=lambda x: x.image_name)
    nerf_normalization = getNerfppNorm(cam_infos)

    points3D_xyz = dataset["points3D_xyz"]
    if scale_coords is not None:
        points3D_xyz = points3D_xyz * scale_coords
    points3D_rgb = dataset["points3D_rgb"]
    if points3D_xyz is None and dataset["metadata"].get("id", None) == "blender":
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/dataset_readers.py#L221C4-L221C4
        num_pts = 100_000
        logging.info(f"generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        points3D_xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        points3D_rgb = (SH2RGB(shs) * 255).astype(np.uint8)

    storePly(os.path.join(tempdir, "scene.ply"), points3D_xyz, points3D_rgb)
    pcd = fetchPly(os.path.join(tempdir, "scene.ply"))
    scene_info = SceneInfo(point_cloud=pcd, train_cameras=cam_infos, test_cameras=[], nerf_normalization=nerf_normalization, ply_path=os.path.join(tempdir, "scene.ply"))
    return scene_info


class _TraceFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, setup, part_opac,part_xyz,part_scale,part_rot,part_sh,part_color):
        tracer = setup['tracer']
        tracer.load_gaussians(part_xyz,part_rot,part_scale,part_opac,part_sh,setup['sh_deg'],part_color)
        out = tracer.trace_rays(setup['ray_origins'],setup['ray_directions'],
                                setup['width'],setup['height'],
                                False,torch.tensor(0))
        ctx.setup = setup
        return out["radiance"]

    @staticmethod
    def backward(ctx, *grad_outputs):
        dout_dC = grad_outputs[0]
        setup = ctx.setup
        out = setup['tracer'].trace_rays(setup['ray_origins'],setup['ray_directions'],
                                         setup['width'],setup['height'],
                                         True,dout_dC)
        grad_xyz = out["grad_xyz"]
        grad_opacity = out["grad_opacity"][:,None]
        grad_scale = out["grad_scale"]
        grad_rot = out["grad_rot"]
        grad_sh = out["grad_sh"]
        grad_color = out["grad_color"]
        for grad,n in [(grad_xyz,"grad_xyz"),
                       (grad_opacity,"grad_opacity"),
                       (grad_scale,"grad_scale"),
                       (grad_rot,"grad_rot"),
                       (grad_sh,"grad_sh"),
                       (grad_color,"grad_color")]:
            nan_mask = torch.isnan(grad)
            if torch.any(nan_mask):
                print(f"found NaN grad in {n}")
            grad[nan_mask] = 0.
        return None,grad_opacity,grad_xyz,grad_scale,grad_rot,grad_sh,grad_color

        
def trace_function(*args):
    out = _TraceFunction.apply(*args)
    return out


class GSRTMethod(Method):

    def __init__(self, *,
                checkpoint: str = None, 
                train_dataset: Dataset = None,
                config_overrides: Dict[str, Any] = None):
        super().__init__()

        self.train_dataset = train_dataset
        self.hparams = {
            "init_num_points": 100000,
        }

        self.setup_functions()

        self.checkpoint = checkpoint

        self.scene_extent = None
        if self.train_dataset is not None:
            self.scene_extent = self.train_dataset['metadata']['expected_scene_scale']
        self._3dgs_data = False
        if self.checkpoint is None:
            self.initialize_for_blender()
        else:
            self._3dgs_data = True
            self.load_3dgs_checkpoint()

        self.tracer = GaussiansTracer(torch.device("cuda:0"))


    def setup_functions(self):
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid


    def initialize_for_blender(self):

        torch.manual_seed(0)
        n = self.hparams['init_num_points']
        scene_min = torch.tensor([-1.,-1.,-1.])*self.scene_extent/2
        scene_max = torch.tensor([1.,1.,1.])*self.scene_extent/2
        self._xyz = ((torch.rand(n,3)-.5)*(scene_max-scene_min)*.5+(scene_max+scene_min)*.5).cuda()
        self._scaling = torch.log(torch.ones(1,3).repeat(n,1)*0.01).cuda()
        self._rotation = torch.hstack([torch.ones(n,1),torch.zeros(n,3)]).cuda()
        self._opacity = self.inverse_opacity_activation(torch.ones(n,1)*0.01).cuda()
        self._features_dc = torch.zeros(n,1,3).cuda()
        self._features_rest = torch.zeros(n,15,3).cuda()
        self.active_sh_degree = 0
        self.max_sh_degree = 3
        self._color = (torch.ones(n,3)*torch.tensor([0.,1.,0.])).cuda() 

        #self.scene = self._build_scene(self.train_dataset)

    def load_checkpoint(self):
        ch = torch.load(self.checkpoint)
        self._xyz = ch['xyz']
        self._scaling = ch['scaling']
        self._rotation = ch['rotation']
        self._opacity = ch['opacity']
        self._features_dc = ch['f_dc']
        self._features_rest = ch['f_rest']
        self.active_sh_degree = ch['sh_deg']
        self._color = ch['color']
    

    def load_3dgs_checkpoint(self):
        gaussians,it = torch.load(self.checkpoint)
        self._xyz = gaussians[1].detach().cuda()
        self._scaling = gaussians[4].detach().cuda()
        self._rotation = gaussians[5].detach().cuda()
        self._opacity = gaussians[6].detach().cuda()
        self._features_dc = gaussians[2].detach().cuda()
        self._features_rest = gaussians[3].detach().cuda()
        self.active_sh_degree = gaussians[0]
        self._color = (torch.ones(self._xyz.shape[0],3)*torch.tensor([0.,1.,0.])).cuda() 

    
    def _build_scene(self, dataset):
        opt = copy.copy(self.dataset)
        with tempfile.TemporaryDirectory() as td:
            os.mkdir(td + "/sparse")
            opt.source_path = td  # To trigger colmap loader
            opt.model_path = td if dataset is not None else str(self.checkpoint)
            backup = sceneLoadTypeCallbacks["Colmap"]
            try:
                info = self.get_info()
                def colmap_loader(*args, **kwargs):
                    del args, kwargs
                    return _convert_dataset_to_gaussian_splatting(dataset, td, white_background=self.dataset.white_background, scale_coords=self.dataset.scale_coords)
                sceneLoadTypeCallbacks["Colmap"] = colmap_loader
                loaded_step = info.get("loaded_step")
                scene = Scene(opt, self.gaussians, load_iteration=str(loaded_step) if dataset is None else None)
                # NOTE: This is a hack to match the RNG state of GS on 360 scenes
                _tmp = list(range((len(next(iter(scene.train_cameras.values()))) + 6) // 7))
                random.shuffle(_tmp)
                return scene
            finally:
                sceneLoadTypeCallbacks["Colmap"] = backup


    def rendering_setup(self):
        self.retained_scaling = self.get_scaling
        self.retained_opacity = self.get_opacity
        self.retained_sh = self.get_features
        self.tracer.load_gaussians(self._xyz,self._rotation,self.retained_scaling,
                                   self.retained_opacity,self.retained_sh,
                                   self.active_sh_degree,self._color)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self):

        self.viewpoint_ids = torch.arange(len(self.train_dataset['cameras']))

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") 

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 1500#500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.percent_dense = 0.01

        position_lr_init = 0.00016
        position_lr_final = 0.0000016
        position_lr_delay_mult = 0.01
        position_lr_max_steps = 30_000
        feature_lr = 0.0025
        opacity_lr = 0.025
        scaling_lr = 0.005
        rotation_lr = 0.001

        self.spatial_lr_scale = 0

        l = [
            {'params': [self._xyz], 'lr': position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': rotation_lr, "name": "rotation"}
        ]
        for p in l:
            p['params'][0].requires_grad_(True)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=position_lr_init*self.spatial_lr_scale,
                                                    lr_final=position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=position_lr_delay_mult,
                                                    max_steps=position_lr_max_steps)
        
        self.optimizer = torch.optim.AdamW(l,lr=0.0, eps=1e-15)
        
    
    def update_learning_rate(self, iteration):

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_scaling(self):
        if self._3dgs_data:
            return self.scaling_activation(self._scaling)*1.5
        return self.scaling_activation(self._scaling)

    @property
    def get_xyz(self):
        return self._xyz 

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1) 


    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
   

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        #self.max_radii2D = self.max_radii2D[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors 


    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)


    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation) 


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            #big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

  
    #def add_densification_stats(self, viewspace_point_tensor, update_filter):
    #    self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
    #    self.denom[update_filter] += 1 

    def add_densification_stats(self):
        self.xyz_gradient_accum += torch.linalg.norm(self._xyz.grad,dim=1,keepdim=True)
        self.denom += 1
        

    def train_iteration(self, step: int) -> Dict[str, float]:

        iteration = step+1

        self.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            self.oneupSHdegree()

        # if not self._viewpoint_stack:
        #     loadCam.was_called = False  # type: ignore
        #     self._viewpoint_stack = self.scene.getTrainCameras().copy()
        #     if any(not getattr(cam, "_patched", False) for cam in self._viewpoint_stack):
        #         raise RuntimeError("could not patch loadCam!")
        # viewpoint_cam = self._viewpoint_stack.pop(randint(0, len(self._viewpoint_stack) - 1))
        vp_id = self.viewpoint_ids[randint(0, self.viewpoint_ids.shape[0] - 1)] 
        train_cameras = self.train_dataset['cameras']
        cameras_th = train_cameras.apply(lambda x, _: torch.from_numpy(x).contiguous().cuda())
        camera_th = cameras_th.__getitem__(vp_id)
        xy = cameras.get_image_pixels(camera_th.image_sizes)
        ray_origins, ray_directions = cameras.get_rays(camera_th, xy[None])
        res_x, res_y = camera_th.image_sizes
        setup = {
            'tracer':self.tracer,
            'ray_origins':ray_origins.float().squeeze(0).contiguous(),
            'ray_directions':ray_directions.float().squeeze(0).contiguous(),
            'width':res_x,
            'height':res_y,
            'sh_deg':self.active_sh_degree
        }
        gt_image = torch.from_numpy(self.train_dataset['images'][vp_id][:,:,:3]).reshape(res_x*res_y,3).cuda()
        image = trace_function(setup, self.get_opacity, self._xyz, 
                                   self.get_scaling, self._rotation,
                                   self.get_features,
                                   self._color)
        
        Ll1 = l1_loss(image, gt_image)
        # ssim_value = ssim(image.reshape(res_y,res_x,3), gt_image.reshape(res_y,res_x,3))
        loss = Ll1#(1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        with torch.no_grad():

            # Log
            if iteration % 10 == 0:
                print(f'Iter {iteration}, viewpoint {vp_id}, loss {loss.detach().item()}')

            # Densification
            if iteration < self.densify_until_iter:

                self.add_densification_stats()

                if iteration > self.densify_from_iter and step % self.densification_interval == 0:
                    size_threshold = 20 if step > self.opacity_reset_interval else None
                    self.densify_and_prune(self.densify_grad_threshold,0.005,self.scene_extent,size_threshold)

                if iteration % self.opacity_reset_interval == 0: #or (dataset.white_background and iteration == opt.densify_from_iter):
                    self.reset_opacity() 

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()


        return {"loss":loss.detach().item(),"vp_id":vp_id, "out_image":image.detach().cpu().reshape(res_y,res_x,3).numpy()}


    @torch.no_grad()
    def render(self, camera : Cameras, *, options=None):

        camera_th = camera.apply(lambda x, _: torch.from_numpy(x).contiguous().cuda())
        vp_id = 0 if options is None else options.get('vp_id',0)
        camera_th = camera_th.__getitem__(vp_id)
        xy = cameras.get_image_pixels(camera_th.image_sizes)
        ray_origins, ray_directions = cameras.get_rays(camera_th, xy[None])
        res_x, res_y = camera_th.image_sizes

        time_ms = 0
        nit = 1 if options is None else options.get('num_avg_it',1)
        for i in range(nit):
            res = self.tracer.trace_rays(ray_origins.float().squeeze(0).contiguous(),
                                         ray_directions.float().squeeze(0).contiguous(),
                                         res_x, res_y,
                                         False,torch.tensor(0.))
            time_ms += res["time_ms"]
        time_ms /= nit
        
        color = res["radiance"].cpu().reshape(res_y,res_x,3).numpy()
        transmittance = res["transmittance"].cpu().reshape(res_y,res_x)[:,:,None].repeat(1,1,3).numpy()
        debug_map_0 = res["debug_map_0"].cpu().reshape(res_x,res_y,3).numpy()
        debug_map_1 = res["debug_map_1"].cpu().reshape(res_x,res_y,3).numpy()
        time_ms = res["time_ms"]
        num_its = res["num_its"]
        
        return {
            "color": color, #+ transmittance,
            "transmittance": transmittance,
            "debug_map_0": debug_map_0,
            "debug_map_1": debug_map_1,
            "time_ms": time_ms,
            "num_its": num_its,
            "res_xy": (res_x,res_y) 
        }

    
    def save(self, path):
        torch.save({'xyz':self._xyz.detach(),
                    'f_dc':self._features_dc.detach(),
                    'f_rest':self._features_rest.detach(),
                    'opacity':self._opacity.detach(),
                    'scaling':self._scaling.detach(),
                    'rotation':self._rotation.detach(),
                    'color':self._color.detach(),
                    'sh_deg':self.active_sh_degree
                    },f'{path}/checkpoint.pt')


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
