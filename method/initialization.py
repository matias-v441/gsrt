import torch
import numpy as np
import logging

from utils.sh_utils import SH2RGB,RGB2SH  # type: ignore
from utils.graphics_utils import fov2focal,BasicPointCloud  # type: ignore
from utils.general_utils import PILtoTorch  # type: ignore


import warnings
import os

from PIL import Image
from typing import Optional
from nerfbaselines import Dataset
from simple_knn._C import distCUDA2

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
def _patch_camera_utils():
    """Lazy patching of camera_utils to avoid circular imports"""
    from utils import camera_utils  # type: ignore
    import scene.dataset_readers  # type: ignore
    from scene.dataset_readers import CameraInfo as _old_CameraInfo  # type: ignore
    
    # Your patching code here
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
    
    # Patch CameraInfo
    class CameraInfo(_old_CameraInfo):
        def __new__(cls, *args, sampling_mask=None, cx=None, cy=None, **kwargs):
            self = super(CameraInfo, cls).__new__(cls, *args, **kwargs)
            self.sampling_mask = sampling_mask
            self.cx = cx
            self.cy = cy
            return self
    scene.dataset_readers.CameraInfo = CameraInfo
    
    return camera_utils, scene.dataset_readers


def _load_caminfo(idx, pose, intrinsics, image_name, image_size, image=None, image_path=None, sampling_mask=None, scale_coords=None):
    from scene.dataset_readers import SceneInfo, getNerfppNorm, focal2fov  # type: ignore
    camera_utils, scene_readers = _patch_camera_utils()
    CameraInfo = scene_readers.CameraInfo

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


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def _convert_dataset_to_gaussian_splatting(dataset: Optional[Dataset], tempdir: str, white_background: bool = False, scale_coords=None):

    from scene.dataset_readers import SceneInfo, getNerfppNorm, focal2fov  # type: ignore
    import warnings
    import os

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
        if im_data.shape[-1] == 4:
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
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
    print(f"nerf_normalization: {nerf_normalization}")
    
    points3D_xyz = dataset["points3D_xyz"]
    if scale_coords is not None:
        points3D_xyz = points3D_xyz * scale_coords
    points3D_rgb = dataset["points3D_rgb"] 
    if points3D_xyz is not None:
        points3D_rgb = points3D_rgb / 255.0
    if points3D_xyz is None and dataset["metadata"].get("id", None) == "blender":
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/dataset_readers.py#L221C4-L221C4
        num_pts = 100_000
        logging.info(f"generating random point cloud ({num_pts})...")
        # We create random points inside the bounds of the synthetic Blender scenes
        points3D_xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        points3D_rgb = SH2RGB(shs) 

    pcd = BasicPointCloud(points=points3D_xyz,colors=points3D_rgb,normals=None)
    scene_info = SceneInfo(point_cloud=pcd, train_cameras=None, test_cameras=[],
                        nerf_normalization=nerf_normalization,ply_path=None)
    return scene_info


def initialize(cfg, dataset):
    white_bg = dataset["metadata"].get("white_background", False)
    if cfg.get("white_bg", False):
        warnings.warn("Overriding dataset white_background to True as per cfg")
        white_bg = True
    scene_info = _convert_dataset_to_gaussian_splatting(dataset, "", white_background=white_bg, scale_coords=cfg.parameters.get("scale_coords", None))

    torch.manual_seed(0)
    #self._xyz = (torch.rand(n,3)*self.scene_extent*.5-self.scene_extent*.25).cuda()
    xyz = scene_info.point_cloud.points
    n = xyz.shape[0]
    max_sh_degree = cfg.parameters.max_sh_degree
    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(xyz)).float().cuda()), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    fused_color = RGB2SH(torch.tensor(np.asarray(scene_info.point_cloud.colors)).float().cuda())
    print(type(fused_color))
    features = torch.zeros((fused_color.shape[0], 3, (max_sh_degree + 1) ** 2)).float().cuda()
    features[:, :3, 0 ] = fused_color
    features[:, 3:, 1:] = 0.0
    from .model import GaussianModel, Activations
    act = Activations(cfg)
    return GaussianModel(cfg,
            white_bg=white_bg,
            activations=act,
            xyz = torch.from_numpy(xyz).float().cuda(),
            scaling = scales,
            rotation = torch.hstack([torch.ones(n,1),torch.zeros(n,3)]).cuda(),
            opacity = act.opacity_inverse(0.1 * torch.ones(n,1).cuda()),
            active_sh_degree = 0,
            max_sh_degree = max_sh_degree,
            features_dc = features[:,:,0:1].transpose(1, 2).contiguous(),
            features_rest = features[:,:,1:].transpose(1, 2).contiguous(),
            scene_extent = scene_info.nerf_normalization["radius"])

