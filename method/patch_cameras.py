import numpy as np 

def _compute_residual_and_jacobian(
    x,
    y,
    xd,
    yd,
    k1 = 0.0,
    k2 = 0.0,
    k3 = 0.0,
    k4 = 0.0,
    p1 = 0.0,
    p2 = 0.0,
):
    """Auxiliary function of radial_and_tangential_undistort()."""
    # Adapted from https://github.com/google/nerfies/blob/main/nerfies/camera.py
    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 +
    #                   k4 * r(x, y)^4;
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4))
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y

def _radial_and_tangential_undistort(
    xd,
    yd,
    k1 = 0,
    k2 = 0,
    k3 = 0,
    k4 = 0,
    p1 = 0,
    p2 = 0,
    eps = 1e-9,
    max_iterations=10,
    xnp = np,
):
    """Computes undistorted (x, y) from (xd, yd)."""
    # From https://github.com/google/nerfies/blob/main/nerfies/camera.py
    # Initialize from the distorted point.
    x = xnp.clone(xd)
    y = xnp.clone(yd)

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, k4=k4, p1=p1, p2=p2
        )
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = xnp.where(
            xnp.abs(denominator) > eps,
            x_numerator / denominator,
            xnp.zeros_like(denominator),
        )
        step_y = xnp.where(
            xnp.abs(denominator) > eps,
            y_numerator / denominator,
            xnp.zeros_like(denominator),
        )

        x = x + step_x
        y = y + step_y

    return x, y

import nerfbaselines

from nerfbaselines._types import Cameras, GenericCameras, TTensor, _get_xnp

from typing import Tuple, Dict, cast, Any, TYPE_CHECKING, Optional
from nerfbaselines.cameras import _is_broadcastable
from nerfbaselines import camera_model_to_int
    
def unproject(cameras: GenericCameras[TTensor], xy: TTensor) -> Tuple[TTensor, TTensor]:
    
    xnp = _get_xnp(xy)
    assert xy.shape[-1] == 2
    assert _is_broadcastable(xy.shape[:-1], cameras.poses.shape[:-2]), \
        "xy must be broadcastable with poses, shapes: {}, {}".format(xy.shape[:-1], cameras.poses.shape[:-2])
    if hasattr(xy.dtype, "kind"):
        if not TYPE_CHECKING:
            assert xy.dtype.kind == "f"
    intrinsics = cameras.intrinsics
    distortion_parameters = cameras.distortion_parameters
    camera_models = cameras.camera_models
    poses = cameras.poses

    fx: TTensor
    fy: TTensor
    cx: TTensor
    cy: TTensor
    fx, fy, cx, cy = xnp.moveaxis(intrinsics, -1, 0)
    x = xy[..., 0]
    y = xy[..., 1]
    u = (x - cx) / fx
    v = (y - cy) / fy

    uv = xnp.stack((u, v), -1)
    if camera_models.item() == camera_model_to_int("opencv_fisheye"):

        #uv = _undistort(camera_models, distortion_parameters, uv, xnp=xnp)
        if distortion_parameters.shape[-1] != 0:
            uv[..., 0], uv[..., 1] = _radial_and_tangential_undistort(
                u,
                v,
                k1=distortion_parameters[...,0].item(),
                k2=distortion_parameters[...,1].item(),
                k3=distortion_parameters[...,4].item(),
                k4=distortion_parameters[...,5].item(),
                p1=distortion_parameters[...,2].item(),
                p2=distortion_parameters[...,3].item(),
                max_iterations=10,
                xnp=xnp,
            )
        th = xnp.min(xnp.sqrt(uv[..., 0] ** 2 + uv[..., 1] ** 2),xnp.tensor(xnp.pi))
        sth = xnp.sin(th)
        uv[...,0] = uv[...,0]/th*sth
        uv[...,1] = uv[...,1]/th*sth
        directions = xnp.concatenate((uv, xnp.cos(th)[...,None]), -1)
    elif camera_models.item() == camera_model_to_int("pinhole"):
        uv = cameras._undistort(camera_models, distortion_parameters, uv, xnp=xnp)
        directions = xnp.concatenate((uv, xnp.ones_like(uv[..., :1])), -1)
        directions = directions/ xnp.linalg.norm(directions, dim=-1, keepdim=True)
    else:
        raise ValueError("Camera model not supported")

    rotation = poses[..., :3, :3]  # (..., 3, 3)
    directions = (directions[..., None, :] * rotation).sum(-1)
    origins = xnp.broadcast_to(poses[..., :3, 3], directions.shape)
    return origins, directions

def apply():
    nerfbaselines.cameras.unproject = unproject