
import torch
from . import Rays

class TraceFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, part_opac, part_xyz, part_scale, part_rot, part_sh,
                tracer, active_sh_deg, rays: Rays, white_background: bool, update: bool, as_params: dict):
        if not tracer.has_gaussians() or update:
            tracer.load_gaussians(part_xyz,part_rot,part_scale,part_opac,part_sh,active_sh_deg,as_params)
        out = tracer.trace_fwd(rays.origins, rays.directions, rays.res_x, rays.res_y, white_background)
        ctx.rad = out["radiance"]
        ctx.trans = out["transmittance"]
        ctx.dist = out["distance"]
        ctx.rays = rays
        ctx.tracer = tracer
        ctx.white_background = white_background
        out_color = out["radiance"]
        if white_background:
            out_color = out_color + out["transmittance"]
        return out_color

    @staticmethod
    def backward(ctx, *grad_outputs):
        dout_dC = grad_outputs[0]
        rays = ctx.rays
        out = ctx.tracer.trace_bwd(rays.origins, rays.directions,
                                    rays.res_x, rays.res_y,
                                    ctx.white_background,
                                    dout_dC, ctx.rad, ctx.trans, ctx.dist)
        grad_xyz = out["grad_xyz"]
        grad_opacity = out["grad_opacity"][:,None]
        grad_scale = out["grad_scale"]
        grad_rot = out["grad_rot"]
        grad_sh = out["grad_sh"]
        for grad,n in [(grad_xyz,"grad_xyz"),
                       (grad_opacity,"grad_opacity"),
                       (grad_scale,"grad_scale"),
                       (grad_rot,"grad_rot"),
                       (grad_sh,"grad_sh")]:
            nan_mask = torch.isnan(grad)
            if torch.any(nan_mask):
                print(f"found NaN grad in {n}")
            grad[nan_mask] = 0.
        return grad_opacity, grad_xyz, grad_scale, grad_rot, grad_sh,\
            None, None, None, None, None, None
