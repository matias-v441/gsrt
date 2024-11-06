#%%
#!make
import torch
import numpy as np
import math

import torch.autograd.gradcheck
from extension import GaussiansTracer

device = torch.device("cuda:0")

part_num = 30
torch.manual_seed(0)
scene_min = torch.tensor([-100.,-100.,300.])
scene_max =  torch.tensor([100.,100.,300.])
part_xyz = (torch.rand(part_num,3)-.5)*(scene_max-scene_min)*.5+(scene_max+scene_min)*.5
part_scale = torch.ones(part_num,3)*2
part_rot = torch.hstack([torch.ones(part_num,1),torch.zeros(part_num,3)])
part_opac = torch.ones(part_num,1)
part_sh = torch.zeros(part_num,16,3).contiguous()
active_sh_deg = 3

tracer = GaussiansTracer(device)

tracer.load_gaussians(part_xyz,part_rot,part_scale,
                      part_opac,part_sh,
                      active_sh_deg
                      )


fx,fy,cx,cy = (1000.,1000.,500.,500.)
K = torch.tensor([[fx,0.,cx],[0.,fy,cy],[0.,0.,1.]])
R = torch.eye(3)
C = torch.tensor([0.,0.,1])
c2w = torch.inverse(K@R)
res_x,res_y = 1000,1000
num_rays = res_x*res_y
ray_origins = C.repeat(num_rays,1).to(device,dtype=torch.float32).contiguous()

x,y=torch.meshgrid(torch.linspace(0.,res_x,res_x),torch.linspace(0.,res_y,res_y))
c_rays = torch.stack([x.flatten(),y.flatten(),torch.ones(num_rays)],dim=1)
ray_directions = c_rays @ c2w.T + C
ray_directions /= ray_directions.norm(dim=1)[:,None]
ray_directions = ray_directions.to(device,dtype=torch.float32).contiguous()
dL_dC = torch.ones(res_x*res_y,3).float().contiguous().to(device)
out = tracer.trace_rays(ray_origins,ray_directions,res_x,res_y,True,dL_dC)

radiance = out["radiance"].cpu().reshape(res_x,res_y,3)
transmittance = out["transmittance"].cpu().reshape(res_x,res_y)

print("num_its", out["num_its"])
print("num_its_bwd", out["num_its_bwd"])

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(radiance)

grad_opacity = out["grad_opacity"].cpu()
print("opacity", part_opac.shape, grad_opacity.shape) 
print(grad_opacity)
grad_xyz = out["grad_xyz"].cpu()
print("xyz", grad_xyz.shape,part_xyz.shape)
print(grad_xyz)
grad_sh = out["grad_sh"].cpu()
print("SH", part_sh.shape,grad_sh.shape)
print(grad_sh)
grad_scale = out["grad_scale"].cpu()
print("scale", grad_scale.shape, part_scale.shape)
print(grad_scale)
grad_rot = out["grad_rot"].cpu()
print("rotation", grad_rot.shape, part_rot.shape)
print(grad_rot)

#%%
class _TracerFunction_Check(torch.autograd.Function):

    @staticmethod
    def forward(ctx, part_opac):
        tracer.load_gaussians(part_xyz,part_rot,part_scale,part_opac,part_sh,3)
        out = tracer.trace_rays(ray_origins,ray_directions,res_x,res_y,False,torch.tensor(0))
        return torch.sum(out["radiance"])

    @staticmethod
    def backward(ctx, *grad_outputs):
        dL_dC = torch.ones(res_x*res_y,3).float().contiguous().to(device)
        tracer.trace_rays(ray_origins,ray_directions,res_x,res_y,True,dL_dC)
        grad_xyz = out["grad_xyz"].cpu()
        grad_opacity = out["grad_opacity"].cpu()[:,None]
        return grad_opacity

def trace_function_check(part_opac):
    return _TracerFunction_Check.apply(part_opac)

#part_xyz.requires_grad = False
part_opac.requires_grad = True
check = torch.autograd.gradcheck(trace_function_check,(part_opac))
print(check)
#%%

part_xyz.requires_grad = True
part_rot.requires_grad = True
part_scale.requires_grad = True
part_opac.requires_grad = True
part_sh.requires_grad = True

class _TracerFunction_NoLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, part_xyz, part_scale, part_opac, part_sh):
        tracer.load_gaussians(part_xyz,part_rot,part_scale,part_opac,part_sh,3)
        dL_dC = torch.ones(res_x*res_y,3).float().contiguous().to(device)
        out = tracer.trace_rays(ray_origins,ray_directions,res_x,res_y,True,dL_dC)
        ctx.save_for_backward(out["grad_xyz"],out["grad_rot"],out["grad_scale"],
                              out["grad_opacity"],out["grad_sh"])
        return out["radiance"]

    @staticmethod
    def backward(ctx, *grad_outputs):
        return ctx.saved_tensors


def trace_function(part_opac):
    return _TracerFunction_NoLoss.apply(part_opac)

#%%

class _TracerFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *part_data):
        tracer.load_gaussians(*part_data,3)
        out = tracer.trace_rays(ray_origins,ray_directions,res_x,res_y,False,torch.tensor(0))
        return out["radiance"].cpu()

    @staticmethod
    def backward(ctx, *grad_outputs):
        dL_dC = grad_outputs[0].contiguous().to(device)
        out = tracer.trace_rays(ray_origins,ray_directions,res_x,res_y,True,dL_dC)
        return out["grad_xyz"].cpu(),out["grad_rot"].cpu(),out["grad_scale"].cpu(),\
               out["grad_opacity"][:,None].cpu(),out["grad_sh"].cpu()

def trace_function(*args):
    return _TracerFunction.apply(*args)

target = torch.ones(res_x*res_y,3)
rad = trace_function(part_xyz, part_rot, part_scale, part_opac, part_sh)
loss = torch.mean(torch.sum((target-rad)**2,dim=1))
loss.backward()