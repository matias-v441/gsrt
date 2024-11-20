#%%
#!make
import torch
import numpy as np
import math

import torch.autograd.gradcheck
from extension import GaussiansTracer

import matplotlib.pyplot as plt

device = torch.device("cuda:0")

part_num = 100
torch.manual_seed(0)
scene_min = torch.tensor([-50.,-50.,300.])
scene_max =  torch.tensor([50.,50.,200.])
part_xyz = (torch.rand(part_num,3)-.5)*(scene_max-scene_min)*.5+(scene_max+scene_min)*.5
#part_xyz = torch.tensor([0.,0.,300.])
#part_scale = torch.ones(part_num,3)*60
part_scale = torch.tensor([[1.,1.,2.]]).repeat(part_num,1)*2
part_rot = torch.hstack([torch.ones(part_num,1),torch.zeros(part_num,3)])
#part_rot = torch.ones(part_num,4)*torch.tensor([0.97,0.23,0.,0.])
#part_rot = torch.ones(part_num,4)*torch.tensor([0.91,0.42,0.,0.])
#part_rot = torch.ones(part_num,4)*torch.tensor([1.,0.,0.,0.])
part_opac = torch.ones(part_num,1)*.4
part_sh = torch.zeros(part_num,16,3).contiguous()
part_color = torch.ones(part_num,3).contiguous()*torch.tensor([0.,1.,0.])
active_sh_deg = 3

fx,fy,cx,cy = (1000.,1000.,500.,500.)
K = torch.tensor([[fx,0.,cx],[0.,fy,cy],[0.,0.,1.]])
R = torch.eye(3)
C = torch.tensor([0.,0.,1])
c2w = torch.inverse(K@R)
res_x,res_y = 1000,1000
#res_x,res_y = 100,1
#res_x,res_y = 1,1
num_rays = res_x*res_y
ray_origins = C.repeat(num_rays,1).to(device,dtype=torch.float32).contiguous()
x,y=torch.meshgrid(torch.linspace(0.,res_x,res_x),torch.linspace(0.,res_y,res_y))
#x,y=torch.tensor([[res_x*.5]]),torch.tensor([[res_y*.5]])
#x,y = torch.linspace(0.,res_x,res_x),torch.tensor([[500.]]).repeat(res_x,1)
c_rays = torch.stack([x.flatten(),y.flatten(),torch.ones(num_rays)],dim=1)
ray_directions = c_rays @ c2w.T + C
ray_directions /= ray_directions.norm(dim=1)[:,None]
ray_directions = ray_directions.to(device,dtype=torch.float32).contiguous()
#print(ray_origins,ray_directions)

im_ref = torch.zeros(res_y,res_x,3)
im_ref[400:600,450:550,0] = 1
plt.figure()
plt.imshow(im_ref)
im_ref = im_ref.reshape(res_x*res_y,3).to(device)

tracer = GaussiansTracer(device)

#%%

part_xyz.requires_grad = True
part_rot.requires_grad = True

part_scale.requires_grad = True
part_opac.requires_grad = True
part_sh.requires_grad = False
part_color.requires_grad = True


class _TracerFunction_Check(torch.autograd.Function):

    @staticmethod
    def forward(ctx,part_opac,part_xyz,part_scale,part_rot,part_sh,part_color):
        tracer.load_gaussians(part_xyz,part_rot,part_scale,part_opac,part_sh,3,part_color)
        out = tracer.trace_rays(ray_origins,ray_directions,res_x,res_y,False,torch.tensor(0))
        return out["radiance"]

    @staticmethod
    def backward(ctx, *grad_outputs):
        dout_dC = grad_outputs[0]
        out = tracer.trace_rays(ray_origins,ray_directions,res_x,res_y,True,dout_dC)
        grad_xyz = out["grad_xyz"].cpu()
        grad_opacity = out["grad_opacity"].cpu()[:,None]
        grad_scale = out["grad_scale"].cpu()
        grad_rot = out["grad_rot"].cpu()
        grad_sh = out["grad_sh"].cpu()
        grad_color = out["grad_color"].cpu()
        return grad_opacity,grad_xyz,grad_scale,grad_rot,grad_sh,grad_color

def trace_function_check(*args):
    out = _TracerFunction_Check.apply(*args)
    L = torch.mean(torch.sum((out-im_ref)**2,dim=1))
    return L

# for i in range(3):
#     print(_TracerFunction_Check.backward(None,torch.tensor(1.).to(device)))

check = torch.autograd.gradcheck(trace_function_check,
                                 (part_opac,part_xyz,part_scale,part_rot,part_sh,part_color),
                                 nondet_tol=1e-7,
                                 atol=1e-3,
                                 eps=1e-1
                                 )
print(check)
#%%
part_xyz.requires_grad = True
part_rot.requires_grad = True
part_scale.requires_grad = True
part_opac.requires_grad = True
part_sh.requires_grad = True
part_color.requires_grad = True
best_model = None
best_loss = torch.inf
for it in range(1000):
    part_opac_ = torch.sigmoid(part_opac)
    part_scale_ = torch.exp(part_scale)
    L = trace_function_check(part_opac_,part_xyz,part_scale_,part_rot,part_sh,part_color)
    print(L)
    L.backward()
    #print(part_opac.grad)
    part_opac.requires_grad = False
    part_xyz.requires_grad = False
    part_scale.requires_grad = False
    part_rot.requires_grad = False
    part_sh.requires_grad = False
    part_color.requires_grad = False
    lr = 0.001
    part_opac -= lr*part_opac.grad
    part_xyz -= lr*part_xyz.grad
    part_scale -= lr*part_scale.grad
    part_rot -= lr*part_rot.grad
    part_sh -= lr*part_sh.grad
    part_color -= lr*part_color.grad
    part_opac.requires_grad = True
    part_xyz.requires_grad = True
    part_scale.requires_grad = True
    part_rot.requires_grad = True
    part_sh.requires_grad = True
    part_color.requires_grad = True
    if L < best_loss:
        best_loss = L.detach()
        best_model = (x.detach().clone() for x in (part_opac,part_xyz,part_rot,part_scale,part_sh,part_color))
    #if it%100 == 0:
    #    out = tracer.trace_rays(ray_origins,ray_directions,res_x,res_y,False,torch.tensor(0.))
    #    radiance = out["radiance"].cpu().reshape(res_x,res_y,3)
    #    plt.figure()
    #    plt.imshow(radiance)

print("MSE ", best_loss)
part_opac,part_xyz,part_rot,part_scale,part_sh,part_color = best_model

#%%

dL_dC = torch.ones(res_x*res_y,3).float().contiguous().to(device)
dL_dC /= dL_dC.numel()

tracer.load_gaussians(part_xyz,part_rot,torch.exp(part_scale),
                      torch.sigmoid(part_opac),part_sh,
                      active_sh_deg, part_color
                      )
out = tracer.trace_rays(ray_origins,ray_directions,res_x,res_y,True,dL_dC)
dL_dC = torch.ones(res_x*res_y,3).float().contiguous().to(device)
#out = tracer.trace_rays(ray_origins,ray_directions,res_x,res_y,True,dL_dC)

radiance = out["radiance"].cpu().reshape(res_x,res_y,3)
print("num_pix",radiance.shape,torch.count_nonzero(radiance))
print("LOSS", torch.mean(radiance))
transmittance = out["transmittance"].cpu().reshape(res_x,res_y)

print("num_its", out["num_its"])
print("num_its_bwd", out["num_its_bwd"])

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

print("dL_dresp", out["grad_resp"].cpu())
#%%

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

print(build_rotation(part_rot))
x_hit = torch.tensor([-66.154587, -66.154587, 265.883240])
v = 1/60*(x_hit-part_xyz)
dresp_dopac = torch.exp(-torch.dot(v,v))
print(dresp_dopac)
dC_dresp = .5
print(dC_dresp*dresp_dopac*3)
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