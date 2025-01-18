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
ray_directions = c_rays @ c2w.T 
ray_directions /= ray_directions.norm(dim=1)[:,None]
ray_directions = ray_directions.to(device,dtype=torch.float32).contiguous()
#print(ray_origins,ray_directions)

im_ref = torch.zeros(res_y,res_x,3)
im_ref[400:600,450:550,0] = 1

# from PIL import Image
# im_ref = torch.from_numpy(np.array(Image.open("buldozer.jpg"))).float()/255.

plt.figure()
plt.imshow(im_ref)
im_ref = im_ref.reshape(res_x*res_y,3).to(device)

tracer = GaussiansTracer(device)

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
#%%
part_xyz.requires_grad = True
part_rot.requires_grad = True

part_scale.requires_grad = True
part_opac.requires_grad = True
part_sh.requires_grad = True
part_color.requires_grad = True
check = torch.autograd.gradcheck(trace_function_check,
                                 (part_opac,part_xyz,part_scale,part_rot,part_sh,part_color),
                                 nondet_tol=1e-7,
                                 atol=1e-3,
                                 eps=1e-1
                                 )
print(check)

#%%
#part_xyz.requires_grad = True
#part_rot.requires_grad = True
#part_scale.requires_grad = True
#part_opac.requires_grad = True
#part_sh.requires_grad = False
#part_color.requires_grad = True

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

best_loss = torch.inf
params = [part_opac,part_xyz,part_scale,part_rot,part_sh,part_color]
scene_extent = 100
percent_dense = 0.2
lr = 0.001
gradient_threshold = 0.05
split_scale = 1.6
opacity_threshold = 0.01

@torch.no_grad()
def prune():
    global params
    mask = (params[0]<opacity_threshold).squeeze()
    if torch.sum(mask) == 0:
        return
    params = [p[~mask].detach() for p in params]
    print("PRUNE", torch.sum(mask))
    return mask

@torch.no_grad()
def clone(grad_xyz, grad_xyz_abs):
    global params
    mask_grad = grad_xyz_abs>gradient_threshold
    mask_scale = torch.max(params[2],dim=1).values<percent_dense*scene_extent
    mask = torch.logical_and(mask_grad,mask_scale).squeeze()
    if torch.sum(mask) == 0:
        return
    stds = params[2][mask]
    means = torch.zeros(stds.size(0),3)
    samples = torch.normal(mean=means,std=stds)
    rots = build_rotation(params[3][mask]).cpu()
    params[1][mask] += torch.bmm(rots,samples.unsqueeze(-1)).squeeze(-1)
    params[1][mask] += grad_xyz[mask]
    params = [torch.vstack([p,p[mask]]) for p in params]
    print("CLONE", torch.sum(mask))

@torch.no_grad()
def split(grad_xyz_abs):
    global params
    mask_grad = grad_xyz_abs>gradient_threshold
    mask_scale = torch.max(params[2],dim=1).values>percent_dense*scene_extent
    mask = torch.logical_and(mask_grad,mask_scale).squeeze()
    if torch.sum(mask) == 0:
        return
    stds = params[2][mask]
    means = torch.zeros(stds.size(0),3)
    samples = torch.normal(mean=means,std=stds)
    rots = build_rotation(params[3][mask]).cpu()
    params[1][mask] += torch.bmm(rots,samples.unsqueeze(-1)).squeeze(-1)
    params[2][mask] = torch.log(params[2][mask]/split_scale)
    params = [torch.vstack([p,p[mask]]) for p in params]
    print("SPLIT", torch.sum(mask))

for p in params:
    p.requires_grad_()
optim = torch.optim.AdamW(params,lr=lr, weight_decay=1e-4, eps=1e-2)

xyz_grad_abs_acc = torch.zeros(params[1].shape[0])

densify = True

import wandb

# wandb.init(
#     project="gsrt",
# 
#     config={
#     "learning_rate": lr,
#     "architecture": "GSRT",
#     "dataset": "dumb rectangle",
#     "epochs": 1,
#     }
# )


for it in range(10000):
    
    part_opac_ = torch.sigmoid(params[0])
    part_scale_ = torch.exp(params[2])
    L = trace_function_check(part_opac_,params[1],part_scale_,params[3],params[4],params[5])
    if L < best_loss:
        best_loss = L.detach()
        best_model = [x.detach().clone() for x in params]

    print(L.detach().cpu().item())
    #wandb.log({"loss":L.detach()})

    optim.zero_grad()
    L.backward()
    optim.step()

    if densify:    
        xyz_grad_abs_acc += torch.linalg.norm(params[1].grad,dim=1)
        xyz_grad = params[1].grad.clone()

        optim.zero_grad()

        xyz_grad_abs_acc[xyz_grad_abs_acc.isnan()] = 0.

        num_orig = params[0].shape[0]
        clone(xyz_grad, xyz_grad_abs_acc)
        num_clone = params[0].shape[0]
        grad_padded = torch.zeros(params[0].shape[0])
        grad_padded[:xyz_grad_abs_acc.shape[0]] = xyz_grad_abs_acc
        split(grad_padded)
        num_split = params[0].shape[0]
        prune_mask = prune()
        num_prune = params[0].shape[0]

        densified = num_split-num_orig!=0
        pruned = num_split-num_prune!=0
        if densified:
            xyz_grad_abs_acc = torch.zeros(params[1].shape[0])
        if pruned:
            xyz_grad_abs_acc = xyz_grad_abs_acc[~prune_mask]
        if densified or pruned:
            for p in params:
                p.requires_grad_()
            optim = torch.optim.AdamW(params,lr=lr, weight_decay=1e-4, eps=1e-2)

print("MSE ", best_loss)
#%%

part_opac,part_xyz,part_scale,part_rot,part_sh,part_color = best_model

dL_dC = torch.ones(res_x*res_y,3).float().contiguous().to(device)
dL_dC /= dL_dC.numel()

part_color = torch.zeros_like(part_color)

tracer.load_gaussians(part_xyz,part_rot,torch.exp(part_scale),
                      torch.sigmoid(part_opac),part_sh,
                      3, part_color
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
plt.title(f"MSE={best_loss}")
plt.axis('off')
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