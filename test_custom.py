#%%
import torch
import numpy as np
import math

import torch.autograd.gradcheck
from extension import GaussiansTracer,TracerCustom

device = torch.device("cuda:0")

part_num = 30
torch.manual_seed(0)
scene_min = torch.tensor([-100.,-100.,300.])
scene_max =  torch.tensor([100.,100.,200.])
part_xyz = (torch.rand(part_num,3)-.5)*(scene_max-scene_min)*.5+(scene_max+scene_min)*.5
part_scale = torch.ones(part_num,3)
part_rot = torch.hstack([torch.ones(part_num,1),torch.zeros(part_num,3)])
part_opac = torch.ones(part_num,1)
part_sh = torch.zeros(part_num,16,3).contiguous()
active_sh_deg = 3


# part_num = 100
# torch.manual_seed(0)
# scene_min = torch.tensor([-50.,-50.,300.])
# scene_max =  torch.tensor([50.,50.,200.])
# part_xyz = (torch.rand(part_num,3)-.5)*(scene_max-scene_min)*.5+(scene_max+scene_min)*.5
# #part_xyz = torch.tensor([0.,0.,300.])
# #part_scale = torch.ones(part_num,3)*60
# part_scale = torch.tensor([[1.,1.,2.]]).repeat(part_num,1)*2
# part_rot = torch.hstack([torch.ones(part_num,1),torch.zeros(part_num,3)])
# #part_rot = torch.ones(part_num,4)*torch.tensor([0.97,0.23,0.,0.])
# #part_rot = torch.ones(part_num,4)*torch.tensor([0.91,0.42,0.,0.])
# #part_rot = torch.ones(part_num,4)*torch.tensor([1.,0.,0.,0.])
# part_opac = torch.ones(part_num,1)*.4
# part_sh = torch.zeros(part_num,16,3).contiguous()
# part_color = torch.ones(part_num,3).contiguous()*torch.tensor([0.,1.,0.])
# active_sh_deg = 3


# gaussians,it = torch.load("data/drums/checkpoint/chkpnt-30000.pth")
# part_xyz = gaussians[1].detach().cpu()
# part_scaling = torch.exp(gaussians[4].detach().cpu())
# part_scale = torch.ones((1,3)).repeat((part_xyz.shape[0],1))*.01
# part_rot = gaussians[5].detach().cpu()
# part_opac = torch.sigmoid(gaussians[6].detach().cpu())
# part_features_dc = gaussians[2].detach().cpu()
# part_features_rest = gaussians[3].detach().cpu()
# active_sh_deg = gaussians[0]
# part_sh = torch.cat((part_features_dc,part_features_rest),dim=1).contiguous()

use_custom = True

if use_custom:
    tracer = TracerCustom(device)
else:
    tracer = GaussiansTracer(device)

tracer.load_gaussians(part_xyz,part_rot,part_scale,
                      part_opac,part_sh,
                      active_sh_deg
                      )

quit()
#%%

# fx,fy,cx,cy = (1000.,1000.,-500.,-500.)
# K = torch.tensor([[fx,0.,cx],[0.,fy,cy],[0.,0.,1.]])
# R = torch.eye(3)
# C = torch.tensor([0.,0.,5.])
# c2w = torch.inverse(K@R)
# res_x,res_y = 1000,1000
# num_rays = res_x*res_y
# ray_origins = C.repeat(num_rays,1).to(device,dtype=torch.float32).contiguous()
# 
# x,y=torch.meshgrid(torch.linspace(0.,res_x,res_x),torch.linspace(0.,res_y,res_y),indexing='xy')
# c_rays = torch.stack([x.flatten(),y.flatten(),-torch.ones(num_rays)],dim=1)
# ray_directions = c_rays @ c2w.T
# ray_directions /= ray_directions.norm(dim=1)[:,None]
# ray_directions = ray_directions.to(device,dtype=torch.float32).contiguous()

fx,fy,cx,cy = (1000.,1000.,500.,500.)
K = torch.tensor([[fx,0.,cx],[0.,fy,cy],[0.,0.,1.]])
R = torch.eye(3)
C = torch.tensor([0.,0.,1])
c2w = torch.inverse(K@R)
res_x,res_y = 1000,1000
#res_x,res_y = 100,1
#res_x,res_y = 1,1
num_rays = res_x*res_y
ray_origins = C.repeat(num_rays,1).float().contiguous()
x,y=torch.meshgrid(torch.linspace(0.,res_x,res_x),torch.linspace(0.,res_y,res_y))
#x,y=torch.tensor([[res_x*.5]]),torch.tensor([[res_y*.5]])
#x,y = torch.linspace(0.,res_x,res_x),torch.tensor([[500.]]).repeat(res_x,1)
c_rays = torch.stack([x.flatten(),y.flatten(),torch.ones(num_rays)],dim=1)
ray_directions = c_rays @ c2w.T + C
ray_directions /= ray_directions.norm(dim=1)[:,None]
ray_directions = ray_directions.float().contiguous()
#print(ray_origins,ray_directions)
if not use_custom:
    ray_origins = ray_origins.to(device=device)
    ray_directions = ray_directions.to(device=device)

out = tracer.trace_rays(ray_origins,ray_directions,res_x,res_y,False,torch.tensor(0.))

radiance = out["radiance"].cpu().reshape(res_x,res_y,3)
transmittance = out["transmittance"].cpu().reshape(res_x,res_y)

print("num_its", out["num_its"])

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(radiance)
