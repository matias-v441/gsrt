#%%
import torch
import numpy as np
from extension import GaussiansTracer
import math

device = torch.device("cuda:0")

tracer = GaussiansTracer(device)
gaussians,it = torch.load("data/drums/checkpoint/chkpnt-30000.pth")
gs_xyz = gaussians[1].detach().cpu()
gs_scaling = torch.exp(gaussians[4].detach().cpu())
#gs_scaling = torch.ones((1,3)).repeat((gs_xyz.shape[0],1))*.01
gs_rotation = gaussians[5].detach().cpu()

gs_opacity = torch.ones_like(gaussians[6].detach().cpu()) 
gs_opacity = torch.sigmoid(gaussians[6].detach().cpu())

gs_features_dc = gaussians[2].detach().cpu()
gs_features_rest = gaussians[3].detach().cpu()
gs_active_sh_degree = gaussians[0]
gs_sh = torch.cat((gs_features_dc,gs_features_rest),dim=1).contiguous()

tracer.load_gaussians(gs_xyz,gs_rotation,gs_scaling,gs_opacity,gs_sh,gs_active_sh_degree)

cam_data = {"width": 800, "height": 800,
              "position": [-0.0, 2.7372601032257085, 2.959291696548462], 
              "rotation": [[-1.0000001192092896, -0.0, -0.0],
                           [0.0, 0.7341100573539734, -0.67903071641922],
                           [-0.0, -0.67903071641922, -0.7341099977493286]],
                "fy": 1250.0000504168488, "fx": 1250.0000504168488}

res_x = cam_data["width"]
res_y = cam_data["height"]

# fx,fy,cx,cy = (cam_data["fx"],cam_data["fy"],res_x*.5,res_y*.5)
# K = torch.tensor([[fx,0.,cx],[0.,fy,cy],[0.,0.,1.]])
# R = torch.tensor(cam_data["rotation"]).T
# C = torch.tensor(cam_data["position"])

fx,fy,cx,cy = (1250.,1250.,-res_x/2,-res_y/2)
K = torch.tensor([[fx,0.,cx],[0.,fy,cy],[0.,0.,1.]])
R = torch.eye(3)
C = torch.tensor([0.,0., 5.])

c2w = torch.inverse(K@R)

num_rays = res_x*res_y
ray_origins = C.repeat(num_rays,1).float().contiguous().to(device)

x,y=torch.meshgrid(torch.linspace(0,res_x,res_x),torch.linspace(0,res_y,res_y))
c_rays = torch.stack([x.flatten(),y.flatten(),-torch.ones(num_rays)],dim=1)
ray_directions = c_rays @ c2w.T
ray_directions /= ray_directions.norm(dim=1)[:,None]
ray_directions = ray_directions.float().contiguous().to(device)

dL_dC = torch.ones(res_x*res_y,3).float().contiguous().to(device)
out = tracer.trace_rays(ray_origins,ray_directions,res_x,res_y,True,dL_dC)
radiance = out["radiance"].cpu().reshape(res_x,res_y,3)
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(radiance.numpy())

print(out["grad_opacity"])

#%%
class _TracerFunction_Check(torch.autograd.Function):

    @staticmethod
    def forward(ctx, part_opac):
        tracer.load_gaussians(gs_xyz,gs_rotation,gs_scaling,part_opac,gs_sh,3)
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
gs_opacity.requires_grad = True
check = torch.autograd.gradcheck(trace_function_check,(gs_opacity))
print(check)