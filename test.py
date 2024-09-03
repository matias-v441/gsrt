#%%
import torch
import numpy as np
from extension import GaussiansTracer
#%%
gaussians,it = torch.load("data/drums.pth")
gs_xyz = gaussians[1].detach()
gs_scaling = gaussians[4].detach()
gs_rotation = gaussians[5].detach()
gs_opacity = gaussians[6].detach()
np.savez("data/drums.npz",
         xyz=gs_xyz.cpu().numpy(),
         scaling=gs_scaling.cpu().numpy(),
         rotation=gs_rotation.cpu().numpy(),
         opacity=gs_opacity.cpu().numpy()
         )
#%%

device = torch.device("cuda:0")
tracer = GaussiansTracer(device)

#%%
tracer.load_gaussians(gs_xyz.cpu(),gs_rotation.cpu(),gs_scaling.cpu(),gs_opacity.cpu())