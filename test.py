#%%
import torch
import numpy as np
from extension import GaussiansTracer
import math

device = torch.device("cuda:0")

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
#%%
tracer1 = GaussiansTracer(device)

# xyz rotation scaling opacity
#gs = [(torch.zeros((1,3)),torch.tensor([[5000.,0.,0.]]),torch.tensor([[-10000.,-10000.,0.]])),
#      (torch.zeros((1,4)), torch.zeros((1,4)), torch.zeros((1,4)) ),
#      (torch.zeros((1,3)), torch.zeros((1,3)), torch.zeros((1,3)) ),
#      (torch.zeros((1,1)), torch.zeros((1,1)), torch.zeros((1,1)) )
#      ]
gs = [[],[],[],[]]
nsteps = 5
w_width = 50#35000.
step_size = w_width/nsteps
scale_x = 1
scale_y = 1
for i in range(nsteps):
    for j in range(nsteps):
        s = -w_width*.5
        gs[0].append(torch.tensor([[s+step_size*i,s+step_size*j,0.]]))
        gs[1].append(torch.zeros((1,4)))
        gs[2].append(torch.tensor([scale_x*(i+1),scale_y*(j+1),1.]))
        gs[3].append(torch.zeros((1,1)))

#gs[0].append(torch.tensor([[0.,0.,0.]]))
#gs[1].append(torch.zeros((1,4)))
#gs[2].append(torch.ones((1,3))*1)
#gs[3].append(torch.zeros((1,1)))

gs_merged = [torch.stack(x,dim=0) for x in gs]
tracer1.load_gaussians(*gs_merged)
origin = torch.tensor([0.,0.,50.])
res_x,res_y = 800,800
w,h = 800,800
num_rays = res_x*res_y
x,y=torch.meshgrid(torch.linspace(-w/2,w/2,res_x),torch.linspace(-h/2,h/2,res_y))
from cameras import cameras
from math import sin, cos, radians

focus = 1250.

def trace_rays(R):
    
    #R=torch.tensor(cameras[50]["rotation"])
    global focus
    #c_im = torch.stack([x,y,torch.ones((res_x,res_y))],dim=2).reshape(num_rays,3)
    c_im = torch.stack([x.flatten(),y.flatten(),-torch.ones(num_rays)*focus],dim=1)
    c_im /= c_im.norm(dim=1)[:,None]
    c_im = c_im@R

    ray_origins = (R.T@origin).repeat(num_rays,1).to(device,dtype=torch.float32)
    ray_directions = c_im.to(device,dtype=torch.float32)
    #ray_origins = (torch.stack([x.flatten(),y.flatten(),torch.zeros(num_rays)],dim=1)@R + origin).to(device,dtype=torch.float32)
    #dirs = torch.tensor([0.,0.,-1.]).repeat(num_rays,1)@R
    #ray_directions = dirs.to(device,dtype=torch.float32)

    res = tracer1.trace_rays(ray_origins,ray_directions)
    radiance = res["radiance"].cpu().reshape(res_x,res_y,3)
    #print(torch.sum(radiance))
    return radiance

yaw = 0.
pitch = 0.
R = torch.eye(3)

import matplotlib.pyplot as plt
fig = plt.figure()
imobj = plt.imshow(trace_rays(R).numpy())

from matplotlib.backend_bases import KeyEvent,MouseEvent,MouseButton
start = None

def update():
    imobj.set_data(trace_rays(R).numpy())
    fig.canvas.draw_idle()

def on_move(event):
    global start
    s = 500
    if event.inaxes and start is not None:
        pos = torch.tensor((event.xdata,event.ydata))
        d = pos-start
        start = pos
        global yaw,pitch
        yaw += -radians(d[1]/res_y)*s
        pitch += radians(d[0]/res_x)*s
        Rx = torch.tensor([[1.,0.,0.],[0.,cos(pitch),-sin(pitch)],[0.,sin(pitch),cos(pitch)]])
        Ry = torch.tensor([[cos(yaw),0.,sin(yaw)],[0.,1.,0.],[-sin(yaw),0.,cos(yaw)]])
        Rxy = Rx@Ry
        global R
        R = Rxy
        update()
        

def on_click(event):
    if event.inaxes:
        global start
        start = torch.tensor((event.xdata,event.ydata))

def on_key(event):
    sx = 5.
    sy = 5.
    sz = 5.
    global origin
    if event.key == "W":
        origin += R.T@torch.tensor([sx,0.,0.])
        update()
    if event.key == "A":
        origin += R.T@torch.tensor([0.,-sy,0.])
        update()
    if event.key == "S":
        origin += R.T@torch.tensor([-sx,0.,0.])
        update()
    if event.key == "D":
        origin += R.T@torch.tensor([0.,sy,0.])
        update()
    if event.key == "E":
        origin += R.T@torch.tensor([0.,0.,-sz])
        update()
    if event.key == "F":
        origin += R.T@torch.tensor([0.,0.,sz])
        update()
        

def on_release(event):
    global start
    start = None

def on_scroll(event):
    global focus
    if event.button == "up":
        focus += 50
    elif event.button == "down":
        focus -= 50
    update()

plt.connect('motion_notify_event', on_move)
plt.connect('button_press_event', on_click)
plt.connect('button_release_event', on_release)
plt.connect('scroll_event', on_scroll)
fig.canvas.mpl_connect('key_press_event', on_key)


#plt.figure()
#print(torch.max(c_im))
#ray_im = c_im.reshape((res_y,res_x,3))*.5+.5
#plt.imshow(ray_im[350:450,350:450])

plt.show()
quit()
#%%
print(torch.det(R),R.T@R)
#%%
# Load data
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
# Init pipeline
tracer = GaussiansTracer(device)
#%%
# Build GAS
tracer.load_gaussians(gs_xyz.cpu(),gs_rotation.cpu(),gs_scaling.cpu(),gs_opacity.cpu())

#%% 
# Trace rays
#cam_data = {"width": 800, "height": 800,
#              "position": [-0.0, 2.7372601032257085, 2.959291696548462], 
#              "rotation": [[-1.0000001192092896, -0.0, -0.0],
#                           [0.0, 0.7341100573539734, -0.67903071641922],
#                           [-0.0, -0.67903071641922, -0.7341099977493286]],
#                "fy": 1250.0000504168488, "fx": 1250.0000504168488}
from cameras import cameras
for cam_data in cameras[:10]:
    T = torch.tensor(cam_data["position"])
    R = torch.tensor(cam_data["rotation"])
    w,h = cam_data["width"],cam_data["height"]
    res_x,res_y = 800,800
    num_rays = res_x*res_y
    x,y=torch.meshgrid(torch.linspace(0,w,res_x),torch.linspace(0,h,res_y))
    c_im = torch.stack([x,y,torch.ones((res_x,res_y))],dim=2).reshape(num_rays,3)
    axis = R@torch.tensor([400.,400.,1.])
    ort = torch.cross(axis,R@torch.tensor([0.,400.,1.]))
    ort /= ort.norm()
    side = torch.cross(axis,ort)
    side /= side.norm()
    #-10*axis+2*ort+200*side
    ray_origins = (-R@T-10*axis+2*ort+200*side).repeat(num_rays,1).to(device,dtype=torch.float32)
    ray_directions = (c_im@R.T)
    ray_directions /= torch.norm(ray_directions,dim=1)[:,None]
    ray_directions = ray_directions.to(device,dtype=torch.float32)

    res = tracer.trace_rays(ray_origins,ray_directions)
    radiance = res["radiance"].cpu().reshape(res_x,res_y,3)
    print(torch.sum(radiance))
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(radiance.numpy())
