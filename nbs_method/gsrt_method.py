from typing import Any, Dict
from nerfbaselines import Method
from nerfbaselines import Dataset, ModelInfo
from nerfbaselines import Cameras, CameraModel
from nerfbaselines import cameras
import torch
from torch import nn
import numpy as np

from extension import GaussiansTracer

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


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


class _TraceFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, setup, part_opac,part_xyz,part_scale,part_rot,part_sh,part_color):
        tracer = setup['tracer']
        tracer.load_gaussians(part_xyz,part_rot,part_scale,part_opac,part_sh,setup['sh_deg'],part_color)
        out = tracer.trace_rays(setup['ray_origins'],setup['ray_directions'],
                                setup['width'],setup['height'],
                                False,torch.tensor(0))
        ctx.setup = setup
        return out["radiance"]

    @staticmethod
    def backward(ctx, *grad_outputs):
        dout_dC = grad_outputs[0]
        setup = ctx.setup
        out = setup['tracer'].trace_rays(setup['ray_origins'],setup['ray_directions'],
                                         setup['width'],setup['height'],
                                         True,dout_dC)
        grad_xyz = out["grad_xyz"]
        grad_opacity = out["grad_opacity"][:,None]
        grad_scale = out["grad_scale"]
        grad_rot = out["grad_rot"]
        grad_sh = out["grad_sh"]
        grad_color = out["grad_color"]
        for grad,n in [(grad_xyz,"grad_xyz"),
                       (grad_opacity,"grad_opacity"),
                       (grad_scale,"grad_scale"),
                       (grad_rot,"grad_rot"),
                       (grad_sh,"grad_sh"),
                       (grad_color,"grad_color")]:
            nan_mask = torch.isnan(grad)
            if torch.any(nan_mask):
                print(f"found NaN grad in {n}")
            grad[nan_mask] = 0.
        return None,grad_opacity,grad_xyz,grad_scale,grad_rot,grad_sh,grad_color

        
def trace_function(setup,*args):
    out = _TraceFunction.apply(setup,*args)
    L = torch.mean(torch.sum((out-setup['target_img'])**2,dim=1))
    return L,out


class GSRTMethod(Method):

    def __init__(self, *,
                checkpoint: str = None, 
                train_dataset: Dataset = None,
                config_overrides: Dict[str, Any] = None):
        super().__init__()

        self.train_dataset = train_dataset
        self.hparams = {
            "init_num_points": 100000,
        }

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.percent_dense = 0.01

        self.checkpoint = checkpoint

        self.best_model = None
        self.best_loss = torch.inf

        self.tracer = GaussiansTracer(torch.device("cuda:0"))

        torch.manual_seed(0)

        part_num = self.hparams['init_num_points']
        scene_min = torch.tensor([-1.,-1.,-1.])
        scene_max = torch.tensor([1.,1.,1.])
        self._xyz = ((torch.rand(part_num,3)-.5)*(scene_max-scene_min)*.5+(scene_max+scene_min)*.5).cuda()
        self._scaling = torch.log(torch.ones(1,3).repeat(part_num,1)*0.01).cuda()
        self._rotation = torch.hstack([torch.ones(part_num,1),torch.zeros(part_num,3)]).cuda()
        self._opacity = (torch.ones(part_num,1)*.4).cuda()
        self._features_dc = torch.zeros(part_num,1,3).cuda()
        self._features_rest = torch.zeros(part_num,15,3).cuda()
        self.active_sh_degree = 3

        self._color = torch.ones(part_num,3)*torch.tensor([0.,1.,0.]).cuda() 

        self.viewpoint_ids = torch.arange(1)

        self.setup_functions()
        self.training_setup()


    def setup_functions(self):
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid


    def training_setup(self):
        self.xyz_gradient_accum = torch.zeros(self.params[1].shape[0])
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") 

        position_lr_init = 0.00016
        position_lr_final = 0.0000016
        position_lr_delay_mult = 0.01
        position_lr_max_steps = 30_000
        feature_lr = 0.0025
        opacity_lr = 0.025
        scaling_lr = 0.005
        rotation_lr = 0.001

        self.spatial_lr_scale = 0
        self.scene_extent = self.train_dataset['metadata']['expected_scene_scale']

        l = [
            {'params': [self._xyz], 'lr': position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': rotation_lr, "name": "rotation"}
        ]
        for p in l:
            p['params'][0].requires_grad_(True)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=position_lr_init*self.spatial_lr_scale,
                                                    lr_final=position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=position_lr_delay_mult,
                                                    max_steps=position_lr_max_steps)
        
        self.optimizer = torch.optim.AdamW(l,lr=0.0, eps=1e-15)
        
    
    def update_learning_rate(self, iteration):

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_xyz(self):
        return self._xyz 

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1) 


    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
   

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        #self.max_radii2D = self.max_radii2D[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors 


    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)


    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation) 


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # if max_screen_size:
        #     big_points_vs = self.max_radii2D > max_screen_size
        #     big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        #     prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

  
    #def add_densification_stats(self, viewspace_point_tensor, update_filter):
    #    self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
    #    self.denom[update_filter] += 1 

    def add_densification_stats(self):
        self.xyz_gradient_accum += torch.linalg.norm(self.params[1].grad,dim=1)
        self.denom += 1
        

    def train_iteration(self, step: int) -> Dict[str, float]:

        self.update_learning_rate(step)

        vp_id = self.viewpoint_ids[step%self.viewpoint_ids.shape[0]] 
        img = torch.from_numpy(self.train_dataset['images'][vp_id][:,:,:3])
        train_cameras = self.train_dataset['cameras']
        cameras_th = train_cameras.apply(lambda x, _: torch.from_numpy(x).contiguous().cuda())
        camera_th = cameras_th.__getitem__(vp_id)
        xy = cameras.get_image_pixels(camera_th.image_sizes)
        ray_origins, ray_directions = cameras.get_rays(camera_th, xy[None])
        res_x, res_y = camera_th.image_sizes
        setup = {
            'tracer':self.tracer,
            'ray_origins':ray_origins.float().squeeze(0).contiguous(),
            'ray_directions':ray_directions.float().squeeze(0).contiguous(),
            'width':res_x,
            'height':res_y,
            'target_img':img.reshape(res_x*res_y,3).cuda(),
            'sh_deg':self.active_sh_degree
        }
        L,out_img = trace_function(setup,self.get_opacity,self._xyz,self.get_scaling,self._rotation,self.get_features,self._color)
        L.backward()

        with torch.no_grad():

            # Log
            print(f'iter {step}, vp_id {vp_id} ---------------------')
            def print_stats(param,name):
                print(name,torch.mean(param,dim=0),torch.min(param,dim=0)[0],torch.max(param,dim=0)[0])
            print_stats(self.xyz.detach().cpu(),'xyz')
            print_stats(self.scale.detach().cpu(),'scale')
            print_stats(self.color.detach().cpu(),'color')
            #print_stats(self.sh.detach().cpu(),'sh')
            print_stats(self.opac.detach().cpu(),'opac')

            print(L.detach().item())
            print('------------------')

            # Densification
            if step < self.densify_until_iter:

                self.add_densification_stats()

                if step > self.densify_from_iter and step % self.densification_interval == 0:
                    size_threshold = 20 if step > self.opacity_reset_interval else None
                    self.densify_and_prune(self.densify_grad_threshold,0.005,self.scene_extent,size_threshold)

                if step % self.opacity_reset_interval == 0: #or (dataset.white_background and iteration == opt.densify_from_iter):
                    self.reset_opacity() 

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()


        return {"mse":L.detach().item(),"vp_id":vp_id, "out_image":out_img.detach().cpu().reshape(res_y,res_x,3).numpy()}


    @torch.no_grad()
    def render(self, camera : Cameras, *, options=None):

        camera_th = camera.apply(lambda x, _: torch.from_numpy(x).contiguous().cuda())
        camera_th = camera_th.__getitem__(0)
        xy = cameras.get_image_pixels(camera_th.image_sizes)
        ray_origins, ray_directions = cameras.get_rays(camera_th, xy[None])
        res_x, res_y = camera_th.image_sizes

        time_ms = 0
        nit = 1
        for i in range(nit):
            res = self.tracer.trace_rays(ray_origins.float().squeeze(0).contiguous(),
                                         ray_directions.float().squeeze(0).contiguous(),
                                         res_x, res_y,
                                         False,torch.tensor(0.))
            time_ms += res["time_ms"]
            #print(i,time_ms)
            #print(res["num_its"])
        time_ms /= nit
        
        color = res["radiance"].cpu().reshape(res_y,res_x,3).numpy()
        transmittance = res["transmittance"].cpu().reshape(res_y,res_x)[:,:,None].repeat(1,1,3).numpy()
        debug_map_0 = res["debug_map_0"].cpu().reshape(res_x,res_y,3).numpy()
        debug_map_1 = res["debug_map_1"].cpu().reshape(res_x,res_y,3).numpy()
        time_ms = res["time_ms"]
        num_its = res["num_its"]
        #print(num_its)
        #print(1000/time_ms, num_its/time_ms, res_x,res_y)
        print(time_ms, num_its, res_x,res_y)
        
        return {
            "color": color, #+ transmittance,
            "transmittance": transmittance,
            "debug_map_0": debug_map_0,
            "debug_map_1": debug_map_1,
        }

    
    def save(self, path):
        torch.save(self.best_model,f'{path}/checkpoint.pt')


    @classmethod
    def get_method_info(cls):
        return {
            # Method ID is provided by the registry
            "method_id": "",  

            # Supported camera models (e.g., pinhole, opencv, ...)
            "supported_camera_models": frozenset(("pinhole",)),

            # Features required for training (e.g., color, points3D_xyz, ...)
            "required_features": frozenset(("color",)),

            # Declare supported outputs
            "supported_outputs": ("color","transmittance","debug_map_0","debug_map_1"),
        }
    

    def get_info(self) -> ModelInfo:
        return {
            **self.get_method_info(),
            "loaded_checkpoint": self.checkpoint
        }
