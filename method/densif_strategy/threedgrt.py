from .base import BaseDensifStrategy
import torch
import torch.nn as nn
from typing import Callable
import gc

from utils.general_utils import safe_state, build_rotation, get_expon_lr_func, inverse_sigmoid  # type: ignore

class Densif3DGRT(BaseDensifStrategy):

    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        assert(model.optimizer is not None)
        for k,v in cfg["parameters"]["densification"].items():
            self.__dict__[k] = v
        self.xyz_gradient_accum = torch.zeros((model.num_gaussians, 1), device="cuda")
        self.denom = torch.zeros((model.num_gaussians, 1), device="cuda") 


    def reset_opacity(self):
        opacities_new = self.model._act.opacity_inverse(torch.min(self.model.opacity, torch.ones_like(self.model._opacity)*0.01)) 
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
   

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.model.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.model.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.model.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.model.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.model.optimizer.param_groups:
            stored_state = self.model.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.model.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.model.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self.model._xyz = optimizable_tensors["xyz"]
        self.model._features_dc = optimizable_tensors["f_dc"]
        self.model._features_rest = optimizable_tensors["f_rest"]
        self.model._opacity = optimizable_tensors["opacity"]
        self.model._scaling = optimizable_tensors["scaling"]
        self.model._rotation = optimizable_tensors["rotation"]
        #self.model.xyz_2d = self.model.xyz_2d[valid_points_mask]
        #self.model.xyz_2d.requires_grad_()

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        #self.max_radii2D = self.max_radii2D[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.model.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.model.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.model.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.model.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors 


    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                               new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self.model._xyz = optimizable_tensors["xyz"]
        self.model._features_dc = optimizable_tensors["f_dc"]
        self.model._features_rest = optimizable_tensors["f_rest"]
        self.model._opacity = optimizable_tensors["opacity"]
        self.model._scaling = optimizable_tensors["scaling"]
        self.model._rotation = optimizable_tensors["rotation"]
        #self.model.xyz_2d = torch.zeros_like(self.model._xyz, requires_grad=True).cuda()

        self.xyz_gradient_accum = torch.zeros((self.model.num_gaussians,1), device="cuda")
        self.denom = torch.zeros((self.model.num_gaussians,1), device="cuda")
        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    @staticmethod 
    def build_scaling_rotation(s, r):
            L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
            R = build_rotation(r)
            L[:, :, 0] = R[:, :, 0] * s[:,0:1]
            L[:, :, 1] = R[:, :, 1] * s[:,1:2]
            L[:, :, 2] = R[:, :, 2] * s[:,2:3]

            # L[:,0,0] = s[:,0]
            # L[:,1,1] = s[:,1]
            # L[:,2,2] = s[:,2]

            # L = R @ L
            return L

    def _cleanup_memory(self):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.model.num_gaussians
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = torch.norm(grads,dim=-1) #grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.model.scaling, dim=1).values > self.percent_dense*scene_extent)
        if torch.sum(selected_pts_mask) + self.model.num_gaussians > 1.7e6:
            return
        #print(f'{self.iteration} SPLIT {torch.sum(selected_pts_mask)} scene_extent={scene_extent} grad_threshold={grad_threshold} percent_dense={self.percent_dense}')
        self.densif_stats["split"] += torch.sum(selected_pts_mask)
        stds = self.model.scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.model.rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.model._xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.model._act.scaling_inverse(self.model.scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self.model._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self.model._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self.model._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self.model._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

        self._cleanup_memory()


    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.model.scaling, dim=1).values <= self.percent_dense*scene_extent)
        if torch.sum(selected_pts_mask) + self.model.num_gaussians > 1.7e6:
            return
        #print(f'{self.iteration} CLONE {torch.sum(selected_pts_mask)} scene_extent={scene_extent} grad_threshold={grad_threshold} percent_dense={self.percent_dense}') 
        self.densif_stats["cloned"] += torch.sum(selected_pts_mask)
        new_xyz = self.model._xyz[selected_pts_mask]
        new_features_dc = self.model._features_dc[selected_pts_mask]
        new_features_rest = self.model._features_rest[selected_pts_mask]
        new_opacities = self.model._opacity[selected_pts_mask]
        new_scaling = self.model._scaling[selected_pts_mask]
        new_rotation = self.model._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation) 

        self._cleanup_memory()

    def get_wandb_log(self):
        import wandb
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        return {"densif_crit": wandb.Histogram(grads.cpu()*100,num_bins=100), **self.densif_stats}

    def clone_and_split(self):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, self.densify_clone_grad_threshold, self.model.scene_extent)
        self.densify_and_split(grads, self.densify_split_grad_threshold, self.model.scene_extent)

        # prune_mask = (self.get_opacity < min_opacity).squeeze()
        # n_pruned_by_opacity = torch.sum(prune_mask)
        # if max_screen_size:
        #     #big_points_vs = self.max_radii2D > max_screen_size
        #     big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        #     prune_mask = torch.logical_or(prune_mask, big_points_ws)
        # print(f'{self.iteration} PRUNE {torch.sum(prune_mask)} / {n_pruned_by_opacity} opac_min={torch.min(self.get_opacity)} scene_extent={extent} min_opacity={min_opacity}')
        # self.densif_stats["pruned"] += torch.sum(prune_mask)
        # self.prune_points(prune_mask)
        

        #torch.cuda.empty_cache()
        self._cleanup_memory()


    def prune_opacity(self):
        prune_mask = (self.model.opacity < self.densify_min_opacity).squeeze()
        #n_pruned_by_opacity = torch.sum(prune_mask)
        #print(f'{self.iteration} PRUNE {torch.sum(prune_mask)} / {n_pruned_by_opacity}')
        self.densif_stats["pruned"] += torch.sum(prune_mask)
        self.prune_points(prune_mask)

  
    #def add_densification_stats(self, viewspace_point_tensor, update_filter):
    #    self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
    #    self.denom[update_filter] += 1 

    def add_densification_stats(self, origin):
        mask = (self.model.xyz.grad != 0).max(dim=1).values
        dist = torch.norm(self.model.xyz[mask] - origin, dim=1, keepdim=True)
        pos_grad_norm = torch.norm(self.model.xyz.grad[mask]*dist,dim=1,keepdim=True)/2
        self.xyz_gradient_accum[mask] += pos_grad_norm
        self.denom[mask] += 1

    # def add_densification_stats(self, origin):
    #     mask = (self.model.xyz.grad != 0).max(dim=1).values
    #     dist = torch.norm(self.model.xyz[mask] - origin, dim=1, keepdim=True)
    #     grad_xyz = self.model.xyz.grad[mask]
    #     pos_grad_norm = torch.tan(torch.norm(grad_xyz,dim=1,keepdim=True)*dist)
    #     self.xyz_gradient_accum[mask] += pos_grad_norm
    #     self.denom[mask] += 1

    # def add_densification_stats(self, origin):
    #     mask = (self.model.xyz_2d.grad != 0).max(dim=1).values
    #     #dist = torch.norm(self.model.xyz[mask] - origin, dim=1, keepdim=True)
    #     pos_grad_norm = torch.norm(self.model.xyz_2d.grad[mask],dim=1,keepdim=True)
    #     self.xyz_gradient_accum[mask] += pos_grad_norm
    #     self.denom[mask] += 1

    @torch.no_grad()
    def densify(self, t_step:int, *, ray_origins: torch.Tensor):
        if t_step < self.densify_until_iter:

            self.add_densification_stats(ray_origins[0])

            if t_step > self.densify_from_iter and (t_step-self.densify_from_iter) % self.densification_interval == 0:
                #size_threshold = 20 if t_step > self.opacity_reset_interval else None
                #size_threshold = 20 if step > 500 else None
                self.clone_and_split()

            if t_step > self.densify_from_iter and (t_step-self.densify_from_iter) % self.prune_interval == 0:
                self.prune_opacity()

            if t_step % self.opacity_reset_interval == 0: # or t_step == self.densify_from_iter: # white_background
                self.reset_opacity()
