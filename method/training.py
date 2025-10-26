import torch

from types import SimpleNamespace
from utils.general_utils import safe_state, build_rotation, get_expon_lr_func, inverse_sigmoid  # type: ignore

from utils.loss_utils import l1_loss, ssim  # type: ignore
from utils.image_utils import psnr

from .densif_strategy import Densif3DGRT

class Training:

    def __init__(self, cfg, model, dataset):
        self.model = model
        self.dataset = dataset

        optim_par = SimpleNamespace(**cfg["parameters"]["optimizer"])
        param_groups = self.model.get_parameter_groups(optim_par)
        for p in param_groups:
            p['params'][0].requires_grad_(True)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=optim_par.position_lr_init*model.scene_extent, 
                                                    lr_final=optim_par.position_lr_final*model.scene_extent, 
                                                    lr_delay_mult=optim_par.position_lr_delay_mult,
                                                    max_steps=optim_par.position_lr_max_steps)
        
        self.model.optimizer = torch.optim.AdamW(param_groups, lr=0.01, eps=1e-15)

        if cfg["white_bg"]:
            self.white_background = True
        else:
            self.white_background = dataset["metadata"].get("white_background", False) 

        self.densif_strategy = Densif3DGRT(cfg, self.model)


    def update_learning_rate(self, iteration):
        for param_group in self.model.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr


    def step(self, *, tracer, t_step, rays, gt_image) -> float:

        self.update_learning_rate(t_step)

        if t_step % 1000 == 0:
            self.model.oneupSHdegree()

        image = self.model.forward(tracer, rays, self.white_background)

        Ll1 = l1_loss(image, gt_image) 
        ssim_value = ssim(image.reshape(1,rays.res_y,rays.res_x,3).permute(0,3,1,2), 
                        gt_image.reshape(1,rays.res_y,rays.res_x,3).permute(0,3,1,2))
        loss = Ll1 # (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - ssim_value)

        loss.backward()

        with torch.no_grad():

            # Densification stats
            self.densif_strategy.densify(t_step, ray_origins=rays.origins)
            # print(f"{t_step} after densif {self._xyz.shape}")

            pos_grad_norm = torch.norm(self.model._xyz.grad,dim=1) if self.model._xyz.grad is not None else torch.zeros(self.model._xyz.shape[0])
            #print(f"{t_step} N={self.model.num_gaussians} L={loss.item()} opac=[{torch.min(self.model.opacity)} {torch.max(self.model.opacity)}] pos_grad_norm ={pos_grad_norm.max()}")

        # Optimizer step
        self.model.optimizer.step()
        self.model.optimizer.zero_grad()
        return {"loss": loss.item(), "image": image}
