import torch
import signal
import atexit

from types import SimpleNamespace
from utils.general_utils import get_expon_lr_func, inverse_sigmoid  # type: ignore

from utils.loss_utils import l1_loss, ssim  # type: ignore
from utils.image_utils import psnr

from .densif_strategy import Densif3DGRT

class Training:

    def __init__(self, cfg, model):
        self.model = model
        self.cfg = cfg 

        self.densif_strategy = Densif3DGRT(cfg, self.model)

        self.model.setup_training()
        self.start_iter = model.iteration + 1

        self.wandb_run = None
        if cfg.use_wandb:
            import wandb
            import os
            train_dir = os.path.join(cfg.results_dir,"train")
            os.makedirs(train_dir, exist_ok=True)
            self.wandb_run = wandb.init(project="gsrt", config=dict(cfg), dir=train_dir)
            wandb.watch(self.model, log="all")

        self._register_cleanup_handlers()

    def _register_cleanup_handlers(self):
        # Register atexit callback (called on normal program termination)
        atexit.register(self.finalize)
        # Register signal handlers for SIGINT (Ctrl+C) and SIGTERM
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, cleaning up wandb...")
            self.finalize()
            # Re-raise the signal to continue with normal termination
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def finalize(self):
        if self.cfg.save_results:
            print(f'saving final checkpoint')
            self.model.save(f'{self.cfg.results_dir}/checkpoint_final.pt')
        if self.wandb_run is not None:
            print("Finishing wandb run...")
            self.wandb_run.finish()
            self.wandb_run = None

    def __del__(self):
        self.finalize()


    def step(self, *, t_step, rays, gt_image) -> float:

        import time
        t0 = time.perf_counter()

        self.model.train()

        self.model.iteration = self.start_iter + t_step

        self.model.optimizer.update_learning_rate(self.model.iteration)

        if self.model.iteration % 1000 == 0:
            self.model.oneupSHdegree()

        image = self.model.forward(rays)

        loss_cfg = self.cfg.parameters.loss

        l1 = l1_loss(image, gt_image)
        ssim_value = ssim(image.reshape(1,rays.res_y,rays.res_x,3).permute(0,3,1,2), 
                        gt_image.reshape(1,rays.res_y,rays.res_x,3).permute(0,3,1,2))
        Ll1 = l1 if loss_cfg.use_l1 else 0
        Lssim = (1.0 - ssim_value) if loss_cfg.use_ssim else 0
        loss = loss_cfg.lambda_l1*Ll1 + loss_cfg.lambda_ssim*Lssim

        loss.backward()

        self.densif_strategy.densify(self.model.iteration, ray_origins=rays.origins)

        self.model.optimizer.step()

        dt = (time.perf_counter() - t0)*1000 # ms

        with torch.no_grad():

            # Log metrics to wandb if enabled
            psnr = 10 * torch.log10(1 / torch.mean((image - gt_image) ** 2))
            if self.wandb_run is not None:
                self.wandb_run.log({
                    "loss": loss.item(),
                    "psnr": psnr.item(),
                    "ssim": ssim_value.item(),
                    "l1_loss": l1.item(),
                    "num_gaussians": self.model.num_gaussians,
                    "learning_rate": self.model.optimizer.param_groups[0]['lr'],
                    "iteration": self.model.iteration,
                    "Tms/fwd": self.model.perf["T_fwd"],
                    "Tms/bwd": self.model.perf["T_bwd"],
                    "Tms/total": dt,
                    "Tfrac/fwd": self.model.perf["T_fwd"]/dt,
                    "Tfrac/bwd": self.model.perf["T_bwd"]/dt,
                    **self.densif_strategy.get_wandb_log(),
                    **self.model.get_wandb_log()
                }, step=self.model.iteration)

        self.model.optimizer.zero_grad()
        
        return {"loss": loss.item(), "image": image, "T":dt}
