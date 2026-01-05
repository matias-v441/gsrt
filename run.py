from tqdm import tqdm
from contextlib import ExitStack
from nerfbaselines import backends
from nerfbaselines.datasets import load_dataset
from nerfbaselines import (
    build_method_class,
    get_method_spec,
)
from nerfbaselines.training import (
    get_presets_and_config_overrides,
)
from nerfbaselines.viewer import Viewer
from contextlib import ExitStack
from nbs_method.gsrt_method import GSRTMethod
method_cls = GSRTMethod
from omegaconf import DictConfig,OmegaConf
import hydra
import os
import wandb
import numpy as np


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    print(cfg.data_path)    
    os.environ["NERFBASELINES_REGISTER"]=f"{os.getcwd()}/nbs_method/gsrt_method_spec.py"
    method_spec = get_method_spec(cfg.method_name)
    method_info = method_cls.get_method_info()
    train_dataset,test_dataset = None,None
    # load final checkpoint from previous run if exists
    if cfg.checkpoint is None and cfg.resume_training and cfg.results_dir is not None and os.path.exists(cfg.results_dir):
        cfg.checkpoint = os.path.join(cfg.results_dir, "checkpoint_final.pt")
    print("Using checkpoint", cfg.checkpoint)
    # require checkpoint if not training
    if not cfg.train and (cfg.checkpoint is None or not os.path.exists(cfg.checkpoint)): 
        raise ValueError("You must provide a valid checkpoint unless you are training.")
    if cfg.train:
        train_dataset = load_dataset(cfg.data_path, 
                                    split="train", 
                                    features=method_info.get("required_features"),
                                    supported_camera_models=method_info.get("supported_camera_models"),
                                    load_features=True)
        presets, config_overrides = get_presets_and_config_overrides(method_spec, train_dataset["metadata"])
    if cfg.evaluate or cfg.use_viewer:
        test_dataset = load_dataset(cfg.data_path, 
                                    split="test", 
                                    features=method_info.get("required_features"),
                                    supported_camera_models=method_info.get("supported_camera_models"),
                                    load_features=True)
        presets, config_overrides = get_presets_and_config_overrides(method_spec, test_dataset["metadata"]) 

    #presets, config_overrides = get_presets_and_config_overrides(method_spec, train_dataset["metadata"])
    model = method_cls(
        checkpoint=cfg.checkpoint,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config_overrides=cfg,
    )

    if cfg.gradcheck:
        print("Running gradcheck...")
        with tqdm(total=len(test_dataset["cameras"])) as pbar:
            for i in range(len(test_dataset["cameras"])):
                is_gradcorrect = model.gradcheck(i)
                print(f"Gradcheck result: {is_gradcorrect}")
                pbar.update()
        quit()

    if cfg.use_viewer:
        stack = ExitStack()
        viewer = stack.enter_context(Viewer(
                        train_dataset=train_dataset, 
                        test_dataset=test_dataset, 
                        model=model))
        tviewer = None
        if cfg.train or cfg.evaluate:
            import threading
            tviewer = threading.Thread(target=viewer.run)
            tviewer.start()
        else:
            viewer.run()

    if cfg.train:
        print("Training...")
        import time
        start_time = time.time()
        num_steps = cfg.parameters.n_iterations - max(model.model.iteration,cfg.start_step)
        train_dir = os.path.join(cfg.results_dir, "train")
        os.makedirs(train_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(train_dir, "config.yaml"))
        psnr_ema = 0
        with tqdm(total=num_steps) as pbar:
            for step in range(num_steps):
                metrics = model.train_iteration(step)
                psnr_ema = metrics["psnr"]*0.8 + psnr_ema*0.2
                if cfg.save_results\
                      and model.model.iteration % 2000 == 0:
                    model.model.save(os.path.join(cfg.results_dir,f'checkpoint_final.pt'))#os.path.join(train_dir,f'chpt_{model.model.iteration}.pt'))
                pbar.update()
        end_time = time.time()
        print(f"Training time: {(end_time - start_time)/60:.2f} minutes") 
        with open(os.path.join(train_dir, "metrics.json"), "wb") as f:
            f.write(str.encode(
                f'{{"psnr_ema": {psnr_ema:.4f}, "time_min": {(end_time - start_time)/60:.2f}}}'
            ))

    if cfg.evaluate:
        print("Evaluating...")
        print("N",model.model.num_gaussians/1000)
        eval_dir = os.path.join(cfg.results_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True) 
        OmegaConf.save(cfg, os.path.join(eval_dir, "config.yaml"))
        num_steps = len(test_dataset["cameras"])
        with tqdm(total=num_steps) as pbar:
            tot_ssim, tot_psnr, tot_lpips = 0.0, 0.0, 0.0
            num_its_avg = 0
            num_its_max = 0
            fps = 0
            for step in range(num_steps):
                ssim, psnr, lpips = model.test_iteration(step)
                tot_ssim += ssim
                tot_psnr += psnr
                tot_lpips += lpips
                # cam_th = model.test_cameras_th.__getitem__(step%len(model.test_cameras_th))
                # model.render(cam_th,patch_camera=False)
                num_its_avg += model.model.num_its
                num_its_max = max(num_its_max,model.model.num_its)
                fps += model.model.fps
                pbar.update()
            print(f"Average SSIM: {tot_ssim/num_steps:.4f}, PSNR: {tot_psnr/num_steps:.4f}, LPIPS: {tot_lpips/num_steps:.4f}")
            print(f"num_its_avg: {num_its_avg/num_steps} num_its_max: {num_its_max} fps: {fps/num_steps}")
            with open(os.path.join(eval_dir, "eval.json"), "wb") as f:
                f.write(str.encode(
                    f'{{"ssim": {tot_ssim/num_steps:.4f}, "psnr": {tot_psnr/num_steps:.4f}, "lpips": {tot_lpips/num_steps:.4f}}}'
                ))
        from nerfbaselines.evaluation import render_all_images, evaluate, build_evaluation_protocol
        renders_dir = os.path.join(cfg.results_dir, "renders")
        for val in render_all_images(model, test_dataset, renders_dir):
            pass
        # protocol=build_evaluation_protocol("nerf")
        # evaluate(renders_dir,os.path.join(eval_dir, "eval_nbs.json"),evaluation_protocol=protocol)
        

    if cfg.use_viewer:
        stack.close()
        if tviewer: tviewer.join()


if __name__ == "__main__":
    main()