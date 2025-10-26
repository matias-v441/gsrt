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
    if not cfg.train and (cfg.checkpoint is None or not os.path.exists(cfg.checkpoint)):
        raise ValueError("You must provide a valid checkpoint unless you are training.")
    if cfg.train:
        train_dataset = load_dataset(cfg.data_path, 
                                    split="train", 
                                    features=method_info.get("required_features"),
                                    supported_camera_models=method_info.get("supported_camera_models"),
                                    load_features=True)
    if cfg.evaluate:
        test_dataset = load_dataset(cfg.data_path, 
                                    split="test", 
                                    features=method_info.get("required_features"),
                                    supported_camera_models=method_info.get("supported_camera_models"),
                                    load_features=True)

    #presets, config_overrides = get_presets_and_config_overrides(method_spec, train_dataset["metadata"])
    model = method_cls(
        checkpoint=cfg.checkpoint,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config_overrides=cfg,
    )
    model_info = model.get_info()
    model_info["num_iterations"] = cfg.parameters.n_iterations

    if cfg.view:
        stack = ExitStack()
        viewer = stack.enter_context(Viewer(
                        #train_dataset=train_dataset, 
                        #test_dataset=test_dataset, 
                        model=model))
        if cfg.train:
            import threading
            tviewer = threading.Thread(target=viewer.run)
            tviewer.start()
        else:
            viewer.run()
        
    if cfg.use_wandb:
        wandb.init(
            project="gsrt",
            config={
            "learning_rate": .001,
            "architecture": "GSRT",
            "dataset": "blender/lego",
            "epochs": 1,
            }
        )

    import time
    start_time = time.time()
    with tqdm(total=model_info["num_iterations"]) as pbar:
        for step in range(cfg.start_iter,model_info["num_iterations"]+1):
            metrics = model.train_iteration(step)
            if cfg.save_results and (step >= cfg.save_start_iter and step%cfg.save_interval==0):
                            print(f'saving checkpoint_{step}')
                            model.save(cfg.results_dir,step)
            #pbar.set_postfix({"psnr": f"{metrics['psnr']:.2f}"})
            # scales = model.get_scaling.detach().cpu().numpy()
            # opacities = model.get_opacity.detach().cpu().numpy()
            # wandb_log = {'loss:':metrics['loss'],
            #             'clone':metrics["densif_stats"]["cloned"],
            #             'prune':metrics["densif_stats"]["pruned"],
            #             'split':metrics["densif_stats"]["split"],
            #             'opacities': wandb.Histogram(opacities,num_bins=100),#np.histogram(opacities,1000)[0]),
            #             'scales_max': wandb.Histogram(scales.max(axis=-1),num_bins=100),#np.histogram(scales.max(axis=-1),500)[0]),
            #             'pos_grad_norm': wandb.Histogram(metrics["pos_grad_norm"],num_bins=100),
            #             #'resp_grad': wandb.Histogram(metrics["resp_grad"],num_bins=100),
            #             'N':metrics["densif_stats"]["total"]
            #             }
            # if (step >= 0 and step%300==0) and metrics["out_image"] is not None:# or step < 10:
            #     wandb_log['image'] = wandb.Image(metrics['out_image']/np.max(metrics['out_image']))
            # if cfg.use_wandb:
            #     wandb.log(wandb_log)
            pbar.update()

    end_time = time.time()
    print(f"Training time: {(end_time - start_time)/60:.2f} minutes") # 185.36 min for 30K lego, 158.74 min for 30K drums
    # Save the model
    model.save(cfg.results_dir,model_info["num_iterations"])
    # Create a minimal nb-info.json file such that the model can be loaded
    with open("nb-info.json", "w") as f:
        f.write(f'{{"method": "{cfg.method_name}"}}')

    # Close the stack. In real code, you should use the context manager
    stack.close()
    tviewer.join()


if __name__ == "__main__":
    main()