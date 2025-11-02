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
    if cfg.resume_training and cfg.results_dir is not None and os.path.exists(cfg.results_dir):
        cfg.checkpoint = os.path.join(cfg.results_dir, "checkpoint_final.pt")
    # require checkpoint if not training
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

    import time
    start_time = time.time()
    with tqdm(total=model_info["num_iterations"]) as pbar:
        for step in range(model_info["num_iterations"]+1):
            model.train_iteration(step)
            pbar.update()

    end_time = time.time()
    print(f"Training time: {(end_time - start_time)/60:.2f} minutes") # 185.36 min for 30K lego, 158.74 min for 30K drums
    # Create a minimal nb-info.json file such that the model can be loaded
    with open("nb-info.json", "w") as f:
        f.write(f'{{"method": "{cfg.method_name}"}}')

    # Close the stack. In real code, you should use the context manager
    stack.close()
    tviewer.join()


if __name__ == "__main__":
    main()