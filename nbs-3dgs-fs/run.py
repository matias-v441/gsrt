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
import nerfbaselines
print(nerfbaselines.__file__)

import os
os.environ["NERFBASELINES_REGISTER"]=f"{os.getcwd()}/method/fsgs_method_spec.py"
import sys

method_name = "fsgs"
unfisheye = False
#data = f"/home/matbi/proj/Fisheye-GS/data/zipnerf-undistorted/{sys.argv[1]}" # FovX 1.713721368216191 FovY 1.1605348134616158 
data = f"/home/matbi/proj/Fisheye-GS/data/zipnerf-fisheye/{sys.argv[1]}" # FovX 1.9344611811489427 FovY 1.5364104594075099
root_dir=f"{method_name}_{sys.argv[1]}_fisheye"

checkpoint=f"{root_dir}/checkpoint"
im_dir=f"{root_dir}/renders"
eval_path=f"{root_dir}/eval.json"

eval_mode = False

backend = "conda"
#stack = ExitStack().__enter__()
#output_path = "output"
#stack.enter_context(backends.mount(output_path, output_path))

method_spec = get_method_spec(method_name)

#method_cls = stack.enter_context(build_method_class(method_spec, backend))
from method.fsgs_method import GaussianSplatting
method_cls = GaussianSplatting

method_info = method_cls.get_method_info()

train_dataset = load_dataset(data, 
                            split="train", 
                            features=method_info.get("required_features"),
                            supported_camera_models=method_info.get("supported_camera_models"),
                            load_features=True)

test_dataset = load_dataset(data, 
                             split="test", 
                             features=method_info.get("required_features"),
                             supported_camera_models=method_info.get("supported_camera_models"),
                             load_features=True)

presets, config_overrides = get_presets_and_config_overrides(
    method_spec, train_dataset["metadata"])

load_chpt = checkpoint if eval_mode else None
model = method_cls(
    unfisheye=unfisheye,
    train_dataset=train_dataset,
    config_overrides=config_overrides,
    checkpoint=load_chpt
)

model_info = model.get_info()
print("Loaded step", model_info["loaded_step"])

model_info["num_iterations"] = 30000

if not eval_mode:
    import wandb
    import numpy as np

    wandb.init(
        project="fsgs",
        name = sys.argv[1],
        config={
        "architecture": "FSGS",
        }
    )
    start_step = 0 if not model_info["loaded_step"] else model_info["loaded_step"]
    with tqdm(total=model_info["num_iterations"], initial=start_step) as pbar:
        for step in range(start_step,model_info["num_iterations"]):
            metrics = model.train_iteration(step)
            if (step+1) in [30000]:
                print(f'saving checkpoint_{step+1}')
                model.save(checkpoint)
            scales = metrics["scales"]
            opacities = metrics["opacities"]
            wandb_log = {'loss:':metrics['loss'],
                        'clone':metrics["densif_stats"]["cloned"],
                        'prune':metrics["densif_stats"]["pruned"],
                        'split':metrics["densif_stats"]["split"],
                        'opacities': wandb.Histogram(opacities,num_bins=100),
                        #'scales_max': wandb.Histogram(scales.max(axis=-1),num_bins=100),
                        #'pos_grad_norm': wandb.Histogram(metrics["pos_grad_norm"],num_bins=100),
                        'psnr':metrics['psnr'],
                        'N':metrics["densif_stats"]["total"]
                        }
            wandb.log(wandb_log)
            pbar.set_postfix({"psnr": f"{metrics['psnr']:.2f}"})
            pbar.update()

from nerfbaselines.evaluation import render_all_images, evaluate
for val in render_all_images(model, test_dataset, im_dir):
    pass
evaluate(im_dir, eval_path)

#stack = ExitStack()
#viewer = stack.enter_context(Viewer(
#                train_dataset=train_dataset, 
#                test_dataset=test_dataset, 
#                model=model))
#viewer.run()