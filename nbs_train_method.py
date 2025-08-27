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

method_name = "gsrt"
backend = "conda"

# We will write the output to directory "output"
output_path = "output"

# We will use the data from the "mipnerf360/bicycle" scene
#data = "external://mipnerf360/bonsai"
#data = "external://blender/lego"
data = "/home/matbi/proj/Fisheye-GS/data/zipnerf-fisheye/alameda"
#data = "/home/matbi/proj/Fisheye-GS/data/zipnerf-undistorted/alameda"
#data = "external://blender/drums"

# We use the exit stack to simplify the context management
#stack = ExitStack().__enter__()

# Prepare the output directory, and mount it if necessary (e.g., for docker backend)
#stack.enter_context(backends.mouxnp.atan(xnp.atan(xnp.atan(xnp.atan(xnp.atan(xnp.atan(nt(output_path, output_path))

# Get the method specification for a registered method
method_spec = get_method_spec(method_name)

# Build the method class and start the backend
#method_cls = stack.enter_context(build_method_class(method_spec, backend))
from nbs_method.gsrt_method import GSRTMethod
method_cls = GSRTMethod

# Load train dataset
# We use the method info to load the required features and supported camera models
method_info = method_cls.get_method_info()
train_dataset = load_dataset(data, 
                             split="train", 
                             features=method_info.get("required_features"),
                             supported_camera_models=method_info.get("supported_camera_models"),
                             load_features=True)

print(train_dataset.keys())
print(train_dataset['cameras'][0])
print(train_dataset["metadata"])

# Load eval dataset
test_dataset = load_dataset(data, 
                            split="test", 
                            features=method_info.get("required_features"),
                            supported_camera_models=method_info.get("supported_camera_models"),
                            load_features=True)


# Each method can specify custom presets and config overrides
# Apply config overrides for the train dataset
presets, config_overrides = get_presets_and_config_overrides(method_spec, train_dataset["metadata"])

chpt_iter = 30000
test = False
track = False
view_async = False
use_chpt = True
save_chpt = False
chpt_dir = "drums_checkpoint"
config_overrides["3dgs_data"] = False
config_overrides["3dgrt_data"] = False
config_overrides["white_bg"] = False
if use_chpt:
    model = method_cls(
        #checkpoint=f'gsrt_checkpoint_full/checkpoint_{chpt_iter}.pt',
        checkpoint=f'{chpt_dir}/checkpoint_{chpt_iter}.pt',
        #checkpoint=f'gsrt_checkpoint/checkpoint_{chpt_iter}.pt',
        #checkpoint=f'gsrt_checkpoint_zipnerf/checkpoint_{chpt_iter}.pt',
        #checkpoint=f'/home/matbi/proj/3dgrut/runs/lego-2204_020424/ours_{chpt_iter}/ckpt_{chpt_iter}.pt',
        #checkpoint=f'/home/matbi/proj/3dgrut/runs/bonsai-2304_211030/ours_{chpt_iter}/ckpt_{chpt_iter}.pt',
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config_overrides=config_overrides,
    )
else:
    model = method_cls(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config_overrides=config_overrides,
    )

# import matplotlib.pyplot as plt
# import numpy as np
# 
# model.rendering_setup()
# im = model.render(train_dataset['cameras'],options={'vp_id':10})['color']
# print(np.max(im))
# plt.figure()
# plt.imshow(im)
# plt.show()
# quit()

# model.training_setup()
# tr = model.train_iteration(chpt_iter)
# tr = model.train_iteration(chpt_iter+1)
# plt.figure()
# print(np.max(tr["out_image"]))
# plt.imshow(tr['out_image']/np.max(tr['out_image']))
# plt.show()
# quit()

# Training loop
model_info = model.get_info()

# In this example we override the total number of iterations
# to make the training faster
model_info["num_iterations"] = 30000
start_iteration = chpt_iter if use_chpt else 1


if test:
    ssim, psnr, lpips = model.test()
    print(f"PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips}")
# drums PSNR: 26.04239504814148, SSIM: 0.9543158429861068, LPIPS: 0.034241623212583366
# PSNR: 26.115967416763304, SSIM: 0.9551858580112458, LPIPS: 0.032885719225741926

import wandb
import numpy as np


if track:
    wandb.init(
        project="gsrt",

        config={
        "learning_rate": .001,
        "architecture": "GSRT",
        "dataset": "blender/lego",
        "epochs": 1,
        }
    )

from nerfbaselines.viewer import Viewer

from contextlib import ExitStack


# vp_id = 0
# ref_img = wandb.Image(train_dataset['images'][vp_id][:,:,:3])
# ref_img_alpha = wandb.Image(train_dataset['images'][vp_id][:,:,3][:,:,None])
# wandb.log({
#             f'ref_rgb_{vp_id}':ref_img,
#             f'ref_a_{vp_id}':ref_img_alpha
#             })

model.training_setup()

stack = ExitStack()
viewer = stack.enter_context(Viewer(
                #train_dataset=train_dataset, 
                #test_dataset=test_dataset, 
                model=model))

if view_async:
    import threading
    tviewer = threading.Thread(target=viewer.run)
    tviewer.start()
else:
    viewer.run()

import time
start_time = time.time()
#with tqdm(total=model_info["num_iterations"]) as pbar:
for step in range(start_iteration,model_info["num_iterations"]+1):
    metrics = model.train_iteration(step)
    if save_chpt and (step >= 29000 and step%100==0):# or (step in [1,2,3,50]):
                    print(f'saving checkpoint_{step}')
                    model.save(chpt_dir,step)
    #pbar.set_postfix({"psnr": f"{metrics['psnr']:.2f}"})
    vp_id = metrics['vp_id'] 
    scales = model.get_scaling.detach().cpu().numpy()
    opacities = model.get_opacity.detach().cpu().numpy()
    wandb_log = {'loss:':metrics['loss'],
                 'clone':metrics["densif_stats"]["cloned"],
                 'prune':metrics["densif_stats"]["pruned"],
                 'split':metrics["densif_stats"]["split"],
                 'opacities': wandb.Histogram(opacities,num_bins=100),#np.histogram(opacities,1000)[0]),
                 'scales_max': wandb.Histogram(scales.max(axis=-1),num_bins=100),#np.histogram(scales.max(axis=-1),500)[0]),
                 'pos_grad_norm': wandb.Histogram(metrics["pos_grad_norm"],num_bins=100),
                 #'resp_grad': wandb.Histogram(metrics["resp_grad"],num_bins=100),
                 'N':metrics["densif_stats"]["total"]
                 }
    
    if (step >= 0 and step%300==0) and metrics["out_image"] is not None:# or step < 10:
        wandb_log['image'] = wandb.Image(metrics['out_image']/np.max(metrics['out_image']))
    
    if track:
        wandb.log(wandb_log)

    # pbar.update()

end_time = time.time()
print(f"Training time: {(end_time - start_time)/60:.2f} minutes") # 185.36 min for 30K lego, 158.74 min for 30K drums
# Save the model
model.save(chpt_dir,model_info["num_iterations"])
# Create a minimal nb-info.json file such that the model can be loaded
with open("nb-info.json", "w") as f:
    f.write(f'{{"method": "{method_name}"}}')

# Close the stack. In real code, you should use the context manager
#stack.close()

tviewer.join()