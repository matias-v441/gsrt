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

import wandb
import numpy as np
from random import randint
import random
random.seed(0)
from PIL import Image

from nbs_method.gsrt_method import GSRTMethod
method_cls = GSRTMethod

blender = "external://blender/"
mipnerf = "external://mipnerf360/"
#data = [("stump",mipnerf)] #[("lego",blender),("drums",blender),("bonsai",mipnerf),("stump",mipnerf)]
data = [("stump",mipnerf)] #[("lego",blender),("drums",blender),("bonsai",mipnerf),("stump",mipnerf)]


wandb.init(
    project="gsrt",

    config={
    "architecture": "GSRT",
    "dataset": "blender/lego",
    "epochs": 1,
    }
)

table = wandb.Table(columns=["scene","time[ms]","FPS","mil.r/s"])

for scene,ds in data:

    method_info = method_cls.get_method_info()
    train_dataset = load_dataset(ds+scene, 
                             split="train", 
                             features=method_info.get("required_features"),
                             supported_camera_models=method_info.get("supported_camera_models"),
                             load_features=True)
    model = method_cls(
        checkpoint=f"data/{scene}/checkpoint/chkpnt-30000.pth",
        train_dataset=train_dataset
    )
    model.rendering_setup()
    num_vps = 1
    time_ms = 0
    for _ in range(num_vps):
        metrics = model.render(camera=train_dataset["cameras"],
                            options={"num_avg_it":1,
                                "vp_id":randint(0,len(train_dataset["cameras"])-1)})
        time_ms += metrics['time_ms']
    time_ms /= num_vps
     
    pil_img = Image.fromarray((metrics['color']/np.max(metrics['color'])*255).astype(np.uint8))
    pil_img.save(scene+".png")
    out_img = wandb.Image(pil_img)
    wandb.log({scene:out_img})
    w,h = metrics['res_xy']
    print(w,h)
    table.add_data(scene,time_ms,1000/time_ms,w*h/time_ms/1000) 

wandb.log({'rendering_perf':table})
