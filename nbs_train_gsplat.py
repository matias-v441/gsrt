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

method_name = "gaussian-splatting"
backend = "conda"

# We will write the output to directory "output"
output_path = "output"

# We will use the data from the "mipnerf360/bicycle" scene
#data = "external://mipnerf360/bicycle"
data = "external://blender/lego"

# We use the exit stack to simplify the context management
stack = ExitStack().__enter__()

# Prepare the output directory, and mount it if necessary (e.g., for docker backend)
stack.enter_context(backends.mount(output_path, output_path))

# Get the method specification for a registered method
method_spec = get_method_spec(method_name)

# Build the method class and start the backend
method_cls = stack.enter_context(build_method_class(method_spec, backend))

# Load train dataset
# We use the method info to load the required features and supported camera models
method_info = method_cls.get_method_info()
train_dataset = load_dataset(data, 
                             split="train", 
                             features=method_info.get("required_features"),
                             supported_camera_models=method_info.get("supported_camera_models"),
                             load_features=True)

# Load eval dataset
#test_dataset = load_dataset(data, 
#                            split="test", 
#                            features=method_info.get("required_features"),
#                            supported_camera_models=method_info.get("supported_camera_models"),
#                            load_features=True)

# Each method can specify custom presets and config overrides
# Apply config overrides for the train dataset
presets, config_overrides = get_presets_and_config_overrides(
    method_spec, train_dataset["metadata"])

# Build the method
model = method_cls(
    #checkpoint='gsrt_checkpoint/checkpoint.pt',
    train_dataset=train_dataset,
    config_overrides=config_overrides,
)

import matplotlib.pyplot as plt
import numpy as np

# Training loop
model_info = model.get_info()

# In this example we override the total number of iterations
# to make the training faster
model_info["num_iterations"] = 10000

import wandb
import numpy as np

wandb.init(
    project="gsrt",

    config={
    #"learning_rate": .001,
    "architecture": "3DGS",
    "dataset": "blender/lego",
    "epochs": 1,
    }
)

with tqdm(total=model_info["num_iterations"]) as pbar:
    for step in range(model_info["num_iterations"]):
        metrics = model.train_iteration(step)
        out_img = wandb.Image(metrics['out_image']/np.max(metrics['out_image']))
        pbar.set_postfix({"psnr": f"{metrics['psnr']:.2f}"})
        #wandb.log({'loss':metrics['loss'],'image':out_img})
        pbar.update()

# Save the model
model.save("gsplat_checkpoint")
# Create a minimal nb-info.json file such that the model can be loaded
with open("nb-info.json", "w") as f:
    f.write(f'{{"method": "{method_name}"}}')

# Close the stack. In real code, you should use the context manager
stack.close()
