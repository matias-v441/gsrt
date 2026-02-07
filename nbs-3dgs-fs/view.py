from tqdm import tqdm
from contextlib import ExitStack
from _nerfbaselines import backends
from _nerfbaselines.datasets import load_dataset
from _nerfbaselines import (
    build_method_class,
    get_method_spec,
)
from _nerfbaselines.training import (
    get_presets_and_config_overrides,
)

from _nerfbaselines.viewer import Viewer
import sys

method_name = "fsgs"
#data = "/home/matbi/proj/Fisheye-GS/data/zipnerf-undistorted/alameda"
data = f"/home/matbi/proj/Fisheye-GS/data/zipnerf-fisheye/{sys.argv[1]}"
checkpoint=f"zipnerf_fisheye_wide/fsgs_{sys.argv[1]}_fisheye/checkpoint"
#checkpoint=f"fsgs_{sys.argv[1]}_fisheye/checkpoint"
#checkpoint=f"zipnerf_undist/fsgs_{sys.argv[1]}_undistorted/checkpoint"
backend = "conda"

import os
os.environ["NERFBASELINES_REGISTER"]=f"{os.getcwd()}/method/fsgs_method_spec.py"

output_path = "output"

stack = ExitStack().__enter__()
stack.enter_context(backends.mount(output_path, output_path))

method_spec = get_method_spec(method_name)
from method.fsgs_method import GaussianSplatting
method_cls = GaussianSplatting

method_info = method_cls.get_method_info()
test_dataset = load_dataset(data, 
                             split="test", 
                             features=method_info.get("required_features"),
                             supported_camera_models=method_info.get("supported_camera_models"),
                             load_features=True)
presets, config_overrides = get_presets_and_config_overrides(
    method_spec, test_dataset["metadata"])

model = method_cls(
    train_dataset=test_dataset,
    config_overrides=config_overrides,
    checkpoint=checkpoint
)
# cam = test_dataset["cameras"][78]
# fx,fy = cam.intrinsics[0], cam.intrinsics[1]
# while True:
#     s = float(input())
#     cam.intrinsics[0] = s*fx
#     cam.intrinsics[1] = s*fy
#     img = model.render(cam)["color"]
#     import matplotlib.pyplot as plt
#     plt.figure()
#     plt.axis('off')
#     plt.imshow(img)
#     plt.savefig(f"img_90.png",bbox_inches="tight",pad_inches=0)

viewer = stack.enter_context(Viewer(
                model=model))
viewer.run()