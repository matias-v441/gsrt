from nerfbaselines.viewer import Viewer
from contextlib import ExitStack
from nbs_method.gsrt_method import GSRTMethod
from nerfbaselines.datasets import load_dataset
import sys
scene = sys.argv[1] if len(sys.argv)>1 else "lego"

method_info = GSRTMethod.get_method_info()

data_path = "external://blender/lego"
chpt = f"data/{scene}/checkpoint/chkpnt-30000.pth"
#chpt = f"drums_checkpoint/checkpoint_30000.pt"

train_dataset = load_dataset(data_path, 
                             split="train", 
                             features=method_info.get("required_features"),
                             supported_camera_models=method_info.get("supported_camera_models"),
                             load_features=True)

method = GSRTMethod(checkpoint=chpt, train_dataset=train_dataset)

with ExitStack() as stack:
    viewer = stack.enter_context(Viewer(model=method))
    viewer.run()