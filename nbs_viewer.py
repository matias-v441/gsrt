from nerfbaselines.viewer import Viewer

from contextlib import ExitStack
from nerfbaselines import backends
from nerfbaselines.datasets import load_dataset

stack = ExitStack()

from nbs_method.gsrt_method import GSRTMethod
method_cls = GSRTMethod

method_info = method_cls.get_method_info()

train_dataset = load_dataset("external://blender/lego", 
                             split="train", 
                             load_features=False)

test_dataset = load_dataset("external://blender/lego", 
                            split="test", 
                            load_features=False)

import sys

model = GSRTMethod(
    checkpoint=f"data/lego/checkpoint/chkpnt-30000.pth",
    #checkpoint=f'gsrt_checkpoint/checkpoint_{sys.argv[1]}.pt',
    config_overrides={'_3dgs_data':True}
    )
model.rendering_setup()

viewer = stack.enter_context(Viewer(
    train_dataset=train_dataset, 
    test_dataset=test_dataset, 
    model=model))

viewer.run()
