#%%
from contextlib import ExitStack
stack = ExitStack().__enter__()

import pprint
from nerfbaselines import load_checkpoint

checkpoint_path=f"data/lego/checkpoint/chkpnt-30000.pth"

# Start the conda backend and load the checkpoint
model, _ = stack.enter_context(load_checkpoint(checkpoint_path, backend="conda"))

# Print model information
pprint.pprint(model.get_info())

#%%

from nerfbaselines import new_cameras, camera_model_to_int
import numpy as np
from PIL import Image

# Camera parameters 
# pose: a 3x4 matrix representing the camera pose as camera-to-world transformation
pose = np.array([
    [ 0.981, -0.026,  0.194, -0.463],
    [ 0.08,   0.958, -0.275,  2.936],
    [-0.179,  0.286,  0.941, -3.12 ]], dtype=np.float32)

# Camera intrinsics
fx, fy, cx, cy = 481, 481, 324, 210
# Image resolution
w, h =  648, 420

# Create camera object
camera = new_cameras(
    poses=pose,
    intrinsics=np.array([fx, fy, cx, cy], dtype=np.float32),
    image_sizes=np.array([w, h], dtype=np.int32),
    camera_models=np.array(camera_model_to_int("pinhole"), dtype=np.int32),
)

# Render the image
outputs = model.render(camera=camera, options={"output_type_dtypes": {"color": "uint8"}})

# Display the rendered image
im = Image.fromarray(outputs["color"])
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(im)
plt.axis('off')
plt.show()