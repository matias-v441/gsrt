from nerfbaselines.viewer import run_viser_viewer
from nbs_method.gsrt_method import GSRTMethod
import sys
scene = sys.argv[1]

method = GSRTMethod(checkpoint=f"data/{scene}/checkpoint/chkpnt-30000.pth")

run_viser_viewer(method)