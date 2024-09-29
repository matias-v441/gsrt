from nerfbaselines.viewer import run_viser_viewer
from nbs_method.gsrt_method import GSRTMethod

method = GSRTMethod(checkpoint="data/lego/checkpoint/chkpnt-30000.pth")

run_viser_viewer(method)