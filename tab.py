from pathlib import Path
import sys
import json
p = Path(sys.argv[1])
#scenes = ["drums","lego","materials","mic","ship"]
#scenes = ["hotdog","ficus","chair"]
#scenes = ["bicycle","garden","stump","bonsai","counter"]
#scenes = ["alameda","london","nyc"]
scenes = ["chair"]

for s in scenes:
    for d in p.iterdir():
        if d.is_dir() and s in str(d):
            with open(f"{d}/eval/eval.json", "r", encoding="utf-8") as f:
                data = json.load(f)#["metrics"]
                print(f"{s} & "r"\hcell{50}{"f"{round(data['psnr'],2):.2f}""}"
                      f" & "r"\hcell{50}{"f"{round(data['ssim'],3):.3f}""}"
                      f" & "r"\hcell{50}{"f"{round(data['lpips'],3):.3f}""}") 

