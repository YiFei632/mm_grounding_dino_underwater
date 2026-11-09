"""Start the reviewed viewer on an unused server port."""

import json
import os
from pathlib import Path
import socket
import subprocess

root = Path("/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection")
port = 7862
while port < 7900:
    with socket.socket() as sock:
        try:
            sock.bind(("0.0.0.0", port))
            break
        except OSError:
            port += 1
else:
    raise RuntimeError("No available viewer port.")
folder = root / "output/encoder_response_viewer_checks"
folder.mkdir(parents=True, exist_ok=True)
config = root / (
    "work_dirs/oag_ablation/contrastive_loss/02_add_sonar_rgb/"
    "rgbs20_finetune_12e/rgbs20_finetune_12e.py")
checkpoint = config.with_name("epoch_12.pth")
command = [
    "/home/fishyu/anaconda3/envs/openmmlab/bin/python",
    str(root / "tools/analysis_tools/initial_query_response_web.py"),
    str(config), str(checkpoint), "--device", "cuda:0",
    "--host", "0.0.0.0", "--port", str(port),
]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", TOKENIZERS_PARALLELISM="false")
with (folder / "web.log").open("ab") as log:
    process = subprocess.Popen(
        command, cwd=str(root), env=env, stdout=log, stderr=log,
        stdin=subprocess.DEVNULL, start_new_session=True)
record = dict(
    pid=process.pid, port=port, url="http://162.105.195.38:{}".format(port),
    command=command, physical_gpu=1)
(folder / "running_viewer.json").write_text(
    json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2), flush=True)
