import torch

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = "/home/ammonbro/CLT/models/round2" 

from circuit_tracer import ReplacementModel

device = "cuda"
model = ReplacementModel.from_pretrained(
    model_name="google/gemma-2-2b", 
    transcoder_set="mntss/clt-gemma-2-2b-2.5M", 
    dtype=torch.bfloat16,
    device=device,
    local_files_only=True
)

print(dir(model))

del model