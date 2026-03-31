import torch
import glob
import os

os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = "/home/ammonbro/CLT/models/round2" 

from circuit_tracer import ReplacementModel
from transformers import AutoTokenizer

device = "cuda"

# Get the absolute path to the model
cache_dir = os.environ["HF_HUB_CACHE"]
snapshot_paths = glob.glob(f"{cache_dir}/models--google--gemma-2-2b/snapshots/*")
absolute_model_path = snapshot_paths[0]

tokenizer = AutoTokenizer.from_pretrained(absolute_model_path, local_files_only=True)
model = ReplacementModel.from_pretrained(
    model_name="google/gemma-2-2b", 
    transcoder_set="mntss/clt-gemma-2-2b-2.5M", 
    dtype=torch.bfloat16,
    device=device,
    local_files_only=True,
    tokenizer=tokenizer)
from data.dataloading import get_fineweb_dataloader
dataloader = get_fineweb_dataloader(tokenizer, data_path = "data/fineweb_10bt_offline", batch_size=16, max_length = 128, device = "cuda", start_index = 0, end_index = 12800)

for i, batch in enumerate(dataloader):
    print(i)
del model
