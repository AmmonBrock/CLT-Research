from local_load import get_model_and_tokenizer
from dataloading import get_dummy_dataloader
from gather_activations import gather_clt_activations
import torch
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