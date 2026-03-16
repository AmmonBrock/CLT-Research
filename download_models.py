device = "cpu"

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import torch
import os





# Download and save the base model
torch.set_grad_enabled(False) # avoid blowing up mem
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-pt",
    device_map='auto',
)
tokenizer =  AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")

# Save to a local directory
model.save_pretrained("./models/gemma-3-1b-pt")
tokenizer.save_pretrained("./models/gemma-3-1b-pt")




# Create directory to store the weights
save_dir = "./models/gemma-scope-2-1b-pt-clt"
os.makedirs(save_dir, exist_ok=True)

# Download all layer params
num_layers = 26
category = "clt"
subcategory = "width_262k_l0_medium_affine"

for layer_idx in range(num_layers):
    filename = f"{category}/{subcategory}/params_layer_{layer_idx}.safetensors"
    
    # Download from HuggingFace
    downloaded_path = hf_hub_download(
        repo_id="google/gemma-scope-2-1b-pt",
        filename=filename,
    )
    
    # Copy to local directory with same structure
    local_path = os.path.join(save_dir, f"params_layer_{layer_idx}.safetensors")
    import shutil
    shutil.copy(downloaded_path, local_path)
    print(f"Saved layer {layer_idx} to {local_path}")

print(f"All files saved to {save_dir}")