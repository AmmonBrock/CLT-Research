import os
import torch
import time
from safetensors.torch import save_file

# Strictly offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = "/home/ammonbro/CLT/models/round2" 

from circuit_tracer import ReplacementModel
from dataloading import get_fineweb_dataloader

def compute_feature_activation_counts(
    device='cuda', 
    d_sae=98304, 
    num_layers=26,
    num_batches_to_process=2000, 
    save_dir="./fineweb_feature_stats"
):
    os.makedirs(save_dir, exist_ok=True)
    torch.set_grad_enabled(False)
    
    print("Loading ReplacementModel offline...")
    model = ReplacementModel.from_pretrained(
        model_name="google/gemma-2-2b", 
        transcoder_set="mntss/clt-gemma-2-2b-2.5M", 
        dtype=torch.bfloat16,
        device=device,
        local_files_only=True
    )
    
    tokenizer = model.tokenizer
    dataloader = get_fineweb_dataloader(tokenizer, data_path="data/fineweb_10bt_offline", batch_size=16, max_length=128)
    
    activation_counts = torch.zeros((num_layers, d_sae), device=device, dtype=torch.int32)
    total_tokens = 0
    
    clt = model.transcoders if not isinstance(model.transcoders, dict) else list(model.transcoders.values())[0]

    print(f"Starting activation count over {num_batches_to_process} batches...")
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches_to_process:
            break
            
        if batch_idx % 50 == 0:
            print(f"Processing batch {batch_idx}/{num_batches_to_process}... (Tokens seen: {total_tokens})", flush=True)
            
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device) 
        
        # 1. Run forward pass, strictly caching only the explicitly defined input hook
        _, cache = model.run_with_cache(
            input_ids,
            names_filter=lambda n: n.endswith(".hook_resid_mid")
        )
        
        mask_expanded = attention_mask.unsqueeze(-1)  
        
        # 2. Iterate layer-by-layer directly from the known hook
        for layer in range(num_layers):
            resid_stream = cache[f"blocks.{layer}.hook_resid_mid"]
            
            W_enc = clt.W_enc[layer] 
            b_enc = clt.b_enc[layer] 
            
            latents = torch.relu(resid_stream @ W_enc.T + b_enc)
            
            masked_latents = latents * mask_expanded  
            fired_mask = (masked_latents > 0).int()
            
            activation_counts[layer] += fired_mask.sum(dim=(0, 1))
            
            del resid_stream, latents, masked_latents, fired_mask
            
        total_tokens += attention_mask.sum().item()
        
        del cache, input_ids, attention_mask, mask_expanded
        torch.cuda.empty_cache()

    end_time = time.time()
    print(f"\nFinished! Processed {total_tokens} valid tokens in {end_time - start_time:.2f} seconds.")
    
    save_path = os.path.join(save_dir, "feature_activation_counts.safetensors")
    print(f"Saving counts to {save_path}...")
    save_file({"activation_counts": activation_counts.cpu()}, save_path)
    print("Done.")

def compute_feature_positional_counts(
    device='cuda', 
    d_sae=98304, 
    num_layers=26,
    max_length=128, # Added to define the sequence dimension
    num_batches_to_process=2000, 
    track_activation_strength = False,
    save_dir="./fineweb_feature_stats"
):
    os.makedirs(save_dir, exist_ok=True)
    torch.set_grad_enabled(False)
    
    print("Loading ReplacementModel offline...")
    model = ReplacementModel.from_pretrained(
        model_name="google/gemma-2-2b", 
        transcoder_set="mntss/clt-gemma-2-2b-2.5M", 
        dtype=torch.bfloat16,
        device=device,
        local_files_only=True
    )
    
    tokenizer = model.tokenizer
    dataloader = get_fineweb_dataloader(
        tokenizer, 
        data_path="data/fineweb_10bt_offline", 
        batch_size=16, 
        max_length=max_length
    )
    
    positional_counts = torch.zeros((num_layers, d_sae, max_length), device=device, dtype=torch.int32)
    if track_activation_strength:
        positional_strengths = torch.zeros((num_layers, d_sae, max_length), device=device, dtype=torch.float32)
    total_tokens = 0
    
    clt = model.transcoders if not isinstance(model.transcoders, dict) else list(model.transcoders.values())[0]

    print(f"Starting positional count over {num_batches_to_process} batches...")
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches_to_process:
            break
            
        if batch_idx % 50 == 0:
            print(f"Processing batch {batch_idx}/{num_batches_to_process}... (Tokens seen: {total_tokens})", flush=True)
            
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device) 
        
        # 1. Run forward pass, strictly caching only the explicitly defined input hook
        _, cache = model.run_with_cache(
            input_ids,
            names_filter=lambda n: n.endswith(".hook_resid_mid")
        )
        
        mask_expanded = attention_mask.unsqueeze(-1)  
        
        # 2. Iterate layer-by-layer directly from the known hook
        for layer in range(num_layers):
            resid_stream = cache[f"blocks.{layer}.hook_resid_mid"]
            
            W_enc = clt.W_enc[layer] 
            b_enc = clt.b_enc[layer] 
            
            latents = torch.relu(resid_stream @ W_enc.T + b_enc)
            
            masked_latents = latents * mask_expanded  
            fired_mask = (masked_latents > 0).int()
            
            positional_counts[layer] += fired_mask.sum(dim=0).transpose(0, 1)
            if track_activation_strength:
                # Sum activation values (not just binary fired) over the batch dimension
                positional_strengths[layer] += masked_latents.sum(dim=0).transpose(0, 1).float()
            
            del resid_stream, latents, masked_latents, fired_mask
            
        total_tokens += attention_mask.sum().item()
        
        del cache, input_ids, attention_mask, mask_expanded
        torch.cuda.empty_cache()

    end_time = time.time()
    print(f"\nFinished! Processed {total_tokens} valid tokens in {end_time - start_time:.2f} seconds.")
    
    save_path = os.path.join(save_dir, "feature_positional_counts.safetensors")
    print(f"Saving positional counts to {save_path}...")
    save_file({"positional_counts": positional_counts.cpu()}, save_path)
    if track_activation_strength:
        save_file({"positional_strengths": positional_strengths.cpu()}, save_path.replace("feature_positional_counts", "feature_positional_strengths"))
    print("Done.")

if __name__ == "__main__":
    # compute_feature_activation_counts(num_batches_to_process = 1000)
    compute_feature_positional_counts(num_batches_to_process = 500, track_activation_strength=True)