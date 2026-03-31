import torch
from safetensors.torch import save_file
from safetensors import safe_open
import os
import gc
import resource
import time
import argparse
import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = "/home/ammonbro/CLT/models/round2" 

from circuit_tracer import ReplacementModel
from CLT.activations.dataloading import get_fineweb_dataloader




def compute_coactivation_stats_for_layer(
        device='cuda', 
        d_sae=98304, 
        source_layer=0, 
        max_target_layer = 25, 
        save_dir="./stats_checkpoints",
        sample_indices_path = "feature_filtering/sampled_features_small.npy"):
    """
    Compute expected activations and co-activations for a given source layer to all future target layers
    
    Args:
        device: Device to run computations on
        d_sae: SAE feature dimension
        source_layer: Source layer to compute stats for
        max_target_layer: Maximum target layer to compute stats for
        save_dir: Where to save the statistics
    """
    if not save_dir:
        raise ValueError("save_dir must be provided to save co-activation statistics.")

    sample_indices = torch.from_numpy(np.load(sample_indices_path)).to(device) if sample_indices_path else None
    if sample_indices is None:
        assert(d_sae < 14000), "If not using sample indices, d_sae must be less than 14000 to avoid OOM issues."
        n_sampled=d_sae
    else:
        n_sampled = sample_indices.size(1)



    os.makedirs(save_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    print("Loading ReplacementModel offline...")
    model = ReplacementModel.from_pretrained(
        model_name = "google/gemma-2-2b",
        transcoder_set = "mntss/clt-gemma-2-2b-2.5M",
        dtype=torch.bfloat16,
        device = device,
        local_files_only = True,
    )
    tokenizer = model.tokenizer
    # dataloader = get_fineweb_dataloader(tokenizer, data_path="data/fineweb_10bt_offline", batch_size=64, max_length=128)
    dataloader = get_fineweb_dataloader(tokenizer, data_path="data/fineweb_10bt_offline", batch_size=32, max_length=128) 
    clt = model.transcoders if not isinstance(model.transcoders, dict) else list(model.transcoders.values())[0]

    
    # Storage for statistics
    # we need: E[a_i], E[a_i * a_j] for all pairs
    expected_activations = torch.zeros(n_sampled, device = device, dtype = torch.float32)  # E[a_i] for each feature
    expected_coactivations = {(source_layer, target_layer): torch.zeros(n_sampled, n_sampled, device = device, dtype = torch.float32) for target_layer in range(source_layer + 1, max_target_layer + 1)}  # E[a_i * a_j] for pairs where i < j in layers
    expected_indicator_coactivations = {(source_layer, target_layer): torch.zeros(n_sampled, n_sampled, device = device, dtype = torch.float32) for target_layer in range(source_layer + 1, max_target_layer + 1)}  # E[1{a_j > 0} * a_i]
    total_tokens = 0

    
    print("Gather activations and co-activations for source layer", source_layer)
    hook_name_base = clt.feature_input_hook
    hook_names = [f"blocks.{layer}.{hook_name_base}" for layer in range(clt.n_layers)]


    start = time.time() 
    for batch_idx, batch in enumerate(dataloader):

        # Get the inputs from the batch object
        if batch_idx % 100 == 0:
            print(f"Processing batch {batch_idx+1}/{len(dataloader)}...", flush = True)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device) 
        batch_size, seq_len = input_ids.shape

        # Run forward pass while caching the necessary hooks
        _, cache = model.run_with_cache(
            input_ids,
            names_filter = lambda n: n in hook_names
        )
        activations_list = [cache[name] for name in hook_names]
        stacked_activations = torch.stack(activations_list) 
        x = stacked_activations.reshape(clt.n_layers, batch_size * seq_len, clt.d_model) # shape (num_layers, batch_size * seq_len, d_model)

        # Before encoding, make sure we zero out the padding and the BOS token
        tokens_to_zero_mask = (attention_mask == 0) # padding tokens
        tokens_to_zero_mask[:, 0] = True # BOS token
        flattened_mask = tokens_to_zero_mask.view(-1)
        zero_indices = flattened_mask.nonzero(as_tuple=True)[0].tolist()

        # Encode into features
        sparse_features, active_encoders = clt.encode_sparse(x, zero_positions = zero_indices)
        sparse_features = sparse_features.coalesce()  # Ensure indices are sorted and duplicates are summed
        indices = sparse_features.indices() # shape (3, num_active)
        values = sparse_features.values() # shape (num_active,)
        layer_ids = indices[0]
        pos_ids = indices[1]
        feature_ids = indices[2]
        n_layers, n_pos, d_transcoder = sparse_features.size()

        if sample_indices is not None:
            # Filter features to only the sampled ones
            # Shape: (26, 98304) initialized to False
            n_layers, n_pos, d_transcoder = sparse_features.size()

            # Create a mask of shape (26, 98304) where it is -1 if the feature is not sampled and the "sample_index" if it is sampled
            inverse_map = torch.full(
                (n_layers, d_transcoder), 
                -1, 
                dtype=torch.long, 
                device=sparse_features.device
            )
            new_indices = torch.arange(n_sampled, device=sparse_features.device).expand(n_layers, n_sampled) # shape (26, n_sampled) used to assign new sample indices
            inverse_map.scatter_(1, sample_indices, new_indices) # fills in the sample indices at the appropriate positions, leaving -1 for non-sampled features
            mapped_feature_ids = inverse_map[layer_ids, feature_ids] # gets the new sample indices for each active feature, or -1 if not sampled
            keep_mask = mapped_feature_ids != -1 
            filtered_layer_ids = layer_ids[keep_mask]
            filtered_pos_ids = pos_ids[keep_mask]
            filtered_feature_ids = mapped_feature_ids[keep_mask]
            filtered_values = values[keep_mask]

            # Rebuild sparse tensor with new indices. Now has shape (26, total_tokens, n_sampled)
            new_indices_tensor = torch.stack([filtered_layer_ids, filtered_pos_ids, filtered_feature_ids])
            sparse_features = torch.sparse_coo_tensor(
                new_indices_tensor,
                filtered_values,
                size = (n_layers, n_pos, n_sampled),
                device = sparse_features.device,
                dtype=sparse_features.dtype
            ).coalesce()
        

        # Convert to dense tensor of shape (num_layers, total_tokens, n_sampled)
        sampled_feature_activations = sparse_features.to_dense()

        expected_activations += sampled_feature_activations[source_layer].sum(dim=0).float()
        total_tokens += (attention_mask.sum().item() - batch_size) # Exclude BOS tokens


        # Compute co-activations from source_layer to all future target layers
        target_slice = slice(source_layer + 1, max_target_layer + 1)
        if target_slice.start >= target_slice.stop:
            continue
        src = sampled_feature_activations[source_layer].float()  # shape (total_tokens, n_source_features)
        targets = sampled_feature_activations[target_slice].float()  # shape (num_targets, total_tokens, n_target_features)
        coacts_batch = torch.einsum("ns,lnt->lst", src, targets)  # shape (num_targets, n_source_features, n_target_features)
        indicator = (targets > 0).float()  # shape (num_targets, total_tokens, n_target_features)
        ind_coacts_batch = torch.einsum("ns,lnt->lst", src, indicator)  # shape (num_targets, n_source_features, n_target_features)

        # Sum into the running totals for each target layer
        for i, target_layer in enumerate(range(source_layer + 1, max_target_layer + 1)):
            key = (source_layer, target_layer)
            expected_coactivations[key] += coacts_batch[i].float()
            expected_indicator_coactivations[key] += ind_coacts_batch[i].float()
        del src, targets, coacts_batch, indicator, ind_coacts_batch, sampled_feature_activations
        torch.cuda.empty_cache()
        

    end = time.time()
    
    print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"Current VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Time elapsed: {end - start:.2f} seconds")

    
    # Normalize and save
    safe_tokens = max(total_tokens, 1)  # Avoid division by zero
    tensors_to_save = {}
    normalized = (expected_activations / safe_tokens).half().cpu()
    tensors_to_save[f"E_a_{source_layer}"] = normalized
    del expected_activations


    keys = list(expected_coactivations.keys())
    for key in keys:
        normalized_co = (expected_coactivations[key] / safe_tokens).half().cpu()
        tensors_to_save[f"E_ab_{key[0]}_{key[1]}"] = normalized_co
        del expected_coactivations[key]

        normalized_ind_co = (expected_indicator_coactivations[key] / safe_tokens).half().cpu()
        tensors_to_save[f"E_ind_ab_{key[0]}_{key[1]}"] = normalized_ind_co
        del expected_indicator_coactivations[key]
    
    peak_cpu_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # Convert KB to GB
    print(f"Peak CPU RAM: {peak_cpu_ram:.2f} GB")
    torch.cuda.empty_cache()
    gc.collect()
    save_file(tensors_to_save, f"{save_dir}/coactivation_stats_layer_{source_layer}.safetensors")
    

    





def compute_TWERA_weights(virtual_weights_path="./virtual_weights",
                          coactivation_path="./fineweb_feature_stats/small_coactivations",
                          save_path="./twera",
                          layers_to_analyze=range(26)):
    """
    Compute Target-Weighted Expected Residual Attribution (TWERA) weights.
    TWERA_{ij} = (E[a_j * a_i] / E[a_j]) * V_{ij}
    
    Args:
        virtual_weights_path: Path to virtual weights
        coactivation_path: Path to co-activation statistics
        save_path: Where to save TWERA weights
        layers_to_analyze: Layers to compute TWERA for
    """
    
    print("Computing TWERA weights...")
    os.makedirs(save_path, exist_ok=True)

    for source_layer in layers_to_analyze:
        for target_layer in layers_to_analyze:
            if source_layer >= target_layer:
                continue

            with safe_open(f"{coactivation_path}/coactivation_stats_layer_{source_layer}.safetensors", framework="pt", device="cpu") as coact_f:
                E_ab = coact_f.get_tensor(f"E_ab_{source_layer}_{target_layer}")
            with safe_open(f"{coactivation_path}/coactivation_stats_layer_{target_layer}.safetensors", framework="pt", device="cpu") as coact_f:
                E_target = coact_f.get_tensor(f"E_a_{target_layer}")
            with safe_open(f"{virtual_weights_path}/weight_{source_layer}_{target_layer}_sampled_features_small.safetensors", framework="pt", device="cpu") as vw_f:
                V = vw_f.get_tensor(f"weight_{source_layer}_{target_layer}")

            E_target_safe = E_target.float().clamp(min=1e-10)  # Avoid division by zero
            coact_ratio = E_ab.float() / E_target_safe.unsqueeze(0)  # Broadcast E_target across the source dimension
            TWERA = coact_ratio * V.float()

            print(f"  Layer {source_layer} → {target_layer}: Mean TWERA = {TWERA.abs().mean():.6f}")
            save_file({f"TWERA_{source_layer}_{target_layer}": TWERA.half().cpu()}, f"{save_path}/twera_{source_layer}_{target_layer}.safetensors")
    

def compute_ERA_weights(virtual_weights_path="./virtual_weights",
                        coactivation_path="./fineweb_feature_stats/small_coactivations",
                        save_path="./era_weights",
                        layers_to_analyze=range(26)):
    """
    Compute Expected Residual Attribution (ERA) weights.
    ERA_{ij} = E[1{a_j > 0} * a_i] * V_{ij}
    
    Args:
        virtual_weights_path: Path to virtual weights
        coactivation_path: Path to co-activation statistics
        save_path: Where to save ERA weights
        layers_to_analyze: Layers to compute ERA for
    """
    os.makedirs(save_path, exist_ok=True)

    for source_layer in layers_to_analyze:
        for target_layer in layers_to_analyze:
            if source_layer >= target_layer:
                continue

            with safe_open(f"{coactivation_path}/coactivation_stats_layer_{source_layer}.safetensors", framework="pt", device="cpu") as coact_f:
                E_ind_ab = coact_f.get_tensor(f"E_ind_ab_{source_layer}_{target_layer}")
            with safe_open(f"{virtual_weights_path}/weight_{source_layer}_{target_layer}_sampled_features_small.safetensors", framework="pt", device="cpu") as vw_f:
                V = vw_f.get_tensor(f"weight_{source_layer}_{target_layer}")

            ERA = E_ind_ab.float() * V.float()

            print(f"  Layer {source_layer} → {target_layer}: Mean ERA = {ERA.abs().mean():.6f}")
            save_file({f"ERA_{source_layer}_{target_layer}": ERA.half().cpu()}, f"{save_path}/era_{source_layer}_{target_layer}.safetensors")


def main():
    parser = argparse.ArgumentParser(description="Compute coactivation stats for a specific layer.")
    parser.add_argument("source_layer", type=int, choices = range(26), help="The source layer for which to compute coactivation stats.")
    args = parser.parse_args()
    compute_coactivation_stats_for_layer(device = "cuda", d_sae = 98304, source_layer = args.source_layer, save_dir="./fineweb_feature_stats/small_coactivations_150M", sample_indices_path = "feature_filtering/sampled_features_small.npy")
    

if __name__ == "__main__":
    # main()
    compute_TWERA_weights(virtual_weights_path = "./virtual_weights", coactivation_path = "./fineweb_feature_stats/small_coactivations_150M", save_path = "./twera_small_sample_150M", layers_to_analyze=range(26))