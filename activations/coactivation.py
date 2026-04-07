import torch
from safetensors.torch import save_file
from safetensors import safe_open
import os
import gc
import resource
import time
import argparse
import numpy as np
from pathlib import Path
os.environ["HF_HUB_OFFLINE"] = "1"
from configs.config_data import NetworkConfig
import glob



def compute_coactivation_stats_for_layer(config, source_layer, mins_until_timeout):
    real_start = time.time()

    #Configure inputs
    device = config.device
    d_sae = config.features_per_layer
    max_target_layer = config.n_layers - 1
    save_dir = config.coacts_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    if os.path.exists(save_dir / f"coactivation_stats_layer_{source_layer}.safetensors"):
        print(f"Coactivation stats for source layer {source_layer} already exist, skipping computation.")
        return True
    
    sample_indices_path = config.network_dir / "sampled_features.npy"

    # Import hf related stuff with the correct hub cache
    cache_dir = config.model_storage_absolute
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    from circuit_tracer import ReplacementModel
    from transformers import AutoTokenizer
    from data.dataloading import get_dataloader

    # Load samples
    sample_indices = torch.from_numpy(np.load(sample_indices_path)).to(device)
    n_sampled = sample_indices.size(1)
    assert(n_sampled == config.n_samples_per_layer), f"Number of sampled features does not match config. Expected {config.n_samples_per_layer} but got {n_sampled}."

    torch.set_grad_enabled(False)

    print("Loading ReplacementModel offline...")
    snapshot_paths = glob.glob(f"{cache_dir}/models--{config.study_model_name.replace('/', '--')}/snapshots/*/")
    assert len(snapshot_paths) > 0, f"No snapshots found for model {config.study_model_name} in cache directory {cache_dir}. Please make sure the model is downloaded and the path is correct."
    absolute_model_path = snapshot_paths[0]
    tokenizer = AutoTokenizer.from_pretrained(absolute_model_path, local_files_only = True)
    model = ReplacementModel.from_pretrained(
        model_name = config.study_model_name,
        transcoder_set = config.feature_tool_name,
        dtype = torch.bfloat16,
        device = device,
        local_files_only = True,
        tokenizer = tokenizer)
    dataloader = get_dataloader(tokenizer, activation = False, config = config)
    clt = model.transcoders

    # Storage for statistics
    # we need: E[a_i], E[a_i * a_j] for all pairs
    expected_activations = torch.zeros(n_sampled, device = device, dtype = torch.float32)
    expected_coactivations = {(source_layer, target_layer): torch.zeros(n_sampled, n_sampled, device = device, dtype = torch.float32) for target_layer in range(source_layer + 1, max_target_layer + 1)}  # E[a_i * a_j] for pairs where i < j in layers
    expected_indicator_coactivations = {(source_layer, target_layer): torch.zeros(n_sampled, n_sampled, device = device, dtype = torch.float32) for target_layer in range(source_layer + 1, max_target_layer + 1)}  # E[1{a_j > 0} * a_i]
    total_tokens = 0

    print("Gather activations and co-activations for source layer", source_layer)
    hook_name_base = clt.feature_input_hook
    hook_names = [f"blocks.{layer}.{hook_name_base}" for layer in range(clt.n_layers)]

    start = -1
    total_batches = len(dataloader)
    for batch_idx, batch in enumerate(dataloader):
        
        # Get the inputs from the batch object
        if batch_idx % 100 == 0:
            print(f"Processing batch {batch_idx+1}/{len(dataloader)}...", flush = True)

        if batch_idx % 1000 == 5:
            # Logic to potentially cancel the job early if it's being too slow
            end = time.time()
            if start > 0:
                lap_time = end - start
                total_time = end - real_start
                time_per_batch_lap = (lap_time / 1000.)/60.
                time_per_batch_total = (total_time / batch_idx)/60.
                
                remaining_batches = total_batches - batch_idx
                lap_remaining_time_estimate = time_per_batch_lap * remaining_batches
                total_remaining_time_estimate = time_per_batch_total * remaining_batches

                true_remaining_time = mins_until_timeout - ((total_time / 60.))
                if (lap_remaining_time_estimate > true_remaining_time) and (total_remaining_time_estimate > true_remaining_time):
                    # Raise an error to cancel the job and avoid wasting resources
                    raise TimeoutError(f"Estimated remaining time ({lap_remaining_time_estimate:.2f} mins by lap time, {total_remaining_time_estimate:.2f} mins by total time) exceeds the provided timeout threshold ({true_remaining_time:.2f} mins). Cancelling job to avoid wasting resources.")
        
            start = time.time()
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

        # Filter fetaures to only the sampled ones
        n_layers, n_pos, d_transcoder = sparse_features.size()
        inverse_map = torch.full(
            (n_layers, d_transcoder),
            -1,
            dtype = torch.long,
            device = sparse_features.device
        )
        new_indices = torch.arange(n_sampled, device = sparse_features.device).expand(n_layers, n_sampled) # shape (n_layers, n_sampled) used to assign new sample indices
        inverse_map.scatter_(1, sample_indices, new_indices) # fills in the sample indices at the appropriate positions, leaving -1 for non-sampled features
        mapped_feature_ids = inverse_map[layer_ids, feature_ids] # gets the new sample indices for each active feature, or -1 if not sampled
        keep_mask = mapped_feature_ids != -1 
        filtered_layer_ids = layer_ids[keep_mask]
        filtered_pos_ids = pos_ids[keep_mask]
        filtered_feature_ids = mapped_feature_ids[keep_mask]
        filtered_values = values[keep_mask]

        # Rebuild sparse tensor with new indices. Now has shape (n_layers, total_tokens, n_sampled)
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
        if target_slice.start < target_slice.stop:
            # Skip coactivations logic if on the last layer
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
            del src, targets, coacts_batch, indicator, ind_coacts_batch
        del sampled_feature_activations
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
    save_path = save_dir /  f"coactivation_stats_layer_{source_layer}.safetensors"
    save_file(tensors_to_save, str(save_path))

    return True



def main():
    parser = argparse.ArgumentParser(description="Compute coactivation stats for a specific layer.")
    parser.add_argument("--source_layer", type=int, required = True, help="The layer for which to compute coactivation stats.")
    parser.add_argument("--config", type=str, required = True, help="Name of config yaml file")
    parser.add_argument("--mins", type=int, required = True, help="Time in minutes until the job times out" )
    args = parser.parse_args()
    clt_dir = Path(__file__).resolve().parent.parent
    config_path = clt_dir / "configs" / args.config
    config = NetworkConfig.from_yaml(config_path)
    config.validate_params()
    source_layer = args.source_layer
    compute_coactivation_stats_for_layer(config, source_layer, args.mins)
    config.lock_weight_params()
    


if __name__ == "__main__":
    main()