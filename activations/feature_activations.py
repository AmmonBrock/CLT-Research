import torch
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import glob
import time
from safetensors.torch import save_file



def compute_feature_position_stats(config):

    # Set up parameters
    device = config.device
    d_sae = config.features_per_layer
    num_layers = config.n_layers
    max_length = config.max_tokens_activation
    n_text_examples = (config.n_tokens_act_freq // max_length) + (1 if config.n_tokens_act_freq % max_length != 0 else 0)
    num_batches_to_process = n_text_examples // config.activations_batch_size + (1 if n_text_examples % config.activations_batch_size != 0 else 0)



    # Load model and dataloader
    torch.set_grad_enabled(False)

    # The following imports have to be done after the environment variable is set from the config
    os.environ["HF_HUB_CACHE"] = config.model_storage_absolute
    from circuit_tracer import ReplacementModel
    from transformers import AutoTokenizer
    from data.dataloading import get_dataloader

    # Get the tokenizer
    cache_dir = os.environ["HF_HUB_CACHE"]
    snapshot_paths = glob.glob(f"{cache_dir}/models--{config.study_model_name.replace('/', '--')}/snapshots/*/")
    absolute_model_path = snapshot_paths[0]
    tokenizer = AutoTokenizer.from_pretrained(absolute_model_path, local_files_only=True)

    print("Loading ReplacementModel offline...")
    model = ReplacementModel.from_pretrained(
        model_name=config.study_model_name, 
        transcoder_set=config.feature_tool_name, 
        dtype=torch.bfloat16,
        device=device,
        local_files_only=True,
        tokenizer = tokenizer
    )
    dataloader = get_dataloader(tokenizer, activation = True, config = config)

    # Set up storage for activations
    positional_counts = torch.zeros((num_layers, max_length, d_sae), device = device, dtype = torch.int32)
    positional_strengths = torch.zeros((num_layers, max_length, d_sae), device = device, dtype = torch.float32)
    total_tokens = 0

    # Prepare hooks for activation tracking
    clt = model.transcoders 
    hook_name_base = clt.feature_input_hook
    hook_names = [f"blocks.{layer}.{hook_name_base}" for layer in range(clt.n_layers)]


    # Loop through dataloader and gather activations
    start = time.time()
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 100 == 0:
            print(f"Processing batch {batch_idx}/{num_batches_to_process}... (Tokens seen: {total_tokens})", flush=True)
            print(f"Average time per 100 batches: {(time.time() - start) / (batch_idx + 1):.2f} seconds")
        
        # Get the inputs
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_size, seq_len = input_ids.shape

        # Run forward pass while caching the necessary hooks
        _, cache = model.run_with_cache(
            input_ids,
            names_filter=lambda n: n in hook_names
        )
        activations_list = [cache[name] for name in hook_names]
        stacked_activations = torch.stack(activations_list)
        x = stacked_activations.reshape(clt.n_layers, batch_size * seq_len, clt.d_model) # shape (num_layers, batch_size * seq_len, d_model)

        # zero out the padding and BOS tokens
        tokens_to_zero_mask = (attention_mask == 0) # padding tokens
        tokens_to_zero_mask[:, 0] = True # BOS tokens
        flattened_mask = tokens_to_zero_mask.view(-1) # shape (batch_size * seq_len)
        zero_indices = flattened_mask.nonzero(as_tuple=True)[0].tolist()

        sparse_features, _ = clt.encode_sparse(x, zero_positions = zero_indices)
        sparse_features = sparse_features.coalesce()
        indices = sparse_features.indices() # shape (3, num_active)
        values = sparse_features.values() # shape (num_active,)
        layer_indices = indices[0]
        flat_seq_indices = indices[1]
        feature_indices = indices[2]
        n_layers, n_batch_times_seq_len, d_transcoder = sparse_features.size()

        # Work-around: Instead of converting to dense and then slicing, I compute a new set of indices corresponding to the token positions which will result in the coalesce function summing up the repeats
        seq_indices = flat_seq_indices % seq_len 
        new_indices = torch.stack([layer_indices, seq_indices, feature_indices], dim=0) # shape (3, num_active)
        new_shape = (n_layers, max_length, d_transcoder)
        sparse_strengths = torch.sparse_coo_tensor(new_indices, values, size=new_shape).coalesce()
        positional_strengths += sparse_strengths.to_dense()
        ones_values = torch.ones_like(values, dtype = torch.int32)
        sparse_counts = torch.sparse_coo_tensor(new_indices, ones_values, size=new_shape).coalesce()
        positional_counts += sparse_counts.to_dense()

        total_tokens += attention_mask.sum().item()

    activation_counts = positional_counts.sum(dim=1) # shape (num_layers, d_sae)
    end = time.time()
    print(f"Finished processing {num_batches_to_process} batches. Total tokens seen: {total_tokens}. Time taken: {end - start:.2f} seconds.")

    positional_path = config.feature_stats_on_corpus_dir / "feature_positional_counts.safetensors"
    strength_path = config.feature_stats_on_corpus_dir / "feature_positional_strengths.safetensors"
    activation_count_path = config.feature_stats_on_corpus_dir / "feature_activation_counts.safetensors"
    os.makedirs(config.feature_stats_on_corpus_dir, exist_ok=True)
    save_file({"positional_counts": positional_counts.cpu()}, str(positional_path))# shape (num_layers, max_length, d_sae)
    save_file({"positional_strengths": positional_strengths.cpu()}, str(strength_path)) # shape (num_layers, max_length, d_sae)
    save_file({"activation_counts": activation_counts.cpu()}, str(activation_count_path)) # shape (num_layers, d_sae)
    
    print("Done.")
    return True



        
