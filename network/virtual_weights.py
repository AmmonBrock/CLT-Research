import torch
from safetensors import safe_open
from safetensors.torch import save_file
import os
import numpy as np
import pandas as pd
import json
from tabulate import tabulate
import os
import glob
from pathlib import Path
import argparse
from configs.config_data import NetworkConfig


# Notes about implementation change
# 1. sample_indices no longer has the option to be None
# 2. weight_ prefix and sample_indices_path suffix removed from file names --> have to change the way they are saved
# 3. clt_path is now clt_model_path
# 4. weight_ prefix now removed from tensor name within the saved files (hopefully so the prefix param isn't needed in the VirtualWeightNeighbors class)

def compute_virtual_weights(config):

    device = config.device
    cache_dir = config.model_storage_absolute
    snapshot_paths = glob.glob(f"{cache_dir}/models--{config.feature_tool_name.replace('/', '--')}/snapshots/*/")
    assert len(snapshot_paths) > 0, f"No snapshots found for model {config.feature_tool_name} in cache directory {cache_dir}. Please make sure the model is downloaded and the path is correct."
    clt_model_path = Path(snapshot_paths[0])
    save_dir = config.virtual_weight_dir
    sample_indices_path = config.network_dir / "sampled_features.npy"

    save_dir.mkdir(parents=True, exist_ok=True)

    layers_to_analyze = range(config.n_layers)
    torch.set_grad_enabled(False)

    sample_indices = torch.from_numpy(np.load(sample_indices_path)).to(device)

    for source_layer in layers_to_analyze:
        targets_left = [t for t in layers_to_analyze if t > source_layer]
        if all ((save_dir / f"{source_layer}_{t}.safetensors").exists() for t in targets_left):
            print(f"All weights for source layer {source_layer} already computed, skipping.", flush = True)
            continue
        
        # Get the decoder from this source layer to all future layers
        dec_path = clt_model_path / f"W_dec_{source_layer}.safetensors"
        if not dec_path.exists():
            print(f"Decoder file for source layer {source_layer} not found at {dec_path}. Skipping.")
            continue
    
        with safe_open(dec_path, framework="pt", device = device) as f:
            tensor_name = f.keys()[0]
            w_dec_source = f.get_tensor(tensor_name)
            w_dec_source = w_dec_source[sample_indices[source_layer], : , :].bfloat16()
        assert w_dec_source.shape[1] == (len(layers_to_analyze) - source_layer), f"Expected decoder to have shape (d_sae, {len(layers_to_analyze) - source_layer}, d_model), but got {w_dec_source.shape}"

        for target_layer in layers_to_analyze:
            if source_layer >= target_layer:
                continue
            if (save_dir / f"{source_layer}_{target_layer}.safetensors").exists():
                print(f"Weight from layer {source_layer} to layer {target_layer} already computed, skipping.", flush = True)
                continue

            print(f"Computing virtual weight from layer {source_layer} to layer {target_layer}...", flush = True)
            save_path = save_dir / f"{source_layer}_{target_layer}.safetensors"
            enc_path = clt_model_path / f"W_enc_{target_layer}.safetensors"

            with safe_open(enc_path, framework="pt", device=device) as f:
                tensor_name = f.keys()[0]
                target_encoders = f.get_tensor(tensor_name)
                target_encoders = target_encoders[sample_indices[target_layer], :].bfloat16()

            # Sum decoders from source layer across intermediate layers
            summed_decoders = w_dec_source[:, :(target_layer - source_layer), :].sum(dim=1) # (d_sae_source, d_model)

            # Virtual weight matrix: (d_sae_source, d_sae_target)
            V = torch.einsum("sm, tm->st", summed_decoders, target_encoders)

            print(f"Shape: {V.shape}, Mean abs value: {V.abs().mean():.4f}")
            print(f"Saving virtual weights from source layer {source_layer} to target layer {target_layer}...")
            save_file({f"{source_layer}_{target_layer}": V.cpu()}, str(save_path))

            del target_encoders
            del summed_decoders
            del V
        del w_dec_source
        torch.cuda.empty_cache()

    print("\nDone! Virtual weights saved to folder (Index by source. Source = row, target = columns).")


def main():
    parser = argparse.ArgumentParser(description="Specify config file for feature network computation.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    clt_dir = Path(__file__).resolve().parent.parent
    config_path = clt_dir / "configs" / args.config
    config = NetworkConfig.from_yaml(config_path)
    config.validate_params()
    compute_virtual_weights(config)


if __name__ == "__main__":
    main()
