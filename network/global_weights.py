from network.virtual_weights import compute_virtual_weights
from safetensors.torch import save_file
from safetensors import safe_open
import argparse
from pathlib import Path
from configs.config_data import NetworkConfig

# Notes:
# Removed the TWERA_ prefix from tensor names and twera_prefix from save files

def compute_twera_weights(config):

    # Get old arguments from config
    virtual_weights_path = config.virtual_weight_dir
    coactivation_path = config.coacts_dir
    save_path = config.twera_dir
    layers_to_analyze = range(config.n_layers)

    save_path.mkdir(parents=True, exist_ok=True)

    for source_layer in layers_to_analyze:
        for target_layer in layers_to_analyze:
            if source_layer >= target_layer:
                continue
            if (save_path / f"{source_layer}_{target_layer}.safetensors").exists():
                print(f"TWERA weight from layer {source_layer} to layer {target_layer} already computed, skipping.", flush = True)
                continue
            
            with safe_open(f"{coactivation_path}/coactivation_stats_layer_{source_layer}.safetensors", framework="pt", device="cpu") as coact_f:
                E_ab = coact_f.get_tensor(f"E_ab_{source_layer}_{target_layer}")
            with safe_open(f"{coactivation_path}/coactivation_stats_layer_{target_layer}.safetensors", framework="pt", device="cpu") as coact_f:
                E_target = coact_f.get_tensor(f"E_a_{target_layer}")
            with safe_open(f"{virtual_weights_path}/{source_layer}_{target_layer}.safetensors", framework="pt", device="cpu") as vw_f:
                V = vw_f.get_tensor(f"{source_layer}_{target_layer}")
    

            E_target_safe = E_target.float().clamp(min=1e-10)  # Avoid division by zero
            coact_ratio = E_ab.float() / E_target_safe.unsqueeze(0)  # Broadcast E_target across the source dimension
            TWERA = coact_ratio * V.float()

            save_file({f"{source_layer}_{target_layer}": TWERA.half().cpu()}, str(save_path / f"{source_layer}_{target_layer}.safetensors"))

    return True

def compute_era_weights(config):
    virtual_weights_path = config.virtual_weight_dir
    coactivation_path = config.coacts_dir
    save_path = config.era_dir
    layers_to_analyze = range(config.n_layers)

    save_path.mkdir(parents=True, exist_ok=True)

    for source_layer in layers_to_analyze:
        for target_layer in layers_to_analyze:
            if source_layer >= target_layer:
                continue
            if (save_path / f"{source_layer}_{target_layer}.safetensors").exists():
                print(f"ERA weight from layer {source_layer} to layer {target_layer} already computed, skipping.", flush = True)
                continue
            
            with safe_open(f"{coactivation_path}/coactivation_stats_layer_{source_layer}.safetensors", framework="pt", device="cpu") as coact_f:
                E_ind_ab = coact_f.get_tensor(f"E_ind_ab_{source_layer}_{target_layer}")
            with safe_open(f"{virtual_weights_path}/{source_layer}_{target_layer}.safetensors", framework="pt", device="cpu") as vw_f:
                V = vw_f.get_tensor(f"{source_layer}_{target_layer}")

            ERA = E_ind_ab.float() * V.float()

            print(f"  Layer {source_layer} → {target_layer}: Mean ERA = {ERA.abs().mean():.6f}")
            save_file({f"{source_layer}_{target_layer}": ERA.half().cpu()}, str(save_path / f"{source_layer}_{target_layer}.safetensors"))
    return True

def main():
    parser = argparse.ArgumentParser(description="Specify config file for global weight computation.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    clt_dir = Path(__file__).resolve().parent.parent
    config_path = clt_dir / "configs" / args.config
    config = NetworkConfig.from_yaml(config_path)
    config.validate_params()

    if "twera" in config.to_compute:
        compute_twera_weights(config)
    if "era" in config.to_compute:
        compute_era_weights(config)
    if "twera" not in config.to_compute and "era" not in config.to_compute:
        print("No valid weight type specified in to_compute. Please include 'twera' or 'era' in the to_compute list in the config file.")
    
if __name__ == "__main__":
    main()
