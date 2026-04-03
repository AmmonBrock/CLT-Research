from sample.filter_positional_features import filter_features
import numpy as np
from safetensors.torch import load_file
import torch
from activations.feature_activations import compute_feature_position_stats
import os
from matplotlib import pyplot as plt
import argparse
from pathlib import Path
from configs.config_data import NetworkConfig


def plot_activation_distribution(config, sampled_features = None, features_to_exclude = None, layer = None):
    """Plots activation distribution given the parameters.
    Args:
        config: CLTConfig object containing configuration parameters.
        sampled_features: Optional numpy array of shape (n_layers, n_samples_per_layer) containing indices of sampled features. If provided, plots distribution for these features.
        features_to_exclude: Optional tuple of (exclude_layers, exclude_indices) representing indices to filter out before plotting. Is ignored if sampled_feature is provided.
        layer: Optional integer representing the layer index for which to plot the distribution. If None, plots the distribution for all layers.
    """

    activation_count_path = config.feature_stats_on_corpus_dir / "feature_activation_counts.safetensors"
    counts = load_file(activation_count_path)["activation_counts"] # Shape: (config.n_layers, config.features_per_layer)
    if sampled_features is not None:
        # Plot distribution of activations for sampled features
        title = "Feature Activation Distribution for Sampled Features"
        save_file_name = "activation_distribution_sampled"
        if layer is not None:
            activation_frequencies_to_plot = {layer: counts[layer, sampled_features[layer]] / config.n_tokens_act_freq}
            
        else:
            activation_frequencies_to_plot = {}
            for i in range(config.n_layers):
                activation_frequencies_to_plot[i] = counts[i, sampled_features[i]] / config.n_tokens_act_freq
    elif features_to_exclude is not None:
        # Plot distribution of activations for filtered features
        title = "Feature Activation Distribution for Filtered Features"
        save_file_name = "activation_distribution_filtered"
        exclude_layers, exclude_indices = features_to_exclude
        if layer is not None:
            available_indices = np.setdiff1d(np.arange(config.features_per_layer), exclude_indices[exclude_layers == layer])
            activation_frequencies_to_plot = {layer: counts[layer, available_indices] / config.n_tokens_act_freq}
        else:
            activation_frequencies_to_plot = {}
            for i in range(config.n_layers):
                available_indices = np.setdiff1d(np.arange(config.features_per_layer), exclude_indices[exclude_layers == i])
                activation_frequencies_to_plot[i] = counts[i, available_indices] / config.n_tokens_act_freq
    else:
        title = "Feature Activation Distribution for All Features"
        save_file_name = "activation_distribution_all"
        # Plot distribution of activations for all features
        activation_frequencies_to_plot = {layer: counts[layer, :] / config.n_tokens_act_freq for layer in range(config.n_layers)}

    os.makedirs(config.network_dir / "figures", exist_ok=True)
    # Plot log10 distribution in subplots if necessary
    if layer is not None:
        plt.figure(figsize=(8, 6))
        plt.hist(np.log10(activation_frequencies_to_plot[layer] + 1e-10).cpu().numpy(), bins=30)
        plt.title(f"{title} - Layer {layer}")
        plt.xlabel("Log10 Activation Frequency")
        plt.ylabel("Number of Features")
        plt.grid()
        plt.savefig(config.network_dir / "figures" / f"{save_file_name}_layer_{layer}.png")
    else:
        n_cols = 4
        n_rows = (config.n_layers + n_cols - 1) // n_cols
        plt.figure(figsize=(n_cols * 5, n_rows * 4), layout="constrained")
        plt.suptitle(title, fontsize=16)
        for i in range(config.n_layers):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.hist(np.log10(activation_frequencies_to_plot[i] + 1e-10).cpu().numpy(), bins=30)
            plt.title(f"Layer {i}")
            plt.xlabel("Log10 Activation Frequency")
            plt.ylabel("Number of Features")
            plt.grid()
        plt.savefig(config.network_dir / "figures" / f"{save_file_name}{'_layer_'+str(layer) if layer is not None else ''}.png")

def sample_pipeline(config):
    if (config.network_dir / "sampled_features.npy").exists():
        print("Sampled features already exist, skipping sampling.", flush = True)
        return True
    
    if config.compute_activations:
        compute_feature_position_stats(config)

    print("Finished computing activations", flush = True)
    
    # Create pre-filtering graphs
    plot_activation_distribution(config)



    features_to_exclude = filter_features(config) if 'filtered' in config.sample_method else (np.array([]), np.array([]))
    exclude_layers, exclude_indices = features_to_exclude
    print(f"Found features to exclude...", flush = True)


    # Create post-filtering pre-sampling graphs
    plot_activation_distribution(config, features_to_exclude=features_to_exclude)

    
    sampled_features = np.zeros((0, config.n_samples_per_layer), dtype = np.int64)
    if 'proportional' in config.sample_method:
        print("Sampling proportionally", flush = True)
        activation_count_path = config.feature_stats_on_corpus_dir / "feature_activation_counts.safetensors"
        activation_count_tensors = load_file(activation_count_path)
        counts = activation_count_tensors["activation_counts"] # Shape: (config.n_layers, config.features_per_layer)

        for layer in range(config.n_layers):
            print(f"Sampling layer {layer}...", flush = True)
            available_indices = np.setdiff1d(np.arange(config.features_per_layer), exclude_indices[exclude_layers == layer])
            available_counts = counts[layer, available_indices].cpu().numpy()
            sum_counts = available_counts.sum()
            assert sum_counts > 0, f"Layer {layer} has no available features to sample from after filtering."
            available_proportions = available_counts / sum_counts
            sampled_indices = np.sort(np.random.choice(available_indices, size=config.n_samples_per_layer, replace=False, p=available_proportions))
            sampled_features = np.vstack((sampled_features, sampled_indices))
    else: 
        print("Sampling uniformly", flush = True)
        for layer in range(config.n_layers):
            print(f"Sampling layer {layer}...", flush = True)
            available_indices = np.setdiff1d(np.arange(config.features_per_layer), exclude_indices[exclude_layers == layer])
            sampled_indices = np.sort(np.random.choice(available_indices, size=config.n_samples_per_layer, replace=False))
            sampled_features = np.vstack((sampled_features, sampled_indices))



    # Create post-sampling graphs
    plot_activation_distribution(config, sampled_features=sampled_features)

    # Save sampled features to CLT/sample/network_name/sampled_features.npy
    os.makedirs(config.network_dir, exist_ok=True)
    np.save(config.network_dir / "sampled_features.npy", sampled_features)
    return True
    
def main():
    parser = argparse.ArgumentParser(description="Specify config file for feature sampling.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    clt_dir = Path(__file__).resolve().parent.parent
    config_path = clt_dir / "configs" / args.config
    config = NetworkConfig.from_yaml(config_path)
    config.validate_params()
    sample_pipeline(config)
    config.lock_sample_params()


if __name__ == "__main__":
    main()