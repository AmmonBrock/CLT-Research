from safetensors.torch import load_file
import torch
import numpy as np
import os


def positional_strength_test(positional, position, threshold = .9):
    """Returns True if the feature has over a threshold of its weight on a single token position, indicating it's likely a positional feature."""
    return (positional[:, position, :] / (positional.sum(axis=1) + 1e-9) > threshold)

def activation_count_test(positional_counts, position, threshold = .8, num_samples = 8000):
    """Returns True if the feature is active in a position over a threshold percentage of the time"""
    proportion_of_activations_per_sample = (positional_counts[:, position, :] / num_samples)
    greater_than_lower_bound = (proportion_of_activations_per_sample > threshold)
    less_than_upper_bound = (proportion_of_activations_per_sample < (2-threshold))
    return (greater_than_lower_bound & less_than_upper_bound)


def is_positional_feature(positional_counts, positional_strengths, strength_threshold = .9, count_threshold = .8, num_samples = 8000):
    n_layers, context_length, d_sae = positional_counts.shape

    result = torch.zeros((n_layers, d_sae), dtype=torch.bool)
    for position in range(context_length):
        strength_test = positional_strength_test(positional_strengths, position, threshold=strength_threshold)
        count_test = activation_count_test(positional_counts, position, threshold=count_threshold, num_samples=num_samples)
        result |= (strength_test & count_test)
    return result

def is_nearly_dead_feature(counts, threshold = 10):
    """Returns True if the feature is active less than a threshold number of times, indicating it's nearly dead and should be filtered out."""
    return (counts < threshold)

def filter_features(config):
    positional_file_path = config.feature_stats_on_corpus_dir / "feature_positional_counts.safetensors"
    positional_tensors = load_file(positional_file_path)
    positional_counts = positional_tensors["positional_counts"]
    print("Loaded positional counts", flush = True)

    p_strength_file_path = config.feature_stats_on_corpus_dir / "feature_positional_strengths.safetensors"
    p_strength_tensors = load_file(p_strength_file_path)
    positional_strengths = p_strength_tensors["positional_strengths"]
    print("Loaded positional strengths", flush = True)

    activation_count_file_path = config.feature_stats_on_corpus_dir / "feature_activation_counts.safetensors"
    activation_count_tensors = load_file(activation_count_file_path)
    counts = activation_count_tensors["activation_counts"] # Shape: (26, 98304)
    print("Loaded activation counts", flush = True)

    yes_positional = is_positional_feature(positional_counts, positional_strengths, strength_threshold = config.positional_strength_threshold, count_threshold = config.positional_count_threshold, num_samples=config.n_tokens_act_freq)
    num_positional = yes_positional.sum(dim=1)

    print("Computed positional feature mask", flush = True)
    nearly_dead = is_nearly_dead_feature(counts, threshold=config.dead_threshold)
    num_nearly_dead = nearly_dead.sum(dim=1)
    print("Computed nearly dead feature mask", flush = True)
    positional_or_dead = (yes_positional | nearly_dead)
    num_positional_or_dead = positional_or_dead.sum(dim=1)
    print("Computed positional or dead feature mask", flush = True)
    features_to_exclude = np.where(positional_or_dead.numpy())
    print("Computed features to exclude", flush = True)

    # Save excluding info for later analysis
    os.makedirs(config.network_dir / "filter_history", exist_ok=True)
    with open(config.network_dir / "filter_history" / f"feature_filtering_info.txt", "w") as f:
        f.write(f"positional features filtered: {num_positional.tolist()}\n")
        f.write(f"nearly dead features filtered: {num_nearly_dead.tolist()}\n")
        f.write(f"positional or dead features filtered: {num_positional_or_dead.tolist()}\n")
    return features_to_exclude

