"""Filter out positional features (especially BOS token features) as they can lead to erroneous feature explanations"""

import os
from safetensors.torch import load_file
import torch
import numpy as np

positional_file_path = "./fineweb_feature_stats/feature_positional_counts.safetensors"
positional_tensors = load_file(positional_file_path)
positional_counts = positional_tensors["positional_counts"]

p_strength_file_path = "./fineweb_feature_stats/feature_positional_strengths.safetensors"
p_strength_tensors = load_file(p_strength_file_path)
positional_strengths = p_strength_tensors["positional_strengths"]

file_path = "./fineweb_feature_stats/feature_activation_counts.safetensors"
tensors = load_file(file_path)
counts = tensors["activation_counts"] # Shape: (26, 98304)

def positional_strength_test(positional, position, threshold = .9):
    """Returns True if the feature has over a threshold of its weight on a single token position, indicating it's likely a positional feature."""
    return (positional[:, :, position] / (positional.sum(axis=2) + 1e-9) > threshold)

def activation_count_test(positional_counts, position, threshold = .8, num_samples = 8000):
    """Returns True if the feature is active in a position over a threshold percentage of the time"""
    proportion_of_activations_per_sample = (positional_counts[:, :, position] / num_samples)
    greater_than_lower_bound = (proportion_of_activations_per_sample > threshold)
    less_than_upper_bound = (proportion_of_activations_per_sample < (2-threshold))
    return (greater_than_lower_bound & less_than_upper_bound)


def is_positional_feature(positional_counts, positional_strengths, strength_threshold = .9, count_threshold = .8, num_samples = 8000):
    result = torch.zeros((positional_counts.shape[0], positional_counts.shape[1]), dtype=torch.bool)
    for position in range(positional_counts.shape[2]):
        strength_test = positional_strength_test(positional_strengths, position, threshold=strength_threshold)
        count_test = activation_count_test(positional_counts, position, threshold=count_threshold, num_samples=num_samples)
        result |= (strength_test & count_test)
    return result

def is_nearly_dead_feature(counts, threshold = 10):
    """Returns True if the feature is active less than a threshold number of times, indicating it's nearly dead and should be filtered out."""
    return (counts < threshold)

yes_positional = is_positional_feature(positional_counts, positional_strengths, strength_threshold = .9, count_threshold = .8, num_samples = 8000)
nearly_dead = is_nearly_dead_feature(counts, threshold=10)

positional_or_dead = (yes_positional | nearly_dead)

features_to_exclude = np.where(positional_or_dead.numpy())
np.save("feature_filtering/features_to_exclude.npy", features_to_exclude)

# feature_layers, feature_indices = np.load("feature_filtering/features_to_exclude.npy")