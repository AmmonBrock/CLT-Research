import numpy as np

feature_layers, feature_indices = np.load("feature_filtering/features_to_exclude.npy")
max_index = 98304


def sample_features(num_samples = 1000):
    result = np.zeros((26, num_samples), dtype=np.int32)
    for layer in range(26):
        available_indices = np.setdiff1d(np.arange(max_index), feature_indices[feature_layers == layer])
        sampled_indices = np.sort(np.random.choice(available_indices, size=num_samples, replace=False))
        print(f"Layer {layer}: Sampled feature indices: {sampled_indices}")
        result[layer] = sampled_indices
    return result

if __name__ == "__main__":
    sampled_features = sample_features(num_samples=1000)
    np.save("feature_filtering/sampled_features_small.npy", sampled_features)
    # sampled_features_small = np.load("feature_filtering/sampled_features_small.npy")