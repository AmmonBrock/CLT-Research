import torch
from safetensors import safe_open
import os
import numpy as np
import pandas as pd
from configs.config_data import NetworkConfig
from launch_pipeline import load_config

class NetworkNeighbors():
    def __init__(self, config:str | NetworkConfig, root_folder_name = "twera"):
        """Class to load and query weights between features in different layers. 
        
        Args:
            config: Either a NetworkConfig object or a string path of the config file (relative from the confis directory)
            sample_indices_path: Path to the numpy file containing the sampled feature indices for each layer. If empty string, assumes all features are used and that file names do not contain sample info.
            root_folder_name: Name of the folder where the weights are stored. Can be "virtual_weights", "twera", or "era"
        """
        if isinstance(config, str):
            config = load_config(config)
        self.config = config
        sample_indices_path = config.network_dir / "sampled_features.npy"
        self.root_folder = config.network_dir / root_folder_name # 
    
        try:
            self.sample_indices = np.load(str(sample_indices_path))
            assert self.sample_indices is not None, "sample_indices is None. Check if the file exists and is a valid numpy file."
        except Exception as e:
            raise ValueError(f"Error loading sample indices from {sample_indices_path}, {e}")

        
    def convert_original_to_sample_index(self, original_index, layer):
        sampled_layer_indices = self.sample_indices[layer]
        sample_index = np.where(sampled_layer_indices == original_index)[0]
        if len(sample_index) == 0:
            raise ValueError(f"Original index {original_index} not found in sampled indices for layer {layer}.")
        return int(sample_index[0])  # Return the index in the sampled features
    
    def convert_sample_to_original_index(self, sample_index, layer):
        sampled_layer_indices = self.sample_indices[layer]
        if sample_index < 0 or sample_index >= len(sampled_layer_indices):
            raise ValueError(f"Sample index {sample_index} is out of bounds for layer {layer} with {len(sampled_layer_indices)} sampled features.")
        return int(sampled_layer_indices[sample_index])  # Return the original feature index corresponding to the sample index
    
    def convert_neighbor_results_to_original_indices(self, neighbor_results):
        converted_results = []
        for layer, feature_idx_in_sampled, weight_value in neighbor_results:
            original_feature_idx = self.sample_indices[layer][feature_idx_in_sampled]
            converted_results.append((layer, int(original_feature_idx), weight_value))
        return converted_results
    
    def get_k_downstream_neighbors(self, layer, feature_idx, k=100, method="top", max_layer=None, index_in_sampled = True):
        """Get the k target features most or least influenced by a specific source feature.
        
        Args:
            layer: The source layer of the feature of interest.
            feature_idx: The index of the source feature in the source layer. (This index could be either the original feature index or the index in the sampled features, depending on the value of index_in_sampled.)
            k: The number of top or bottom neighbors to return.
            method: "top" for most positively influenced, "bottom" for most negatively influenced, "abs_bottom" for most influenced regardless of sign.
            max_layer: The maximum layer index to consider as a target (exclusive).
            index_in_sampled: Whether the feature_idx indicates the sample feature index or not
        Returns:
            A list of tuples (target_layer, target_feature_idx, weight_value) for the top k neighbors, where target_feature_idx is the index in either the sampled features or the original features depending on the value of index_in_sampled.
        """
        if max_layer is None:
            max_layer = self.config.n_layers - 1
        assert method in ["top", "bottom", "abs_bottom"], "Method must be 'top', 'bottom', or 'abs_bottom'"

        if not index_in_sampled:
            feature_idx = self.convert_original_to_sample_index(feature_idx, layer) # Convert original feature index to sampled feature index

        slices = []
        target_layers = list(range(layer + 1, max_layer + 1))

        for target_layer in target_layers:
            file_path = str(self.root_folder / f"{layer}_{target_layer}.safetensors")
            tensor_name = f"{layer}_{target_layer}"
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing weight file: {file_path}")
            
            with safe_open(file_path, framework="pt", device="cpu") as f:
                feature_vector = f.get_slice(tensor_name)[feature_idx, :]
                slices.append(feature_vector)

        feature_slice = torch.stack(slices, dim=0)
        flattened_slice = feature_slice.flatten()

        # Create a randomly shuffled version to use for tiebreaking in topk
        rand_idx = torch.randperm(flattened_slice.shape[0])
        shuffled_slice = flattened_slice[rand_idx]
        k = min(k, shuffled_slice.shape[0])  # In case there are fewer than k features total
        if method == "top":
            topk_values, topk_shuffled_indices = torch.topk(shuffled_slice, k=k)
            nonzero_mask = topk_values > 0
            topk_values = topk_values[nonzero_mask]
            topk_shuffled_indices = topk_shuffled_indices[nonzero_mask]
        elif method == "bottom":
            topk_values, topk_shuffled_indices = torch.topk(-shuffled_slice, k=k)
            topk_values = -topk_values
        else:  # abs_bottom
            abs_values = torch.abs(shuffled_slice)
            _, topk_shuffled_indices = torch.topk(-abs_values, k=k)
            topk_values = shuffled_slice[topk_shuffled_indices]

        # Map the shuffled indices back to the original indices
        topk_indices = rand_idx[topk_shuffled_indices]

        layer_offsets = topk_indices // feature_slice.shape[1]
        actual_target_layers = [target_layers[i] for i in layer_offsets.tolist()]
        feature_indices = (topk_indices % feature_slice.shape[1]).tolist()

        result = list(zip(actual_target_layers, feature_indices, topk_values.tolist()))
        if not index_in_sampled:
            result = self.convert_neighbor_results_to_original_indices(result)
        
        return result
        
    def get_k_upstream_neighbors(self, layer, feature_idx, k=100, method="top", min_layer=0, index_in_sampled = True):
        """Get the k source features most or least influencing a specific target feature.
        
        Args:        
            layer: The target layer of the feature of interest.
            feature_idx: The index of the target feature in the target layer. (This index could be either the original feature index or the index in the sampled features, depending on the value of index_in_sampled.)
            k: The number of top or bottom neighbors to return.
            method: "top" for most positively influencing, "bottom" for most negatively influencing, "abs_bottom" for most influencing regardless of sign.
            min_layer: The minimum layer index to consider as a source (inclusive).
            index_in_sampled: Whether the feature_idx indicates the sample feature index or not
        Returns:
            A list of tuples (source_layer, source_feature_idx, weight_value) for the top k neighbors, where source_feature_idx is the index in either the sampled features or the original features depending on the value of index_in_sampled.
        """
        assert method in ["top", "bottom", "abs_bottom"], "Method must be 'top', 'bottom', or 'abs_bottom'"
        if index_in_sampled and self.sample_indices is None:
            raise ValueError("sample_indices must be loaded to use sampled feature indices.")

        if not index_in_sampled:
            feature_idx = self.convert_original_to_sample_index(feature_idx, layer) # Convert original feature index to sampled feature index

        slices = []
        source_layers = list(range(min_layer, layer))

        for source_layer in source_layers:
            file_path = str(self.root_folder / f"{source_layer}_{layer}.safetensors")
            tensor_name = f"{source_layer}_{layer}"
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing weight file: {file_path}")
            
            with safe_open(file_path, framework="pt", device="cpu") as f:
                feature_vector = f.get_slice(tensor_name)[:, feature_idx]
                slices.append(feature_vector)

        feature_slice = torch.stack(slices, dim=0)
        flattened_slice = feature_slice.flatten()

        # Create a randomly shuffled version to use for tiebreaking in topk
        rand_idx = torch.randperm(flattened_slice.shape[0])
        shuffled_slice = flattened_slice[rand_idx]
        k = min(k, shuffled_slice.shape[0])  # In case there are fewer than k features total
        if method == "top":
            topk_values, topk_shuffled_indices = torch.topk(shuffled_slice, k=k)
            nonzero_mask = topk_values > 0
            topk_values = topk_values[nonzero_mask]
            topk_shuffled_indices = topk_shuffled_indices[nonzero_mask]
        elif method == "bottom":
            topk_values, topk_shuffled_indices = torch.topk(-shuffled_slice, k=k)
            topk_values = -topk_values
        else:  # abs_bottom
            abs_values = torch.abs(shuffled_slice)
            _, topk_shuffled_indices = torch.topk(-abs_values, k=k)
            topk_values = shuffled_slice[topk_shuffled_indices]

        # Map the shuffled indices back to the original indices
        topk_indices = rand_idx[topk_shuffled_indices]

        layer_offsets = topk_indices // feature_slice.shape[1]
        actual_source_layers = [source_layers[i] for i in layer_offsets.tolist()]
        feature_indices = (topk_indices % feature_slice.shape[1]).tolist()

        result = list(zip(actual_source_layers, feature_indices, topk_values.tolist()))
        if not index_in_sampled:
            result = self.convert_neighbor_results_to_original_indices(result)
        
        return result
    

    def _compare_to_zero(self, a, method):
        if method == "top":
            return a > 0
        elif method == "bottom":
            return a < 0
        else:  # abs_bottom
            return abs(a) > 0
        
    def get_num_nonzero_neighbors(self, downstream = True, method = "top"):
        num_nonzero_neighbors = np.zeros((self.config.n_layers, self.config.n_samples_per_layer))
        if downstream:
            for source_layer in range(self.config.n_layers - 1):
                print(f"Processing source layer {source_layer}", flush = True)
                for target_layer in range(source_layer + 1, self.config.n_layers):
                    file_path = str(self.root_folder / f"{source_layer}_{target_layer}.safetensors")
                    tensor_name = f"{source_layer}_{target_layer}"
                    
                    if target_layer == 13:
                        print(f"Checking file: {file_path}", flush = True)
                    if not os.path.exists(file_path):
                        raise FileNotFoundError(f"Missing weight file: {file_path}")
                    
                
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        weight_matrix = f.get_slice(tensor_name)[:,:]
                        num_nonzero_neighbors[source_layer, :] += torch.count_nonzero(self._compare_to_zero(weight_matrix, method), dim= 1).cpu().numpy()
        else:
            for target_layer in range(1, self.config.n_layers):
                print(f"Processing target layer {target_layer}", flush = True)
                for source_layer in range(target_layer):
                    
                    file_path = str(self.root_folder / f"{source_layer}_{target_layer}.safetensors")
                    tensor_name = f"{source_layer}_{target_layer}"
                    
                    if target_layer == 13:
                        print(f"Checking file: {file_path}", flush = True)
                    if not os.path.exists(file_path):
                        raise FileNotFoundError(f"Missing weight file: {file_path}")
                    
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        weight_matrix = f.get_slice(tensor_name)[:,:]
                        num_nonzero_neighbors[target_layer, :] += torch.count_nonzero(self._compare_to_zero(weight_matrix, method), dim= 0).cpu().numpy()

        return num_nonzero_neighbors


