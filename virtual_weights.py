import torch
from safetensors import safe_open
from safetensors.torch import save_file
import os
import numpy as np
import pandas as pd
import json
from tabulate import tabulate


def compute_virtual_weights(device = 'cuda',
                            save_dir = "./virtual_weights",
                            clt_dir = "/home/ammonbro/CLT/models/round2/models--mntss--clt-gemma-2-2b-2.5M/snapshots/637a2f8b950fca8623e57658721484c28c166ca5",
                            sample_indices_path = "" # shape (num_layers, num_samples) of indices to sample from each layer. If None, uses all features.
                            ):

    layers_to_analyze = range(26) # Changing this parameter is not built to be flexible yet
    torch.set_grad_enabled(False)
    os.makedirs(save_dir, exist_ok=True)

    sample_indices = torch.from_numpy(np.load(sample_indices_path)) if sample_indices_path != "" else None

    # For each pair of layers, compute all feature-to-feature virtual weights
    for source_layer in layers_to_analyze:
        targets_left = [t for t in layers_to_analyze if t > source_layer]
        if all(os.path.exists(os.path.join(save_dir, f"weight_{source_layer}_{t}_{sample_indices_path.split('/')[-1].split('.')[0]}.safetensors")) for t in targets_left):
            print(f"All weights for source layer {source_layer} already exist. Skipping.")
            continue

        # Get the decoder from this source layer to all future layers
        dec_path = os.path.join(clt_dir, f"W_dec_{source_layer}.safetensors")
        if not os.path.exists(dec_path):
            print(f"Decoder file for source layer {source_layer} not found at {dec_path}. Skipping.")
            continue
        with safe_open(dec_path, framework="pt", device = device) as f:
            tensor_name = f.keys()[0]
            w_dec_source = f.get_tensor(tensor_name).bfloat16()  # (d_sae, num_layers - source_layer, d_model)
        assert w_dec_source.shape[1] == (len(layers_to_analyze) - source_layer), f"Expected decoder to have shape (d_sae, {len(layers_to_analyze) - source_layer}, d_model), but got {w_dec_source.shape}"


        if sample_indices is not None:
            # Only keep the decoders corresponding to the sampled features for this source layer
            w_dec_source = w_dec_source[sample_indices[source_layer], :, :]

        
        for target_layer in layers_to_analyze:
            if source_layer >= target_layer:
                continue
            if os.path.exists(os.path.join(save_dir, f"weight_{source_layer}_{target_layer}_{sample_indices_path.split('/')[-1].split('.')[0]}.safetensors")):
                print(f"Virtual weight from layer {source_layer} to {target_layer} already exists. Skipping.")
                continue
            
            
            print(f"Computing virtual weights: Layer {source_layer} → Layer {target_layer}", flush = True)
            save_path = os.path.join(save_dir, f"weight_{source_layer}_{target_layer}_{sample_indices_path.split('/')[-1].split('.')[0]}.safetensors")
            enc_path = os.path.join(clt_dir, f"W_enc_{target_layer}.safetensors")

            with safe_open(enc_path, framework="pt", device = device) as f:
                tensor_name = f.keys()[0]
                target_encoders = f.get_tensor(tensor_name).bfloat16()  # (d_sae, d_model)

            if sample_indices is not None:
                target_encoders = target_encoders[sample_indices[target_layer], :]
            
            # Sum decoders from source layer across intermediate layers
            # Layers in between: [source_layer, source_layer+1, ..., target_layer-1]
            summed_decoders = w_dec_source[:, :(target_layer - source_layer), :].sum(dim = 1) # (d_sae_source, d_model)
            
            
            # Virtual weight matrix: (d_sae_source, d_sae_target)
            # Each source feature to each target feature
            V = torch.einsum("sm,tm->st", summed_decoders, target_encoders)  # (d_sae_source, d_model) @ (d_model, d_sae_target) = (d_sae_source, d_sae_target)

            
            print(f"  Shape: {V.shape}, Mean abs value: {V.abs().mean():.4f}")
            print(f"Saving virtual weights from source layer {source_layer} to target layer {target_layer}...")
            save_file({f"weight_{source_layer}_{target_layer}": V.cpu()}, save_path)

            del target_encoders
            del summed_decoders
            del V


        del w_dec_source
        torch.cuda.empty_cache()




    print("\nDone! Virtual weights savaed to folder (Indexed by source. Source = row, target = column).")

def verify_decoder_shapes(device = "cuda", layers_to_analyze=range(26), sample_indices_path = "feature_filtering/sampled_features_small.npy", clt_dir="/home/ammonbro/CLT/models/round2/models--mntss--clt-gemma-2-2b-2.5M/snapshots/637a2f8b950fca8623e57658721484c28c166ca5"):
    
    torch.set_grad_enabled(False)

    sample_indices = torch.from_numpy(np.load(sample_indices_path)) if sample_indices_path != "" else None

    # For each pair of layers, compute all feature-to-feature virtual weights
    for source_layer in layers_to_analyze:
        targets_left = [t for t in layers_to_analyze if t > source_layer]

        # Get the decoder from this source layer to all future layers
        dec_path = os.path.join(clt_dir, f"W_dec_{source_layer}.safetensors")
        if not os.path.exists(dec_path):
            print(f"Decoder file for source layer {source_layer} not found at {dec_path}. Skipping.")
            continue
        with safe_open(dec_path, framework="pt", device = device) as f:
            tensor_name = f.keys()[0]
            w_dec_source = f.get_tensor(tensor_name).bfloat16()  # (d_sae, num_layers, d_model)

        print(w_dec_source.shape)


        del w_dec_source
        torch.cuda.empty_cache()




    print("\nDone! Virtual weights savaed to folder (Indexed by source. Source = row, target = column).")
class VirtualWeightNeighbors():
    def __init__(self, save_dir = "./virtual_weights", sample_indices_path = "feature_filtering/sampled_features_small.npy", prefix = "weight_", suffix = None, tensor_prefix = "weight_"):
        """Class to load and query weights between features in different layers. 
        
        Args:
            save_dir: Directory where weight files are saved. Each file should be named at least contain {source_layer}_{target_layer} within the file name
            sample_indices_path: Path to the numpy file containing the sampled feature indices for each layer. If empty string, assumes all features are used and that file names do not contain sample info.
            prefix: Optional prefix for the weight file names. Used to adapt this class to twera weights  etc.
            suffix: Optional suffix for the weight file names. Used to adapt this class to twera weights  etc.
        """
        
        self.save_dir = save_dir
        self.sample_indices_path = sample_indices_path
        self.sample_indices = np.load(sample_indices_path) if sample_indices_path != "" else None

        self.virtual_weights_prefix = prefix
        if suffix is None:
            self.virtual_weights_suffix = "_" + sample_indices_path.split('/')[-1].split('.')[0] + ".safetensors" if sample_indices_path != "" else ".safetensors"
        else:
            self.virtual_weights_suffix = suffix
        self.tensor_prefix = tensor_prefix

        
    def convert_original_to_sample_index(self, original_index, layer):
        if self.sample_indices is None:
            raise ValueError("sample_indices must be loaded to convert original indices.")
        sampled_layer_indices = self.sample_indices[layer]
        sample_index = np.where(sampled_layer_indices == original_index)[0]
        if len(sample_index) == 0:
            raise ValueError(f"Original index {original_index} not found in sampled indices for layer {layer}.")
        return int(sample_index[0])  # Return the index in the sampled features
    
    def convert_sample_to_original_index(self, sample_index, layer):
        if self.sample_indices is None:
            raise ValueError("sample_indices must be loaded to convert sample indices.")
        sampled_layer_indices = self.sample_indices[layer]
        if sample_index < 0 or sample_index >= len(sampled_layer_indices):
            raise ValueError(f"Sample index {sample_index} is out of bounds for layer {layer} with {len(sampled_layer_indices)} sampled features.")
        return int(sampled_layer_indices[sample_index])  # Return the original feature index corresponding to the sample index
    
    def convert_neighbor_results_to_original_indices(self, neighbor_results):
        if self.sample_indices is None:
            raise ValueError("sample_indices must be loaded to convert neighbor results to original indices.")
        converted_results = []
        for layer, feature_idx_in_sampled, weight_value in neighbor_results:
            original_feature_idx = self.sample_indices[layer][feature_idx_in_sampled]
            converted_results.append((layer, int(original_feature_idx), weight_value))
        return converted_results
    
    def get_k_downstream_neighbors(self, layer, feature_idx, k=100, method="top", max_layer=25, index_in_sampled = True):
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
        assert method in ["top", "bottom", "abs_bottom"], "Method must be 'top', 'bottom', or 'abs_bottom'"
        if index_in_sampled and self.sample_indices is None:
            raise ValueError("sample_indices must be loaded to use sampled feature indices.")

        if not index_in_sampled:
            feature_idx = self.convert_original_to_sample_index(feature_idx, layer) # Convert original feature index to sampled feature index

        slices = []
        target_layers = list(range(layer + 1, max_layer + 1))


        

        for target_layer in target_layers:
            file_path = os.path.join(self.save_dir, f"{self.virtual_weights_prefix}{layer}_{target_layer}{self.virtual_weights_suffix}")
            tensor_name = f"{self.tensor_prefix}{layer}_{target_layer}"
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing virtual weight file: {file_path}")
            
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
            file_path = os.path.join(self.save_dir, f"{self.virtual_weights_prefix}{source_layer}_{layer}{self.virtual_weights_suffix}")
            tensor_name = f"{self.tensor_prefix}{source_layer}_{layer}"
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing virtual weight file: {file_path}")
            
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
        
    def query_features(self, queries, index_map_path="/home/ammonbro/CLT/data/feature_labels/feature_index_map.json"):
        """
        Takes a list of (layer, feature_idx) tuples and returns a Pandas DataFrame 
        containing only the requested features.
        """
        # 1. Load the pre-computed index
        with open(index_map_path, 'r') as f:
            # JSON keys are always strings, so we convert them back to ints
            index_map = {int(k): v for k, v in json.load(f).items()}
            
        # 2. Map queries to files to minimize I/O operations
        # Format: { 'path/to/batch-0.jsonl.gz': [feature_idx1, feature_idx2] }
        file_to_queries = {}
        
        for layer, f_idx in queries:
            if layer not in index_map:
                print(f"Warning: Layer {layer} not found in index.")
                continue
                
            found_file = None
            for file_info in index_map[layer]:
                if file_info['min_idx'] <= f_idx <= file_info['max_idx']:
                    found_file = file_info['file_path']
                    break
                    
            if found_file:
                if found_file not in file_to_queries:
                    file_to_queries[found_file] = []
                file_to_queries[found_file].append(f_idx)
            else:
                print(f"Warning: feature {f_idx} not found in layer {layer}")

        # 3. Read the required files and extract the rows
        results = []
        columns_to_keep = ['index', 'layer', 'description', 'typeName', 'explanationModelName']
        
        for file_path, f_indices in file_to_queries.items():
            # Load the specific file
            df = pd.read_json(os.path.join("/home/ammonbro/CLT/data", file_path), lines=True, compression='gzip')
            
            # Filter down to just the requested feature indices
            filtered_df = df[df['index'].isin(f_indices)]
            
            # Filter down columns (handling missing columns gracefully)
            available_cols = [c for c in columns_to_keep if c in filtered_df.columns]
            results.append(filtered_df[available_cols])

        # 4. Combine and return
        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()
        
    def get_labels_for_neighbors(self, layer, feature_idx, neighbor_results, index_in_sampled = True, additional_label_info = [], show_source_feature = True, show_neighbors = True):
        """Get the feature labels for a list of neighbor results.
        
        Args:
            layer: The layer of the source feature.
            feature_idx: The index of the source feature in the source layer. (This index could be either the original feature index or the index in the sampled features, depending on the value of index_in_sampled.)
            neighbor_results: A list of tuples (layer, feature_idx, weight_value) where feature_idx is either the index in the sampled features or the original features depending on the value of index_in_sampled.
            index_in_sampled: Whether the feature_idx in neighbor_results indicates the sample feature index or not
            additional_label_info: A list of additional label info to include in the returned DataFrame (could be ['typeName', 'explanationModelName'])
            show_source_feature: Whether print the source feature information along with the neighbors
            output: Whether to print the results

        Returns:
            A pandas dataframe containing layer, feature_idx, weight_value, description, and any additional label info specified."""
        
        for c in additional_label_info:
            assert c in ['typeName', 'explanationModelName'], "Additional label info must be in ['typeName', 'explanationModelName']"
        
        index_col_name = "original_feature_idx" if not index_in_sampled else "sampled_feature_idx"
        result = pd.DataFrame(neighbor_results, columns = ["layer", index_col_name, "weight_value"])
        result[["layer", index_col_name]] = result[["layer", index_col_name]].astype(int)
        layers = result['layer']
        feature_indices = result[index_col_name]
        

        # Get the original feature indices
        original_source_feature_idx = self.convert_sample_to_original_index(feature_idx, layer) if index_in_sampled else feature_idx
        original_feature_indices = [self.convert_sample_to_original_index(idx, l) for idx, l in zip(feature_indices, layers)] if index_in_sampled else feature_indices
        if index_in_sampled:
            result['original_feature_idx'] = original_feature_indices

        # Prepare the queries for the feature labels
        queries = list(zip(layers, original_feature_indices))
        queries.append((layer, original_source_feature_idx))

        labels_df = self.query_features(queries)
        labels_df.rename(columns={'index': 'original_feature_idx'}, inplace=True)
        labels_df = labels_df[['original_feature_idx', 'layer', 'description'] + additional_label_info]
        labels_df['layer'] = labels_df['layer'].str.replace("-clt-hp", "").astype(int)


        source_feature_slice = labels_df.loc[labels_df['original_feature_idx'] == original_source_feature_idx]
        labels_df.drop(labels_df[labels_df['original_feature_idx'] == original_source_feature_idx].index, inplace=True)
        
        

        result = result.merge(labels_df, on=['layer','original_feature_idx'], how = 'left')

        # Reorder the columns so that any column that ends in _idx comes first and then the remaining layers
        idx_cols = [col for col in result.columns if col.endswith('_idx')]
        other_cols = [col for col in result.columns if not col.endswith('_idx')]
        result = result[idx_cols + other_cols]
        

        if show_source_feature:
            print(f"Info for feature (Layer {layer}, Feature {feature_idx}):")
            max_col_widths= [None if col != 'description' else 50 for col in source_feature_slice.columns]
            print(tabulate(source_feature_slice, showindex=False, headers="keys", tablefmt="grid",maxcolwidths = max_col_widths ))

        if show_neighbors:
            print(f"Neighbor info:")
            max_col_widths= [None if col != 'description' else 50 for col in result.columns]
            print(tabulate(result, showindex = False, headers="keys", tablefmt="grid",maxcolwidths = max_col_widths))

        return result, source_feature_slice
    
    def examine(self, layer, feature_idx, k=10, index_in_sampled = True, additional_label_info = []):
        if layer < 25:
            print("===== Downstream Neighbors =====")
            downstream_topk = self.get_k_downstream_neighbors(layer, feature_idx, k=k, method = "top", index_in_sampled = index_in_sampled)
            downstream_absbottomk = self.get_k_downstream_neighbors(layer, feature_idx, k=k, method = "abs_bottom", index_in_sampled = index_in_sampled)
            downstream_top_df, _ = self.get_labels_for_neighbors(layer, feature_idx, downstream_topk, index_in_sampled = index_in_sampled, additional_label_info = additional_label_info, show_source_feature = True, show_neighbors = True)
            print("Bottom neighbors")
            downstream_absbottom_df, _ = self.get_labels_for_neighbors(layer, feature_idx, downstream_absbottomk, index_in_sampled = index_in_sampled, additional_label_info = additional_label_info, show_source_feature = False, show_neighbors = True)

        if layer > 0:
            print("\n===== Upstream Neighbors =====")
            upstream_topk = self.get_k_upstream_neighbors(layer, feature_idx, k=k, method = "top", index_in_sampled = index_in_sampled)
            upstream_absbottomk = self.get_k_upstream_neighbors(layer, feature_idx, k=k, method = "abs_bottom", index_in_sampled = index_in_sampled)
            upstream_top_df, _ = self.get_labels_for_neighbors(layer, feature_idx, upstream_topk, index_in_sampled = index_in_sampled, additional_label_info = additional_label_info, show_source_feature = False, show_neighbors = True)
            print("Bottom neighbors")
            upstream_absbottom_df, _ = self.get_labels_for_neighbors(layer, feature_idx, upstream_absbottomk, index_in_sampled = index_in_sampled, additional_label_info = additional_label_info, show_source_feature = False, show_neighbors = True)



        






if __name__ == "__main__":
    compute_virtual_weights(sample_indices_path = "feature_filtering/sampled_features_small.npy")


    # layer = 12
    # feature_idx = 100
    # vwn = VirtualWeightNeighbors(save_dir = "twera_small_sample_12M", sample_indices_path = "feature_filtering/sampled_features_small.npy", prefix = "twera_", suffix = ".safetensors", tensor_prefix = "TWERA_")
    # vwn.examine(layer, feature_idx, k=10, index_in_sampled = True)