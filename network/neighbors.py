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
        self.root_folder = config.network_dir / root_folder_name 
        
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


