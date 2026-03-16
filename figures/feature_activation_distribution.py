import torch
import numpy as np
from safetensors.torch import load_file
from matplotlib import pyplot as plt
import math
import os

# Find the current path of this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the current directory's parent path
parent_dir = os.path.dirname(current_dir)

tensors = load_file(os.path.join(parent_dir, "fineweb_feature_stats/feature_activation_counts.safetensors"))
counts = tensors["activation_counts"]
frequencies = counts.float() / 2039935
log_frequencies = torch.log10(frequencies + 1e-9)


num_layers = counts.shape[0]
d_sae = counts.shape[1]
cols = 5
rows = math.ceil(num_layers / cols)

# Make the figure large enough to read the individual plots
fig, axes = plt.subplots(rows, cols, figsize=(24, 4 * rows), sharex=True)
axes = axes.flatten() # Flatten to make indexing easier

print(f"Plotting distributions for {num_layers} layers...")

# 4. Iterate and plot each layer
for layer in range(num_layers):
    ax = axes[layer]
    layer_log_freqs = log_frequencies[layer].numpy()
    
    # Filter out the dead features (log freq ≈ -9.0)
    alive_log_freqs = layer_log_freqs[layer_log_freqs > -8.5]
    
    # Plot histogram
    ax.hist(alive_log_freqs, bins=50, color='skyblue', edgecolor='black')
    
    # Calculate and plot the mean
    if len(alive_log_freqs) > 0:
        mean_freq = np.mean(alive_log_freqs)
        ax.axvline(x=mean_freq, color='red', linestyle='dashed', linewidth=1.5)
    
    ax.set_title(f"Layer {layer}", fontsize=14)
    ax.grid(axis='y', alpha=0.5)
    
    # Only add x-labels to the bottom row to keep it clean
    if layer >= num_layers - cols:
        ax.set_xlabel("Log10(Frequency)", fontsize=12)
        
    # Only add y-labels to the left-most column
    if layer % cols == 0:
        ax.set_ylabel("Number of Features", fontsize=12)

# 5. Hide any extra subplots if num_layers doesn't perfectly fill the grid
for i in range(num_layers, len(axes)):
    fig.delaxes(axes[i])

plt.suptitle("Feature Frequency Distributions Across All Layers", fontsize=20, y=1.02)
plt.tight_layout()

plt.savefig(os.path.join(current_dir, "feature_activation_distribution.png"), dpi=300, bbox_inches='tight')
