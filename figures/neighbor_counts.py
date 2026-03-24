import plotly.graph_objects as go
import torch
from safetensors import safe_open
import os
import numpy as np

twera_dir = "twera_small_sample_150M"
prefix = "twera_"
suffix = ".safetensors"
tensor_prefix = "TWERA_"

def get_downstream_neighbor_counts(weight_dir, prefix, suffix, tensor_prefix, layer, count_threshold = 0):
    neighbor_counts= torch.zeros(1000, dtype = torch.int)
    for target_layer in range(layer + 1, 26):
        with safe_open(os.path.join(weight_dir, f"{prefix}{layer}_{target_layer}{suffix}"), framework="pt") as f:
            feature_matrix = f.get_tensor(f"{tensor_prefix}{layer}_{target_layer}")
        nonzero = feature_matrix > 0
        neighbor_counts += nonzero.sum(dim=1)
    print(f"Num features with <= {count_threshold} neighbors:", (neighbor_counts <= count_threshold).sum().item())
    return neighbor_counts

def get_upstream_neighbor_counts(weight_dir, prefix, suffix, tensor_prefix, layer, count_threshold = 0):
    neighbor_counts= torch.zeros(1000, dtype = torch.int)
    for source_layer in range(layer):
        with safe_open(os.path.join(weight_dir, f"{prefix}{source_layer}_{layer}{suffix}"), framework="pt") as f:
            feature_matrix = f.get_tensor(f"{tensor_prefix}{source_layer}_{layer}")
        nonzero = feature_matrix > 0
        neighbor_counts += nonzero.sum(dim=0)
    print(f"Num features with <= {count_threshold} neighbors:", (neighbor_counts <= count_threshold).sum().item())
    return neighbor_counts

downstream_neighbor_counts = torch.zeros((0, 1000), dtype = torch.int)
upstream_neighbor_counts = torch.zeros((0, 1000), dtype = torch.int)
for layer in range(26):
    layer_neighbor_counts = get_downstream_neighbor_counts(twera_dir, prefix, suffix, tensor_prefix, layer, count_threshold = 0)
    layer_upstream_neighbor_counts = get_upstream_neighbor_counts(twera_dir, prefix, suffix, tensor_prefix, layer, count_threshold = 0)

    downstream_neighbor_counts = torch.cat((downstream_neighbor_counts, layer_neighbor_counts.unsqueeze(0)))
    upstream_neighbor_counts = torch.cat((upstream_neighbor_counts, layer_upstream_neighbor_counts.unsqueeze(0)))
proportion_less_than_45_downstream_neighbors = (downstream_neighbor_counts < 45).sum(dim=1)/1000
proportion_less_than_45_upstream_neighbors = (upstream_neighbor_counts < 45).sum(dim=1)/1000


#visualize
from matplotlib import pyplot as plt
plt.bar(range(26), proportion_less_than_45_downstream_neighbors, alpha = 0.5)
plt.xlabel("Layer")
plt.ylabel("Proportion")
plt.title("Proportion of features with fewer than 45 downstream neighbors")
plt.savefig("proportion_less_than_45_downstream_neighbors.png")
plt.clf()

plt.bar(range(26), proportion_less_than_45_upstream_neighbors, alpha = 0.5)
plt.xlabel("Layer")
plt.ylabel("Proportion")
plt.title("Proportion of features with fewer than 45 upstream neighbors")
plt.savefig("proportion_less_than_45_upstream_neighbors.png")
plt.clf()

def plot_interactive_neighbor_cutoffs(down_stream_neighbor_counts, max_cutoff=100, step=5, downstream = True):
    """
    Creates an interactive Plotly bar chart with a slider to adjust the downstream neighbor cutoff.
    
    Args:
        down_stream_neighbor_counts: torch.Tensor of shape [26, 1000]
        max_cutoff: The maximum cutoff value for the slider
        step: The increment step for the slider cutoffs
    """

    # 1. Safely convert the PyTorch tensor to a NumPy array
    # .cpu() ensures it works even if your tensor is currently on the GPU
    counts_np = down_stream_neighbor_counts.cpu().numpy()
    
    num_layers = counts_np.shape[0]
    layers = np.arange(num_layers)
    neighbor_type = "downstream" if downstream else "upstream"
    
    # Define the cutoffs for the slider (e.g., 1, 6, 11... up to max_cutoff)
    cutoffs = list(range(1, max_cutoff + 1, step))
    
    fig = go.Figure()

    # 2. Add a trace for every cutoff value
    for i, cutoff in enumerate(cutoffs):
        # Calculate the proportion of features strictly less than the cutoff
        # counts_np < cutoff creates a boolean array, np.mean() computes the proportion of Trues
        proportions = np.mean(counts_np < cutoff, axis=1)
        
        fig.add_trace(
            go.Bar(
                x=layers,
                y=proportions,
                visible=(i == 0), # Only the very first trace is visible by default
                name=f"<{cutoff} neighbors",
                marker_color='rgba(130, 170, 210, 0.8)' # Matches your original matplotlib styling
            )
        )

    # 3. Create the Slider Logic
    steps = []
    for i, cutoff in enumerate(cutoffs):
        # Create a boolean array where ONLY the current trace's index is True
        step_visibility = [False] * len(cutoffs)
        step_visibility[i] = True
        
        step = dict(
            method="update",
            args=[
                # Update the visibility of the traces
                {"visible": step_visibility},
                # Update the title to reflect the current cutoff
                {"title.text": f"Proportion of features with fewer than {cutoff} {neighbor_type} neighbors"}
            ],
            label=str(cutoff)
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Cutoff: "},
        pad={"t": 50},
        steps=steps
    )]

    # 4. Format the layout and lock the Y-axis
    fig.update_layout(
        sliders=sliders,
        title=f"Proportion of features with fewer than {cutoffs[0]} {neighbor_type} neighbors",
        xaxis_title="Layer",
        yaxis_title="Proportion",
        yaxis=dict(range=[0, 1.05]), # Lock y-axis so the bars don't jump around when sliding
        template="simple_white",
        height=600,
        width=800
    )

    fig.write_html(f"figures/interactive_{neighbor_type}_neighbor_cutoffs.html")
    fig.show()

plot_interactive_neighbor_cutoffs(downstream_neighbor_counts, max_cutoff=100, step=5, downstream = True)
plot_interactive_neighbor_cutoffs(upstream_neighbor_counts, max_cutoff=100, step=5, downstream = False)