import argparse
from network.neighbors import NetworkNeighbors
from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd

def make_neighbor_plots(network_neighbors, downstream_neighbor_counts, upstream_neighbor_counts):
    # Compute the neighbor counts
    network_dir = network_neighbors.config.network_dir
    
    # Plot the distribution of positive neighbors for each layer
    for layer in range(network_neighbors.config.n_layers):
        plot_positive_neighbor_distribution(network_dir, downstream_neighbor_counts, layer, downstream = True)

    upstream_neighbor_counts = network_neighbors.get_num_nonzero_neighbors(downstream = False, method = "top")
    for layer in range(network_neighbors.config.n_layers):
        plot_positive_neighbor_distribution(network_dir, upstream_neighbor_counts, layer, downstream = False)


def plot_positive_neighbor_distribution(network_dir, neighbor_counts, layer, downstream = True):
    # get the {layer} row from neighbor_counts and plot a histogram of the distribution of the values in that row
    downstream_str = "downstream" if downstream else "upstream"
    plt.figure(figsize=(10, 6))
    plt.hist(neighbor_counts[layer], bins=30, edgecolor='black')
    plt.title(f"distribution of {downstream_str} neighbor counts for layer {layer}")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    os.makedirs(f"{network_dir}/figures/{downstream_str}_positive_neighbor_distribution", exist_ok=True)
    plt.savefig(f"{network_dir}/figures/{downstream_str}_positive_neighbor_distribution/layer_{layer}.png")
    plt.close()


def calc_scorable_proportion(neighbor_counts, threshold = 45):
    """Calculates the proportion of features within each layer that have over a threshold number of positive neighbors."""
    return np.sum(neighbor_counts > threshold, axis=1) / neighbor_counts.shape[1]


def visualize_scorable_proportions(downstream_proportions, upstream_proportions, config, threshold):
    print("Downstream scorable proportions:", downstream_proportions)
    print("Upstream scorable proportions:", upstream_proportions)
    layers = np.arange(len(downstream_proportions))

    # Make a bar chart of the downstream proportions 
    plt.figure(figsize=(10, 6))
    plt.bar(layers - 0.2, downstream_proportions, width=0.4, label='Downstream', color='blue')
    plt.bar(layers + 0.2, upstream_proportions, width=0.4, label='Upstream', color='orange')
    plt.xlabel('Layer')
    plt.ylabel(f'Proportion of Features with > {threshold} Neighbors')
    plt.title(f'Proportion of Scorable Features by Layer (Threshold: {threshold})')
    plt.xticks(layers)
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    os.makedirs(f"{config.network_dir}/figures/scorable_proportions", exist_ok=True)
    plt.savefig(f"{config.network_dir}/figures/scorable_proportions/scorable_proportions_threshold_{threshold}.png")
    plt.close()


def calc_stats_for_network(network_neighbors, threshold = 45):
    downstream_neighbor_counts = network_neighbors.get_num_nonzero_neighbors(downstream = True, method = "top")
    upstream_neighbor_counts = network_neighbors.get_num_nonzero_neighbors(downstream = False, method = "top")
    downstream_proportions = calc_scorable_proportion(downstream_neighbor_counts, threshold = threshold)
    upstream_proportions = calc_scorable_proportion(upstream_neighbor_counts, threshold = threshold)
    median_downstream_neighbor_counts = np.median(downstream_neighbor_counts, axis=1)
    median_upstream_neighbor_counts = np.median(upstream_neighbor_counts, axis=1)
    q3_downstream_neighbor_counts = np.percentile(downstream_neighbor_counts, 75, axis=1)
    q3_upstream_neighbor_counts = np.percentile(upstream_neighbor_counts, 75, axis=1)
    q1_downstream_neighbor_counts = np.percentile(downstream_neighbor_counts, 25, axis=1)
    q1_upstream_neighbor_counts = np.percentile(upstream_neighbor_counts, 25, axis=1)


    visualize_scorable_proportions(downstream_proportions, upstream_proportions, network_neighbors.config, threshold = threshold)
    make_neighbor_plots(network_neighbors, downstream_neighbor_counts, upstream_neighbor_counts)

    df = pd.DataFrame({
        'Layer': np.arange(network_neighbors.config.n_layers),
        'Downstream Scorable Proportion': downstream_proportions,
        'Upstream Scorable Proportion': upstream_proportions,
        'Median Downstream Neighbors': median_downstream_neighbor_counts,
        'Median Upstream Neighbors': median_upstream_neighbor_counts,
        'Q1 Downstream Neighbors': q1_downstream_neighbor_counts,
        'Q1 Upstream Neighbors': q1_upstream_neighbor_counts,
        'Q3 Downstream Neighbors': q3_downstream_neighbor_counts,
        'Q3 Upstream Neighbors': q3_upstream_neighbor_counts
    })
    df.to_csv(f"{network_neighbors.config.network_dir}/neighbor_stats.csv", index=False)
    return downstream_neighbor_counts, upstream_neighbor_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute network stats for a given config")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file (relative from the configs directory)")
    args = parser.parse_args()
    neighbors = NetworkNeighbors(args.config, root_folder_name="twera")
    calc_stats_for_network(neighbors, threshold = 45)



