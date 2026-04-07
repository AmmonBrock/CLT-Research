import os
import subprocess
import math
from pathlib import Path
from configs.config_data import NetworkConfig
import argparse

def load_config(config_name):
    clt_dir = Path(__file__).resolve().parent
    config_path = clt_dir / "configs" / config_name
    return NetworkConfig.from_yaml(config_path)

def format_slurm_time(minutes):
    minutes = int(math.ceil(minutes))
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"

import math
import glob
import os
from pathlib import Path

def estimate_virtual_weights_minutes(config):
    n_layers = config.n_layers
    n_samples = config.n_samples_per_layer
    layers_to_analyze = range(n_layers)
    
    total_pairs = 0
    max_possible_pairs = (n_layers * (n_layers - 1)) / 2

    for source_layer in layers_to_analyze:
        for target_layer in layers_to_analyze:
            if source_layer >= target_layer:
                continue
            if not (config.virtual_weight_dir / f"{source_layer}_{target_layer}.safetensors").exists():
                total_pairs += 1

    if total_pairs == 0:
        print("All virtual weights already computed, skipping Step 2.")
        return None

    base_mins = 10.0
    full_run_compute_mins = 1.2 * ((n_samples / 1000.0) ** 2)
    fraction_remaining = total_pairs / max(1, max_possible_pairs)
    adjusted_compute_mins = full_run_compute_mins * fraction_remaining
    total_mins = (base_mins + adjusted_compute_mins) * 1.2
    safe_mins = max(25, int(math.ceil(total_mins)))


    return safe_mins

def estimate_coactivations_time(n_tokens_coacts: int, n_samples_per_layer: int, n_layers: int) -> int:
    """
    Estimates the SLURM time required (in minutes) for the coactivations script.
    Calibrated to 19,000 tokens/sec with a tight 10% safety buffer.
    """

    # Fixed Setup & Saving Overhead
    fixed_overhead_mins = 10.0
    
    # 2. Forward Pass Time (Linear with tokens)
    # Calibrated to 19,000 tokens/sec based on the 145-minute benchmark
    tokens_per_second = 19_000.0
    forward_pass_mins = (n_tokens_coacts / tokens_per_second) / 60.0
    
    # 3. Einsum Math Time (Quadratic with samples, worst-case Layer 0)
    # FLOPs = 2 * 25 layers * tokens * S^2. 
    # H200 FP32 Effective Throughput = ~36e12 operations per second
    target_layers = max(1., float(n_layers - 1))
    effective_flops_per_sec = 36e12
    total_flops = 2.0 * target_layers * n_tokens_coacts * (n_samples_per_layer ** 2)
    einsum_math_mins = (total_flops / effective_flops_per_sec) / 60.0
    
    # Total raw calculation
    raw_total_mins = fixed_overhead_mins + forward_pass_mins + einsum_math_mins
    
    # Add a 25% safety buffer 
    safe_total_mins = raw_total_mins * 1.25
    
    # Return at least 30 minutes, rounded up
    return max(30, int(math.ceil(safe_total_mins)))

def estimate_global_time_and_memory(config: NetworkConfig):
    """
    Provides conservative time and memory estimates for a SLURM job submission.
    """

    n_layers = config.n_layers
    n_samples_per_layer = config.n_samples_per_layer
    to_compute = config.to_compute
    time_safety_multiplier = 1.5
    mem_safety_multiplier = 2.0
    total_pairs = (n_layers * (n_layers - 1)) / 2
    
    if total_pairs == 0:
        return {"time_minutes": 0, "mem_gb": 0, "slurm_time": "00:00:00"}

    # --- 1. MEMORY ESTIMATION ---
    # Peak memory happens during: TWERA = (E_ab.float() / E_target) * V.float()
    # At this moment, RAM holds: 
    # E_ab (fp16), V (fp16), E_ab_fp32, coact_ratio_fp32, V_fp32, TWERA_fp32
    # Total byte multiplier per element is roughly 2 + 2 + 4 + 4 + 4 + 4 = 20 bytes
    
    elements_per_matrix = n_samples_per_layer ** 2
    peak_tensor_bytes = elements_per_matrix * 20
    peak_tensor_mb = peak_tensor_bytes / (1024 ** 2)
    
    # Add 2048 MB (2 GB) for baseline Python, PyTorch context, and safetensors overhead
    base_ram_mb = peak_tensor_mb + 2048
    
    # Apply safety multiplier and convert to GB (rounded up to nearest whole GB)
    safe_mem_gb = math.ceil((base_ram_mb * mem_safety_multiplier) / 1024)

    # --- 2. TIME ESTIMATION ---
    # File sizes in MB (saved as float16 -> 2 bytes per element)
    matrix_size_mb = (elements_per_matrix * 2) / (1024 ** 2)

    # Conservative network drive speeds (100 MB/s read, 50 MB/s write)
    conservative_read_mb_s = 100.0
    conservative_write_mb_s = 50.0
    compute_overhead_s = 0.5 

    total_time_seconds = 0.0


    if "twera" in to_compute:
        read_time_per_pair = (matrix_size_mb * 2) / conservative_read_mb_s
        write_time_per_pair = matrix_size_mb / conservative_write_mb_s
        time_per_pair = read_time_per_pair + write_time_per_pair + compute_overhead_s
        total_time_seconds += time_per_pair * total_pairs

    if "era" in to_compute:
        read_time_per_pair = (matrix_size_mb * 2) / conservative_read_mb_s
        write_time_per_pair = matrix_size_mb / conservative_write_mb_s
        time_per_pair = read_time_per_pair + write_time_per_pair + compute_overhead_s
        total_time_seconds += time_per_pair * total_pairs

    # Apply safety multiplier
    padded_time_seconds = total_time_seconds * time_safety_multiplier
    padded_time_seconds = max(padded_time_seconds, 600)
    
    # Format for SLURM (HH:MM:SS)
    hours = int(padded_time_seconds // 3600)
    minutes = int((padded_time_seconds % 3600) // 60)
    seconds = int(padded_time_seconds % 60)
    slurm_time_format = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    
    return slurm_time_format, safe_mem_gb

def submit_job(script_content, dependencies=None):
    """Submits a script via sbatch and returns the Job ID."""
    cmd = ["sbatch", "--parsable"]
    if dependencies:
        cmd.append(f"--dependency={dependencies}")
        
    result = subprocess.run(cmd, input=script_content, text=True, capture_output=True)
    if result.returncode == 0:
        job_id = result.stdout.strip().split(';')[0] # --parsable returns just the job ID
        return job_id
    else:
        print(f"Failed to submit job. Error: {result.stderr}")
        exit(1)

def global_weights_script(config, config_path):
    log_dir_name = config_path.split('.')[0]
    slurm_time, mem_gb = estimate_global_time_and_memory(config)
    script = f"""#!/bin/bash
#SBATCH --job-name=step4
#SBATCH --time={slurm_time}
#SBATCH --mem={mem_gb}G
#SBATCH --output=logs/{log_dir_name}/global_weights_%j.out
uv run -m network.global_weights --config {config_path}
"""
    return script


def sample_script(config, config_path):
    mins_1 = 15
    log_dir_name = config_path.split('.')[0]
    if config.compute_activations:
        mins_1 += (config.n_tokens_act_freq / 6000) / 60
    
    
        step1 = f"""#!/bin/bash
#SBATCH --job-name=step1
#SBATCH --time={format_slurm_time(mins_1)}
#SBATCH --gpus=h200:1
#SBATCH --mem=64G
#SBATCH --output=logs/{log_dir_name}/sampling_%j.out
uv run -m sample.sample_features --config {config_path}
"""
    else:
        step1 = f"""#!/bin/bash
#SBATCH --job-name=step1
#SBATCH --time={format_slurm_time(15)}
#SBATCH --mem=8G
#SBATCH --output=logs/{log_dir_name}/sampling_%j.out
uv run -m sample.sample_features --config {config_path}
"""
    return step1

def virtual_weight_script(config, config_path):
    log_dir_name = config_path.split('.')[0]
    mins = estimate_virtual_weights_minutes(config)
    if mins is None:
        return None
    step2 = f"""#!/bin/bash
#SBATCH --job-name=step2
#SBATCH --time={format_slurm_time(mins)}
#SBATCH --gpus=h200:1
#SBATCH --mem=32G
#SBATCH --output=logs/{log_dir_name}/virtual_weights_%j.out
uv run -m network.virtual_weights --config {config_path}
"""
    return step2

def coactivation_script(config, config_path):
    # Find layers that still need calculations
    layers_to_analyze = []
    for source_layer in range(config.n_layers):
        if not (config.network_dir / "coactivations" / f"coactivation_stats_layer_{source_layer}.safetensors").exists():
            layers_to_analyze.append(source_layer)
    if layers_to_analyze == []:
        print("All coactivations already computed, skipping Step 3.")
        return None, None
    array_string = ",".join(map(str, layers_to_analyze))
    
    tokens = config.n_tokens_coacts
    samples = config.n_samples_per_layer
    n_layers = config.n_layers
    base_mins = estimate_coactivations_time(tokens, samples, n_layers)
    log_dir_name = config_path.split('.')[0]
    


    # The --exclude command is specific to my compute setup
    step3 = f"""#!/bin/bash
#SBATCH --job-name=step3
#SBATCH --time={format_slurm_time(base_mins)}
#SBATCH --exclude=m13h-2-1,m13h-1-2
#SBATCH --gpus=h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array={array_string}%1
#SBATCH --output=logs/{log_dir_name}/coactivations_%A_%a.out
uv run -m activations.coactivation --config {config_path} --source_layer $SLURM_ARRAY_TASK_ID --mins {int(base_mins)}
"""
    return step3, base_mins

def network_stats_script(config_path):
    log_dir_name = config_path.split('.')[0]
    step5 = f"""#!/bin/bash
#SBATCH --job-name=step5
#SBATCH --time={format_slurm_time(15)}
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/{log_dir_name}/network_stats_%j.out
uv run -m network.compute_network_stats --config {config_path}
"""
    return step5

def compute_network_pipeline(config_path):
    os.makedirs(f"logs/{config_path.split('.')[0]}", exist_ok=True)
    config = load_config(config_path)
    to_compute = config.to_compute
    
    # 1. Sampling Pipeline
    jid1 = None
    if not (config.network_dir / "sampled_features.npy").exists():
        step1 = sample_script(config, config_path)
        jid1 = submit_job(step1)
        print(f"Submitted Step 1 (JID: {jid1})")


    # 2. Virtual Weights (depends on Step 1)
    jid2 = None
    if "virtual" in to_compute:
        step2 = virtual_weight_script(config, config_path)

        if step2 is not None:
            jid2 = submit_job(step2, f"afterok:{jid1}" if jid1 is not None else None)
            print(f"Submitted Step 2 (JID: {jid2}) hanging on Step 1")
        else:
            print("No virtual weights to compute, skipping Step 2.")


    # 3. Submit Coactivations Array (depends on Step 1)
    jid3 = None
    if "coactivations" in to_compute:
        step3, base_mins = coactivation_script(config, config_path)
        if step3 is not None:
            jid3 = submit_job(step3, f"afterok:{jid1}" if jid1 is not None else None)
            print(f"Submitted Coactivations Array (JID: {jid3}) hanging on Step 1")
        else:
            print("No coactivations to compute, skipping Step 3.")
    
        
    # 4. Submit global weights job (dependent on Step 2 and Step 3)
    jid4 = None
    if "twera" in to_compute or "era" in to_compute:
        step4 = global_weights_script(config, config_path)
        if jid2 is not None:
            if jid3 is not None:
                dependencies = f"afterok:{jid2}:{jid3}"
            else:
                dependencies = f"afterok:{jid2}"
        else:
            if jid3 is not None:
                dependencies = f"afterok:{jid3}"
            else:
                dependencies = None
        jid4 = submit_job(step4, dependencies)
        print(f"Submitted Step 4 (JID: {jid4}) hanging on Step 2 and Step 3")
    else:
        print("No global weights to compute")

    # 5. Submit network stats job (dependent on Step 4)
    if "twera" in to_compute:
        step5 = network_stats_script(config_path)
        dependencies = f"afterok:{jid4}"
        submit_job(step5, dependencies) # Wait for the new twera weights to calculate the stats
    elif (config.network_dir / "twera").exists() and not (config.network_dir / "neighbor_stats.csv").exists():
        step5 = network_stats_script(config_path)
        submit_job(step5) # Submit the job right away because twera already exists and we're not overwriting anything
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the CLT pipeline on SLURM.")
    parser.add_argument("--config", type=str, required=True, help="Name of the config YAML file in the configs/ directory.")
    args = parser.parse_args()
    config_path = args.config
    compute_network_pipeline(config_path)

