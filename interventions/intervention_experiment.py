import torch
from configs.config_data import NetworkConfig
from pathlib import Path
import argparse
import os
import glob
import pandas as pd

def state_hop(config):

    states = [
        "Alabama", "Alaska", "Arizona", "California", "Colorado", 
        "Connecticut", "Florida", "Georgia", "Illinois", 
        "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", 
        "Michigan", "Minnesota", "Mississippi", "Montana", 
        "Nebraska", "Nevada", "New Jersey", "New Mexico", "New York", 
        "North Carolina", "North Dakota", "Ohio", "Oregon", "Pennsylvania", 
        "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", 
        "Washington", "Wisconsin", "Wyoming"
    ]
    state_capitals = [
        "Montgomery", "Juneau", "Phoenix", "Sacramento", "Denver", 
        "Hartford", "Tallahassee", "Atlanta", "Springfield", 
        "Topeka", "Frankfort", "Baton Rouge", "Augusta", "Annapolis", 
        "Lansing", "St. Paul", "Jackson", "Helena", 
        "Lincoln", "Carson City", "Trenton", "Santa Fe", "Albany", 
        "Raleigh", "Bismarck", "Columbus", "Salem", "Harrisburg", 
        "Pierre", "Nashville", "Austin", "Salt Lake City", "Montpelier", 
        "Olympia", "Madison", "Cheyenne"
    ]
    state_cities = [
        "Birmingham", "Anchorage", "Tucson", "Fresno", "Boulder", 
        "Bridgeport", "Miami", "Savannah", "Chicago", 
        "Wichita", "Louisville", "Shreveport", "Bangor", "Baltimore", 
        "Detroit", "Minneapolis", "Biloxi", "Bozeman", 
        "Omaha", "Reno", "Princeton", "Albuquerque", "Buffalo", 
        "Charlotte", "Fargo", "Cleveland", "Eugene", "Philadelphia", 
        "Deadwood", "Memphis", "Dallas", "Provo", "Burlington", 
        "Seattle", "Milwaukee", "Laramie"
    ]
    df = pd.DataFrame({"state": state, "capital": state_capitals, "clue": state_cities})

    device = config.device
    cache_dir = config.model_storage_absolute
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    from circuit_tracer import ReplacementModel
    from transformers import AutoTokenizer

    snapshot_paths = glob.glob(f"{cache_dir}/models--{config.study_model_name.replace('/', '--')}/snapshots/*/")
    assert len(snapshot_paths) > 0, f"No snapshots found for model {config.study_model_name} in cache directory {cache_dir}. Please make sure the model is downloaded and the path is correct."
    absolute_model_path = snapshot_paths[0]
    tokenizer = AutoTokenizer.from_pretrained(absolute_model_path, local_files_only = True)
    model = ReplacementModel.from_pretrained(
        model_name = config.study_model_name,
        transcoder_set = config.feature_tool_name,
        dtype = torch.bfloat16,
        device = device,
        local_files_only = True,
        tokenizer = tokenizer)
    

    clt = model.transcoders
    hook_name_base = clt.feature_input_hook
    hook_names = [f"blocks.{layer}.{hook_name_base}" for layer in range(clt.n_layers)]

    


def main():
    parser = argparse.ArgumentParser(description="Compute coactivation stats for a specific layer.")
    parser.add_argument("--config", type=str, required = True, help="Name of config yaml file")
    args = parser.parse_args()
    clt_dir = Path(__file__).resolve().parent.parent
    config_path = clt_dir / "configs" / args.config
    config = NetworkConfig.from_yaml(config_path)
    config.validate_params()



    


if __name__ == "__main__":
    main()
