from sample.sample_features import sample_pipeline
from configs.config_data import NetworkConfig
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Specify config file for feature network computation.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    parent_dir = Path(__file__).resolve().parent
    config_path = parent_dir / "configs" / args.config
    config = NetworkConfig.from_yaml(config_path)
    config.validate_params()
    sample_pipeline(config)
    config.lock_sample_params() # We have presumably finished sampling at this point, so we lock the sample parameters to keep a history of what parameters were used to generate the samples
    # weight_pipeline(config)
    #config.lock_weight_params() # We have finished computing the weights, so we lock the weight parameters to keep a history of what parameters were used to compute the weights

# from configs.config_data import NetworkConfig
# config_path = "/home/ammonbro/CLT/configs/config_template.yaml"
# config = NetworkConfig.from_yaml(config_path)


if __name__ == "__main__":
    main()
