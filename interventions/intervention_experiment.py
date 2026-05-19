from configs.config_data import NetworkConfig
from circuit_tracer import ReplacementModel
from pathlib import Path
import argparse



def main():
    parser = argparse.ArgumentParser(description="Compute coactivation stats for a specific layer.")
    parser.add_argument("--config", type=str, required = True, help="Name of config yaml file")
    args = parser.parse_args()
    clt_dir = Path(__file__).resolve().parent.parent
    config_path = clt_dir / "configs" / args.config
    config = NetworkConfig.from_yaml(config_path)
    config.validate_params()

    print(config)

    


if __name__ == "__main__":
    main()