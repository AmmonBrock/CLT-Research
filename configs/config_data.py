import yaml
from pydantic import BaseModel, PositiveInt, NonNegativeInt, ConfigDict, model_validator
from pathlib import Path
import os
import json

class NetworkConfig(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    n_samples_per_layer: PositiveInt
    sample_method: str
    network_name: str
    compute_activations: bool

    activation_text_corpus: str
    n_tokens_act_freq: PositiveInt
    start_index_sample_text: NonNegativeInt

    positional_strength_threshold: float
    positional_count_threshold: float
    dead_threshold: NonNegativeInt

    coactivation_text_corpus: str
    start_index_coact_text: int | None
    n_tokens_coacts: PositiveInt
    to_compute: list[str]

    model_storage_absolute: str
    study_model_name: str
    feature_tool_name: str
    device: str

    # Just in case we want to mess around with different models later
    n_layers: PositiveInt = 26
    features_per_layer: PositiveInt = 98304

    # In case we want to mess around with different textgen params
    max_tokens_activation: PositiveInt = 128
    max_tokens_coactivation: PositiveInt = 128
    activations_batch_size: PositiveInt = 32
    coactivations_batch_size: PositiveInt = 32


    @model_validator(mode="after")
    def __post_init__(self):
        """Validate the parameters after initialization."""

        # Check constraints on sampling parameters
        assert self.n_samples_per_layer <= 15000, f"n_samples_per_layer must be less than or equal to 15000. Not {self.n_samples_per_layer}"
        assert self.sample_method in ["filtered_proportional", "proportional", "uniform", "filtered_uniform"], f"sample_method must be one of 'filtered_proportional', 'proportional', 'uniform', 'filtered_uniform'. Not {self.sample_method}"
        assert self.positional_strength_threshold >= 0, f"positional_strength_threshold must be non-negative. Not {self.positional_strength_threshold}"
        assert self.positional_strength_threshold <= 1, f"positional_strength_threshold must be less than or equal to 1. Not {self.positional_strength_threshold}"
        assert self.positional_count_threshold >= 0, f"positional_count_threshold must be non-negative. Not {self.positional_count_threshold}"
        assert self.positional_count_threshold <= 1, f"positional_count_threshold must be less than or equal to 1. Not {self.positional_count_threshold}"
        for item in self.to_compute:
            assert item in ["virtual", "coactivations", "twera", "era"], f"to_compute must be a list containing 'virtual', 'coactivations', 'twera', or 'era'. Not contain {item}"
        if self.start_index_coact_text is None:
            act_examples = (self.n_tokens_act_freq // self.max_tokens_activation) + (1 if self.n_tokens_act_freq % self.max_tokens_activation != 0 else 0)
            self.start_index_coact_text = act_examples + self.start_index_sample_text




        if self.compute_activations == False:
            assert os.path.exists(self.feature_stats_on_corpus_dir/ "feature_positional_counts.safetensors"), "Activation computation is set to False but can't find feature_positional_counts"
            assert os.path.exists(self.feature_stats_on_corpus_dir/ "feature_positional_strengths.safetensors"), "Activation computation is set to False but can't find feature_positional_strengths"
            assert os.path.exists(self.feature_stats_on_corpus_dir/ "feature_activation_counts.safetensors"), "Activation computation is set to False but can't find feature_activation_counts"
        else:
            print(self.feature_stats_on_corpus_dir)
            assert not os.path.exists(self.feature_stats_on_corpus_dir/ "feature_positional_counts.safetensors"), "Activation computation is set to True but found existing feature_positional_counts. Rename or delete the file to avoid accidental overwriting"
            assert not os.path.exists(self.feature_stats_on_corpus_dir/ "feature_positional_strengths.safetensors"), "Activation computation is set to True but found existing feature_positional_strengths. Rename or delete the file to avoid accidental overwriting"
            assert not os.path.exists(self.feature_stats_on_corpus_dir/ "feature_activation_counts.safetensors"), "Activation computation is set to True but found existing feature_activation_counts. Rename or delete the file to avoid accidental overwriting"
        

        # Check the logic for what we are computing based on the paths that already exist
        network_already_exists = os.path.exists(self.network_dir)
        if network_already_exists:

            # Check that requested computations are compatible with what already exists
            for item in self.to_compute:
                if item == "virtual": assert not os.path.exists(self.virtual_weight_dir), f"Virtual weights for network {self.network_name} already exist but to_compute contains 'virtual'. If you want to use the existing virtual weights, remove 'virtual' from to_compute. If not, you may have to manually delete the folder, as overwriting files from configuration is prohibited."
                if item == "coactivations": assert not os.path.exists(self.coacts_dir), f"Coactivations for network {self.network_name} already exist but to_compute contains 'coactivations'. If you want to use the existing coactivations, remove 'coactivations' from to_compute. If not, you may have to manually delete the folder, as overwriting files from configuration is prohibited."
                if item == "twera":
                    assert not os.path.exists(self.twera_dir), f"TWERA for network {self.network_name} already exist but to_compute contains 'twera'. If you want to use the existing TWERA, remove 'twera' from to_compute. If not, you may have to manually delete the folder, as overwriting files from configuration is prohibited."
                    assert "virtual" in self.to_compute or os.path.exists(self.virtual_weight_dir), f"TWERA computation requires virtual weights. Make sure to add 'virtual' to to_compute or that virtual weights already exist for this network."
                    assert "coactivations" in self.to_compute or os.path.exists(self.coacts_dir), f"TWERA computation requires coactivations. Make sure to add 'coactivations' to to_compute or that coactivations already exist for this network."
                if item == "era":
                    assert not os.path.exists(self.era_dir), f"ERA for network {self.network_name} already exist but to_compute contains 'era'. If you want to use the existing ERA, remove 'era' from to_compute. If not, you may have to manually delete the folder, as overwriting files from configuration is prohibited."
                    assert "virtual" in self.to_compute or os.path.exists(self.virtual_weight_dir), f"ERA computation requires virtual weights. Make sure to add 'virtual' to to_compute or that virtual weights already exist for this network."
                    assert "coactivations" in self.to_compute or os.path.exists(self.coacts_dir), f"ERA computation requires coactivations. Make sure to add 'coactivations' to to_compute or that coactivations already exist for this network."
        else:

            # Make sure we're clear about what needs to be computed
            for item in self.to_compute:
                if item == "twera" or item == "era":
                    # We have to either compute virtual weights and coactivations or they have already exist
                    assert "virtual" in self.to_compute, f"{item} is in to_compute but this requires virtual weights. Make sure to add 'virtual' to to_compute."
                    assert "coactivations" in self.to_compute, f"{item} is in to_compute but this requires coactivations. Make sure to add 'coactivations' to to_compute."

        if self.start_index_coact_text is None:
            # Calculate the start index for the coactivation text to not overlap with the activation text
            self.start_index_coact_text = (self.n_tokens_act_freq // 128) + 1 + self.start_index_sample_text

        
        # Make sure we have all the models downloaded
        assert os.path.exists(self.model_storage_absolute), f"Model to study not found at {self.model_storage_absolute}. Please make sure the path is correct and the model is downloaded."
        assert os.path.exists(Path(self.model_storage_absolute) / ("models--" + self.feature_tool_name.replace("/", "--"))), f"Feature abstraction tool not found at {self.feature_tool_name}. Please make sure the path is correct and the feature abstraction tool is downloaded."
        assert os.path.exists(Path(self.model_storage_absolute) / ("models--" + self.study_model_name.replace("/", "--"))), f"Study model not found at {self.study_model_name}. Please make sure the path is correct and the model is downloaded."
        return self

    @property
    def CLT_dir(self):
        return Path(__file__).resolve().parent.parent
        
    @property
    def feature_stats_on_corpus_dir(self):
        return self.CLT_dir / "activations" / "feature_stats" / (self.activation_text_corpus + "_" + str(int(self.n_tokens_act_freq / 1_000_000)) + "M")
    @property
    def network_dir(self):
        return self.CLT_dir / "sample" / self.network_name
    
    @property
    def virtual_weight_dir(self):
        return self.network_dir / "virtual_weights"
    
    @property
    def coacts_dir(self):
        return self.network_dir / "coactivations"
    
    @property
    def twera_dir(self):
        return self.network_dir / "twera"
    
    @property
    def era_dir(self):
        return self.network_dir / "era"




    def get_sample_params(self):
        return self.model_dump(include = {
            "n_samples_per_layer",
            "sample_method",
            "activation_text_corpus",
            "n_tokens_act_freq",
            "start_index_sample_text",
            "positional_strength_threshold",
            "positional_count_threshold",
            "dead_threshold"
        })
    def lock_sample_params(self):
        if os.path.exists(self.network_dir / "sample_params_lock.json"):
            print("Sample parameters are already locked.")
            return
        sampling_params = self.get_sample_params()

        os.makedirs(self.network_dir, exist_ok=True)
        with open(self.network_dir / "sample_params_lock.json", "w") as f:
            json.dump(sampling_params, f, indent=4)
        print(f"{self.network_name} sample parameters locked")

    def lock_weight_params(self):
        if os.path.exists(self.network_dir / "weight_params_lock.json"):
            print("Weight parameters are already locked.")
            return
        weight_params = self.model_dump(include = {
            "coactivation_text_corpus",
            "start_index_coact_text",
            "n_tokens_coacts"
        })

        os.makedirs(self.network_dir, exist_ok=True)
        with open(self.network_dir / "weight_params_lock.json", "w") as f:
            json.dump(weight_params, f, indent=4)
        print(f"{self.network_name} weight parameters locked")

    def validate_sample_params(self):
        if not os.path.exists(self.network_dir / "sample_params_lock.json"):
            return True
        with open(self.network_dir / "sample_params_lock.json", "r") as f:
            locked_params = json.load(f)
        sampling_params = self.get_sample_params()
        assert sampling_params == locked_params, f"Current sampling parameters do not match locked parameters.\nCurrent: {sampling_params}\nLocked: {locked_params}"
        return True
    
    def validate_weight_params(self):
        if not os.path.exists(self.network_dir / "weight_params_lock.json"):
            return True
        with open(self.network_dir / "weight_params_lock.json", "r") as f:
            locked_params = json.load(f)
        weight_params = self.model_dump(include = {
            "coactivation_text_corpus",
            "start_index_coact_text",
            "n_tokens_coacts"
        })
        assert weight_params == locked_params, f"Current weight parameters do not match locked parameters.\nCurrent: {weight_params}\nLocked: {locked_params}"
        return True
    
    def validate_params(self):
        self.validate_sample_params()
        self.validate_weight_params()




    @classmethod
    def from_yaml(cls, file_path: str):
        """Load parameters from YAML file."""
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return cls(**data)
