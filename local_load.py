from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import torch
import einops
import os
from safetensors.torch import load_file


def get_model_and_tokenizer(model_path = "./models/gemma-3-1b-pt", device = "auto"):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map = device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


class JumpReLUMultiLayerSAE(nn.Module):
    def __init__(self, d_in, d_sae, num_layers, affine_skip_connection=False):
        # Note that we initialise these to zeros because we're loading in pre-trained weights.
        # If you want to train your own SAEs then we recommend using blah
        super().__init__()
        self.w_enc = nn.Parameter(torch.zeros(num_layers, d_in, d_sae))
        self.w_dec = nn.Parameter(torch.zeros(num_layers, d_sae, num_layers, d_in))
        self.threshold = nn.Parameter(torch.zeros(num_layers, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(num_layers, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(num_layers, d_in))
        if affine_skip_connection:
            self.affine_skip_connection = nn.Parameter(torch.zeros(num_layers, d_in, d_in))
        else:
            self.affine_skip_connection = None

    def encode(self, input_acts):
        pre_acts = einops.einsum(
            input_acts, self.w_enc, "... layer d_in, layer d_in d_sae -> ... layer d_sae"
        ) + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return einops.einsum(
            acts, self.w_dec, "... layer_in d_sae, layer_in d_sae layer_out d_dec -> ... layer_out d_dec"
        ) + self.b_dec

    def forward(self, x):
        acts = self.encode(x)
        recon = self.decode(acts)
        if self.affine_skip_connection is not None:
            return recon + einops.einsum(
                x, self.affine_skip_connection, "... layer d_in, layer d_in d_dec -> ... layer d_dec"
            )
        return recon
    

def get_clt(
    local_dir: str = "./models/gemma-scope-2-1b-pt-clt",
    num_layers: int = 26,
    affine: bool = True,
    device = "cuda",
    half_precision: bool = False,
    ) -> JumpReLUMultiLayerSAE:
    
    params_list = []
    
    for layer_idx in range(num_layers):
        path_to_params = os.path.join(local_dir, f"params_layer_{layer_idx}.safetensors")
        
        params = load_file(path_to_params, device=device)
        if half_precision:
            params = {k: v.half() for k, v in params.items()}
        params_list.append(params)
    
    # Stack all params along the leading "layer" dimension
    params = {
        k: torch.stack([p[k] for p in params_list])
        for k in params_list[0].keys()
    }
    del params_list # free up memory
    
    d_model, d_sae = params["w_enc"].shape[1:]
    sae = JumpReLUMultiLayerSAE(d_model, d_sae, num_layers, affine)
    sae.to(device)
    if half_precision:
        sae = sae.half()
    sae.load_state_dict(params)
    del params

    
    return sae