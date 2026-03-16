import torch
from functools import partial


def gather_acts_hook(mod, inputs, outputs, cache: dict, key: str, use_input: bool):
    """Generic hook function which stores activations (either input or output of a particular PyTorch module)."""
    acts = inputs[0] if use_input else outputs
    cache[key] = acts
    return outputs


# def gather_clt_activations(model, num_layers, inputs):
#     act_cache = {}
#     handles = []
#     for layer in range(num_layers):
#         handle_input = model.model.layers[layer].pre_feedforward_layernorm.register_forward_hook(
#             partial(gather_acts_hook, cache=act_cache, key=f"input_{layer}", use_input=False)
#         )
#         handle_target = model.model.layers[layer].post_feedforward_layernorm.register_forward_hook(
#             partial(gather_acts_hook, cache=act_cache, key=f"target_{layer}", use_input=False)
#         )
#         handles.extend([handle_input, handle_target])
#     try:
#         _ = model.forward(inputs)
#     finally:
#         for handle in handles:
#             handle.remove()

#     return (
#         torch.stack([act_cache[f"input_{layer}"] for layer in range(num_layers)], axis=-2).to(model.device),
#         torch.stack([act_cache[f"target_{layer}"] for layer in range(num_layers)], axis=-2).to(model.device),
#     )


def gather_clt_activations(model, num_layers, inputs):
    act_cache = {}
    handles = []
    for layer in range(num_layers):
        handle_input = model.model.layers[layer].pre_feedforward_layernorm.register_forward_hook(
            partial(gather_acts_hook, cache=act_cache, key=f"input_{layer}", use_input=False)
        )
        handle_target = model.model.layers[layer].post_feedforward_layernorm.register_forward_hook(
            partial(gather_acts_hook, cache=act_cache, key=f"target_{layer}", use_input=False)
        )
        handles.extend([handle_input, handle_target])
    
    try:
        _ = model.forward(inputs)
        
    finally:
        for handle in handles:
            handle.remove()

    # Stack along dimension 1 (layer dimension) instead
    input_acts = torch.stack([act_cache[f"input_{layer}"] for layer in range(num_layers)], axis=-2)
    target_acts = torch.stack([act_cache[f"target_{layer}"] for layer in range(num_layers)], axis=-2)
    
    return input_acts.to(model.device), target_acts.to(model.device)
