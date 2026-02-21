import torch


def transpose_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    for key in state_dict:
        if any(val in key for val in ["c_attn.weight", "c_fc.weight", "c_proj.weight"]):
            state_dict[key] = state_dict[key].T
    if "lm_head.weight" in state_dict:
        del state_dict["lm_head.weight"]
    return state_dict
