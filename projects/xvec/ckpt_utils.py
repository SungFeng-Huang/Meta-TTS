import torch
import json


def print_ckpt_stats(ckpt):
    print(ckpt.keys())
    for key in ckpt:
        if key == "state_dict":
            print(key, ckpt[key].keys())
        elif key == "optimizer_states":
            continue
            print(key, len(ckpt[key]), ckpt[key][0].keys())
            # print(ckpt[key]["state"])
            # print(ckpt[key]["param_groups"])
        else:
            continue
            print(key, ckpt[key])
    print("fc3.weight", ckpt["state_dict"]["fc3.weight"].shape)
    print("fc3.bias", ckpt["state_dict"]["fc3.bias"].shape, ckpt["state_dict"]["fc3.bias"])

def load_speaker_map(filename):
    """
    Return:
        dict of speaker-to-id map or list of speakers.
    """
    with open(filename, 'r') as f:
        speaker_map = json.load(f)
    return speaker_map

def update_ckpt_speaker_map(ckpt, old_speaker_map, new_speaker_map, mode):
    """
    Args:
        mode:
            - "test": preserve original output posterior.
                Setting unseen classes' output bias to -inf s.t. their output
                probability woud be 0.
            - "retrain": set unseen classes' output layer to randn for retraining.
                Delete ckpt["optimizer_states"] to avoid loading old training
                states.
    """
    fc3_weight = ckpt["state_dict"]["fc3.weight"]
    fc3_bias = ckpt["state_dict"]["fc3.bias"]
    new_fc3_weight = torch.randn(len(new_speaker_map), fc3_weight.shape[1],
                                 device=fc3_weight.device)
    if mode == "test":
        new_fc3_bias = torch.full((len(new_speaker_map),), -1*float("Inf"),
                                device=fc3_bias.device)
    elif mode == "retrain":
        new_fc3_bias = torch.randn((len(new_speaker_map),),
                                   device=fc3_bias.device)

    for i, spk in enumerate(new_speaker_map):
        if spk in old_speaker_map:
            if isinstance(old_speaker_map, dict):
                spk_id = old_speaker_map[spk]
            elif isinstance(old_speaker_map, list):
                spk_id = old_speaker_map.index(spk)
            new_fc3_weight[i] = fc3_weight[spk_id]
            new_fc3_bias[i] = fc3_bias[spk_id]
    ckpt["state_dict"]["fc3.weight"] = new_fc3_weight
    ckpt["state_dict"]["fc3.bias"] = new_fc3_bias
    if mode == "retrain":
        del ckpt["optimizer_states"]


if __name__ == "__main__":
    ckpt = torch.load("output/xvec/lightning_logs/version_99/checkpoints/epoch=49-step=377950.ckpt")
    print_ckpt_stats(ckpt)

    old_speaker_map = load_speaker_map("preprocessed_data/LibriTTS_VCTK/speakers.json")
    new_speaker_map = load_speaker_map("new_data/preprocessed_data/LibriTTS/speakers.json")\
        + load_speaker_map("new_data/preprocessed_data/VCTK/speakers.json")

    mode = "retrain"
    update_ckpt_speaker_map(ckpt, old_speaker_map, new_speaker_map, mode=mode)
    print_ckpt_stats(ckpt)
    torch.save(ckpt, "output/xvec/speaker.updated.ckpt")
    print_ckpt_stats(torch.load("output/xvec/speaker.updated.ckpt"))
