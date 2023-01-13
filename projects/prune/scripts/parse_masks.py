import torch
import glob


def extract_masks(ckpt, submodule=""):
    params = [
        ckpt["mask_state_dict"][k] if not k.startswith("speaker_emb")
        else ckpt["mask_state_dict"][k][ckpt["speaker_args"]]
        for k in ckpt["mask_state_dict"] if k.startswith(submodule)
    ]
    return torch.nn.utils.parameters_to_vector(params)

def inter_compare(ckpt_dir1, ckpt_dir2):
    path1 = max(
        glob.glob(f"{ckpt_dir1}/epoch=*"),
        key=lambda x: int(x.split('/')[-1].split('-')[0][6:]))
    path2 = max(
        glob.glob(f"{ckpt_dir2}/epoch=*"),
        key=lambda x: int(x.split('/')[-1].split('-')[0][6:]))

    masks = []
    for path in [path1, path2]:
        print(path)
        ckpt = torch.load(path, map_location='cpu')
        modules = set([k.split('.')[0] for k in ckpt["mask_state_dict"]])
        for submodule in modules:
            print(submodule, "sparsity =", 1 - extract_masks(ckpt, submodule).mean())
        mask = extract_masks(ckpt)
        masks.append(mask)
    xor = (masks[0] != masks[1]).sum() / masks[0].shape[0]
    print(xor)

def intra_compare(ckpt_dir):
    masks = []
    for path in sorted(
            glob.glob(f"{ckpt_dir}/epoch=*"),
            key=lambda x: int(x.split('/')[-1].split('-')[0][6:])):
        print(path)
        ckpt = torch.load(path, map_location='cpu')
        modules = set([k.split('.')[0] for k in ckpt["mask_state_dict"]])
        for submodule in modules:
            print(submodule, "sparsity =", 1 - extract_masks(ckpt, submodule).mean())
        mask = extract_masks(ckpt)
        masks.append(mask)
    masks = torch.stack(masks)
    xor = torch.empty(masks.shape[0], masks.shape[0])
    for i in range(masks.shape[0]):
        for j in range(masks.shape[0]):
            xor[i, j] = (masks[i] != masks[j]).sum()
    print(xor / masks.shape[1])

def get_ckpt_dir(speaker, version):
    return f"output/prune_accent/{speaker}/lightning_logs/version_{version}/checkpoints"


if __name__ == "__main__":
    speaker = "p251"
    version = 81
    inter_compare(
        get_ckpt_dir(speaker, 83), get_ckpt_dir(speaker, 84))
