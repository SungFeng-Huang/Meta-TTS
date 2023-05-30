# Personalized Lightweight Text-To-Speech: Voice Cloning with Adaptive Structured Pruning

The code is still refactoring.


## Requirements
Checkout [README.md#requirements](/README.md#requirements).

## Preprocessing
### Offline preprocessing (old)
Offline extract mel-spectrogram, prosody features, etc.
- Document: [preprocessor/README.md](/preprocessor/README.md)
- Configs: [cli_config/prune](/cli_config/prune_v1/)

### Online preprocessing (new)
> Still in progress.


## Training
First `cd` to the root directory of this repository.

```bash
# lottery ticket hypothesis (not used)
python projects/prune/main_lottery_ticket.py fit -c cli_config/prune_v1/lottery_ticket.yaml

# learnable unstructured pruning (not used)
python projects/prune/main_learnable_unstructured.py fit -c cli_config/prune_v1/learnable_unstructured.yaml

# learnable structured pruning
python projects/prune/main_learnable_structured.py fit -c cli_config/prune_v1/learnable_structured_pipeline.*.yaml
```


## Evaluation

To get **"pruned" small model** instead of **"masked" large model** from checkpoint, checkout `ckpt_utils.py`, change the ckpt_path, then run:
```bash
python projects/prune/ckpt_utils.py --ckpt_path <ckpt_path> --output_path <output_path>
```


## Results


## Checkpoints
Would continuously update to hugging face...

Uploaded:
```
spk_id,repo_url
s5,https://huggingface.co/sungfeng/learnable_structured_prune_s5
p292,https://huggingface.co/sungfeng/learnable_structured_prune_p292
p245,https://huggingface.co/sungfeng/learnable_structured_prune_p245
p252,https://huggingface.co/sungfeng/learnable_structured_prune_p252
```