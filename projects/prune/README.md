# Personalized Lightweight Text-To-Speech: Voice Cloning with Adaptive Structured Pruning

The code is still refactoring.


## Requirements

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
# lottery ticket hypothesis
python projects/prune/main_lottery_ticket.py fit -c cli_config/prune_v1/lottery_ticket.yaml

# learnable unstructured pruning (not used)
python projects/prune/main_learnable_unstructured.py fit -c cli_config/prune_v1/learnable_unstructured.yaml

# learnable structured pruning
python projects/prune/main_learnable_structured.py fit -c cli_config/prune_v1/learnable_structured_pipeline.*.yaml
```


## Evaluation


## Results
