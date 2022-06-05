# Using LightningCLI

## Train + Validation
```bash
python main_cli.py -c cli_config/xvec/config.yaml fit
```

## Test
```bash
python main_cli.py -c cli_config/xvec/config.yaml test \
  --ckpt_path output/xvec/lightning_logs/version_?/checkpoints/?.ckpt
```
