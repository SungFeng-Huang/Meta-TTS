# Using LightningCLI

## Train + Validation
```bash
python main_cli.py fit -c cli_config/xvec/fit.yaml
```

## Test
```bash
python main_cli.py test -c cli_config/xvec/test.yaml \
  --ckpt_path output/xvec/lightning_logs/version_?/checkpoints/?.ckpt
```
