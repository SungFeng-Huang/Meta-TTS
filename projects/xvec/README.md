# X-vector

## Requirements

## Preprocessing

## Training
First `cd` to the root directory of this repository.

```bash
# accent classifier
python projects/xvec/main_cli.py fit -c cli_config/xvec/fit.accent.yaml

# speaker classifier
python projects/xvec/main_cli.py fit -c cli_config/xvec/fit.accent.yaml
```

## Evaluation
First `cd` to the root directory of this repository.

```bash
# accent classifier
python projects/xvec/main_cli.py test -c cli_config/xvec/test.accent.yaml --ckpt_path <ckpt_path>

# speaker classifier
python projects/xvec/main_cli.py test -c cli_config/xvec/test.speaker.yaml --ckpt_path <ckpt_path>
```

## Results
### Accent classification
| Accent | Count | Acc (\%) |
| --- | --- | --- |
| American || 99.6 |
| Australian English || 95.2 |
| British || 100 |
| Canadian || 99.7 |
| English || 99.6 |
| Indian || 100 |
| Irish || 100 |
| NewZealand || 100 |
| Northern Irish || 99.6 |
| Scottish | 99.6 |
| South African || 99.4 |
| Welsh || 100 |
| Total || 99.4 |
