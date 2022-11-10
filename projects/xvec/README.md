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
### Accent classification on randomly-split test set
| Accent | Count | Acc (\%) |
| --- | --- | --- |
| American | 840 | 99.6 |
| Australian English | 83 | 95.2 |
| British | 40 | 100 |
| Canadian | 316 | 99.7 |
| English | 1363 | 99.6 |
| Indian | 117 | 100 |
| Irish | 363 | 100 |
| NewZealand | 43 | 100 |
| Northern Irish | 258 | 99.6 |
| Scottish | 761 | 99.6 |
| South African | 169 | 99.4 |
| Welsh | 38 | 100 |
| Total || 99.4 |
