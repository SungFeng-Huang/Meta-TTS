# X-vector

## Requirements

## Preprocessing

### Old preprocessing
Offline extract mel-spectrogram, prosody features, etc.
- Document: [README.md#preprocessing](/README.md#preprocessing)
- Configs: [cli_config/xvec](/cli_config/xvec/)

### New preprocessing
Online extract mel-spectrogram, prosody features, etc.
- Document: [src/data/README.md](/src/data/README.md)
- Configs: [cli_config/xvec_online](/cli_config/xvec_online)

## Training

```bash
# First, cd to repository's root
cd `git rev-parse --show-toplevel`

# train accent classifier
python projects/xvec/main_cli.py fit -c cli_config/xvec/fit.accent.yaml

# train speaker classifier
python projects/xvec/main_cli.py fit -c cli_config/xvec/fit.accent.yaml
```

## Evaluation

```bash
# First, cd to repository's root
cd `git rev-parse --show-toplevel`

# test accent classifier
python projects/xvec/main_cli.py test -c cli_config/xvec/test.accent.yaml --ckpt_path <ckpt_path>

# test speaker classifier
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
