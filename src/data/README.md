# Requirement

Install `src/data/requirements.txt`.

# Preprocess

First `cd` to the root directory of this repository.

```
python preprocess_v2.py [raw_dir] [preprocessed_dir] --dataset [DATASET_TAG] [--parse_raw] [--prepare_mfa] [--mfa] [--preprocess] [--create_dataset [data_info_json_path]] [--force]
```

`DATASET_TAG` can be one of the `LibriTTS`, `AISHELL-3`, `KSS`, `JSUT`, `CSS10`, `TAT`, `TAT_TTS`

For example:
```
python preprocess_v2.py raw_data/LibriTTS preprocessed_data/LibriTTS --dataset LibriTTS [--parse_raw] [--prepare_mfa] [--mfa] [--preprocess] [--create_dataset [data_info_json_path]] [--force]
```

# TextGrid

We provide MFA results for LJSpeech, LibriTTS, AISHELL-3, download it [here](https://drive.google.com/drive/folders/1OyEh823slo4Taw9A-zlC9ruS45hz8Y81?usp=share_link). Unzip to corresponding `preprocessed_dir`, e.g. unzip `LibriTTS.zip` under `preprocessed_data/LibriTTS/`.

# Create train/test split
