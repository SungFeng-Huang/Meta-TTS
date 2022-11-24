# Requirement

Install ```dlhlp_lib``` first.
```
git clone https://github.com/hhhaaahhhaa/dlhlp-lib.git
cd dlhlp-lib
pip install -e ./
```
# Preprocess

```
python preprocess_v2.py [raw_dir] [preprocessed_dir] --dataset [DATASET_TAG] [--parse_raw] [--prepare_mfa] [--mfa] [--preprocess] [--create_dataset] [--force]
```

```DATASET_TAG``` can be one of the ```LibriTTS```, ```AISHELL-3```, ```KSS```, ```JSUT```, ```CSS10```, ```TAT```, ```TAT_TTS```

# TextGrid

We provide MFA results for LJSpeech, LibriTTS, AISHELL-3, download it [here](https://drive.google.com/drive/folders/1OyEh823slo4Taw9A-zlC9ruS45hz8Y81?usp=share_link). Unzip to corresponding ```preprocessed_dir```, e.g. unzip ```LibriTTS.zip``` under ```preprocessed_data/LibriTTS/```.