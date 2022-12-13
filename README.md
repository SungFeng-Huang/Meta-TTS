# Meta-TTS: Meta-Learning for Few-shot SpeakerAdaptive Text-to-Speech

This is the official repository of the following papers and my doctoral dissertation (still working on it):
- [Meta-TTS: Meta-Learning for Few-shot Speaker Adaptive Text-to-Speech](https://doi.org/10.1109/TASLP.2022.3167258)
- [Few-Shot Cross-Lingual TTS Using Transferable Phoneme Embedding](https://arxiv.org/abs/2206.15427)
- [PERSONALIZED LIGHTWEIGHT TEXT-TO-SPEECH: VOICE CLONING WITH ADAPTIVE STRUCTURED PRUNING]() (not publically available)


## Requirements

This is how I build my environment, which is not needed to be exactly the same:
- Sign up for [Comet.ml](https://www.comet.ml/), find out your workspace and API key via [www.comet.ml/api/my/settings](www.comet.ml/api/my/settings) and fill them in `config/comet.py`. Comet logger is used throughout train/val/test stages.
  - Check my training logs [here](https://www.comet.ml/b02901071/meta-tts/view/Zvh3Lz3Wvy2AiWcinD06TaS0G).
- [Optional] Install [pyenv](https://github.com/pyenv/pyenv.git) for Python version
  control, change to Python 3.8.6.
```bash
# After download and install pyenv:
pyenv install 3.8.6
pyenv local 3.8.6
```
- [Optional] Install [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv.git) as a plugin of pyenv for clean virtual environment.
```bash
# After install pyenv-virtualenv
pyenv virtualenv meta-tts
pyenv activate meta-tts
```
- Install requirements:
```bash
pip install -r requirements.txt
```

## Preprocessing

> **Note:** The following is the old preprocessing guidelines. We are currently working on making the main process on-line.
> Checkout `src/data` for new preprocessing guidelines.

First, download [LibriTTS](https://www.openslr.org/60/) and [VCTK](https://datashare.ed.ac.uk/handle/10283/3443), then change the paths in `config/preprocess/LibriTTS.yaml` and `config/preprocess/VCTK.yaml`, then run
```bash
python3 prepare_align.py config/preprocess/LibriTTS.yaml
python3 prepare_align.py config/preprocess/VCTK.yaml
```
for some preparations.

Alignments of LibriTTS is provided [here](https://github.com/kan-bayashi/LibriTTSLabel.git), and
the alignments of VCTK is provided [here](https://drive.google.com/file/d/1ScLIiyIgLRIZ03DqCmrZ8F75miC77o8g/view?usp=sharing).
You have to unzip the files into `preprocessed_data/LibriTTS/TextGrid/` and
`preprocessed_data/VCTK/TextGrid/`.

Then run the preprocessing script:
```bash
python3 preprocess.py config/preprocess/LibriTTS.yaml

# Copy stats from LibriTTS to VCTK to keep pitch/energy normalization the same shift and bias.
cp preprocessed_data/LibriTTS/stats.json preprocessed_data/VCTK/

python3 preprocess.py config/preprocess/VCTK.yaml
```

## Training, Evaluation, Pre-trained Models, Results

We are moving codes of different papers into `projects`.
Checkout their docs respectively:
- [Meta-TTS: Meta-Learning for Few-shot Speaker Adaptive Text-to-Speech](https://doi.org/10.1109/TASLP.2022.3167258)
  - Document: [meta-tts/README.md](./projects/meta-tts/README.md)
- [Few-Shot Cross-Lingual TTS Using Transferable Phoneme Embedding](https://arxiv.org/abs/2206.15427)
  - Document: [cross-lingual/README.md](./projects/cross-lingual/README.md)
- [PERSONALIZED LIGHTWEIGHT TEXT-TO-SPEECH: VOICE CLONING WITH ADAPTIVE STRUCTURED PRUNING]()
  - Document: [prune/README.md](./projects/prune/README.md)


