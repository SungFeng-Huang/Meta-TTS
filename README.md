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

### Offline preprocessing (old)
Offline extract mel-spectrogram, prosody features, etc.
- Document: [preprocessor/README.md](/preprocessor/README.md)

### Online preprocessing (new)
Online extract mel-spectrogram, prosody features, etc.
- Document: [src/data/README.md](/src/data/README.md)


## Training, Evaluation, Pre-trained Models, Results

We are moving codes of different papers into `projects`.
Checkout their docs respectively:
- [Meta-TTS: Meta-Learning for Few-shot Speaker Adaptive Text-to-Speech](https://doi.org/10.1109/TASLP.2022.3167258)
  - Document: [meta-tts/README.md](/projects/meta-tts/README.md)
- [Few-Shot Cross-Lingual TTS Using Transferable Phoneme Embedding](https://arxiv.org/abs/2206.15427)
  - Document: [cross-lingual/README.md](/projects/cross_lingual/README.md)
- [PERSONALIZED LIGHTWEIGHT TEXT-TO-SPEECH: VOICE CLONING WITH ADAPTIVE STRUCTURED PRUNING]()
  - Document: [prune/README.md](/projects/prune/README.md)


