# Preprocessor

This is the guideline for offline extract mel-spectrogram, prosody features, etc.

First `cd` to the root directory of this repository.
Download [LibriTTS](https://www.openslr.org/60/) and [VCTK](https://datashare.ed.ac.uk/handle/10283/3443), then change the paths in `config/preprocess/LibriTTS.yaml` and `config/preprocess/VCTK.yaml`, then run
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
