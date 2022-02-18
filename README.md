# Meta-TTS: Meta-Learning for Few-shot SpeakerAdaptive Text-to-Speech

This repository is the official implementation of ["Meta-TTS: Meta-Learning for Few-shot SpeakerAdaptive Text-to-Speech"](https://arxiv.org/abs/2111.04040v1).

<!--📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials-->

| multi-task learning | meta learning |
| --- | --- |
| ![](evaluation/images/meta-TTS-multi-task.png) | ![](evaluation/images/meta-TTS-meta-task.png) |

### Meta-TTS

![image](evaluation/images/meta-FastSpeech2.png)

## Requirements

This is how I build my environment, which is not exactly needed to be the same:
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

## Proprocessing
First, download [LibriTTS](https://www.openslr.org/60/) and [VCTK](https://datashare.ed.ac.uk/handle/10283/3443), then change the paths in `config/LibriTTS/preprocess.yaml` and `config/VCTK/preprocess.yaml`, then run
```bash
python3 prepare_align.py config/LibriTTS/preprocess.yaml
python3 prepare_align.py config/VCTK/preprocess.yaml
```
for some preparations.

Alignments of LibriTTS is provided [here](https://github.com/kan-bayashi/LibriTTSLabel.git), and
the alignments of VCTK is provided [here](https://drive.google.com/file/d/1ScLIiyIgLRIZ03DqCmrZ8F75miC77o8g/view?usp=sharing).
You have to unzip the files into `preprocessed_data/LibriTTS/TextGrid/` and
`preprocessed_data/VCTK/TextGrid/`.

Then run the preprocessing script:
```bash
python3 preprocess.py config/LibriTTS/preprocess.yaml

# Copy stats from LibriTTS to VCTK to keep pitch/energy normalization the same shift and bias.
cp preprocessed_data/LibriTTS/stats.json preprocessed_data/VCTK/

python3 preprocess.py config/VCTK/preprocess.yaml
```

## Training

To train the models in the paper, run this command:

```bash
python3 main.py -s train \
                -p config/preprocess/<corpus>.yaml \
                -m config/model/base.yaml \
                -t config/train/base.yaml config/train/<corpus>.yaml \
                -a config/algorithm/<algorithm>.yaml
```

To reproduce, please use 8 V100 GPUs for meta models, and 1 V100 GPU for baseline
models, or else you might need to tune gradient accumulation step (grad_acc_step)
setting in `config/train/base.yaml` to get the correct meta batch size.
Note that each GPU has its own random seed, so even the meta batch size is the
same, different number of GPUs is equivalent to different random seed.

After training, you can find your checkpoints under
`output/ckpt/<corpus>/<project_name>/<experiment_key>/checkpoints/`, where the
project name is set in `config/comet.py`.

To inference the models, run:
```bash
python3 main.py -s test \
                -p config/preprocess/<corpus>.yaml \
                -m config/model/base.yaml \
                -t config/train/base.yaml config/train/<corpus>.yaml \
                -a config/algorithm/<algorithm>.yaml \
                -e <experiment_key> -c <checkpoint_file_name>
```
and the results would be under
`output/result/<corpus>/<experiment_key>/<algorithm>/`.

## Evaluation

> **Note:** The evaluation code is not well-refactored yet.

`cd evaluation/` and check [README.md](evaluation/README.md)

## Pre-trained Models

> **Note:** The checkpoints are with older version, might not capatiable with
> the current code. We would fix the problem in the future.

Since our codes are using Comet logger, you might need to create a dummy
experiment by running:
```Python
from comet_ml import Experiment
experiment = Experiment()
```
then put the checkpoint files under
`output/ckpt/LibriTTS/<project_name>/<experiment_key>/checkpoints/`.

You can download pretrained models [here](https://drive.google.com/drive/folders/1Av7afSMcHX6pp2_ZmpHqfJNx6ONM7N8d?usp=sharing).

## Results

| Corpus | LibriTTS | VCTK |
| --- | --- | --- |
| Speaker Similarity | ![](evaluation/images/LibriTTS/errorbar_plot_encoder.png) | ![](evaluation/images/VCTK/errorbar_plot_encoder.png) |
| Speaker Verification | ![](evaluation/images/LibriTTS/eer_encoder.png)<br>![](evaluation/images/LibriTTS/det_encoder.png) | ![](evaluation/images/VCTK/eer_encoder.png)<br>![](evaluation/images/VCTK/det_encoder.png) |
| Synthesized Speech Detection | ![](evaluation/images/LibriTTS/auc_encoder.png)<br>![](evaluation/images/LibriTTS/roc_encoder.png) | ![](evaluation/images/VCTK/auc_encoder.png)<br>![](evaluation/images/VCTK/roc_encoder.png) |


<!--## Contributing-->

<!--📋  Pick a licence and describe how to contribute to your code repository. -->

