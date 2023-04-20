# Meta-TTS: Meta-Learning for Few-shot SpeakerAdaptive Text-to-Speech

This repository is the official implementation of ["Meta-TTS: Meta-Learning for Few-shot Speaker Adaptive Text-to-Speech"](https://doi.org/10.1109/TASLP.2022.3167258).

<!--📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials-->

| multi-task learning | meta learning |
| --- | --- |
| ![](/evaluation/images/meta-TTS-multi-task.png) | ![](/evaluation/images/meta-TTS-meta-task.png) |

### Meta-TTS

![image](/evaluation/images/meta-FastSpeech2.png)


## Requirements


## Preprocessing

### Offline preprocessing (old)
Offline extract mel-spectrogram, prosody features, etc.
- Document: [preprocessor/README.md](/preprocessor/README.md)
- Configs: [cli_config/meta-tts](/cli_config/meta-tts/)

### Online preprocessing (new)
> Still in progress.


## Training

To train the models in the paper, run this command:

```bash
python3 projects/meta-tts/main.py \
    -s train \
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
python3 projects/meta-tts/main.py \
    -s test \
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

`cd evaluation/` and check [README.md](/evaluation/README.md)


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
| Speaker Similarity | ![](/evaluation/images/LibriTTS/errorbar_plot_encoder.png) | ![](/evaluation/images/VCTK/errorbar_plot_encoder.png) |
| Speaker Verification | ![](/evaluation/images/LibriTTS/eer_encoder.png)<br>![](/evaluation/images/LibriTTS/det_encoder.png) | ![](../../evaluation/images/VCTK/eer_encoder.png)<br>![](../../evaluation/images/VCTK/det_encoder.png) |
| Synthesized Speech Detection | ![](/evaluation/images/LibriTTS/auc_encoder.png)<br>![](/evaluation/images/LibriTTS/roc_encoder.png) | ![](../../evaluation/images/VCTK/auc_encoder.png)<br>![](../../evaluation/images/VCTK/roc_encoder.png) |


<!--## Contributing-->

<!--📋  Pick a licence and describe how to contribute to your code repository. -->

