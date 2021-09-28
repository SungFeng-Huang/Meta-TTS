from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np


fpath = Path("raw_data/LibriTTS/test-clean/1089/1089_134686_000001_000001.wav")
wav = preprocess_wav(fpath)

encoder = VoiceEncoder()
embed = encoder.embed_utterance(wav)
# np.set_printoptions(precision=3, suppress=True)
print(embed.shape)
