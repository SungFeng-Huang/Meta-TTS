import torch
import s3prl.hub as hub


print(dir(hub))

model_0 = getattr(hub, 'fbank')()  # use classic FBANK
# build the Wav2Vec 2.0 model with pre-trained weights
model_1 = getattr(hub, 'hubert_large_ll60k')()

device = 'cuda'  # or cpu
model_1 = model_1.to(device)
wavs = [torch.randn(160000, dtype=torch.float).to(device)
        for _ in range(16)]  # wav in 16KHz
with torch.no_grad():
    reps = model_1(wavs)["last_hidden_state"]
    print(reps.shape)
