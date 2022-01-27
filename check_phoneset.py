from tqdm import tqdm


info_path = 'preprocessed_data/JVS/train.txt'
gt_phn_path = '../MFA/lexicon/JVS-phoneset.txt'

phoneset = set()
with open(info_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        if line == '\n':
            continue
        wav_name, spk, phones, raw_text = line.strip().split('|')
        phns = phones[1:-1].split(' ')
        phoneset |= set(phns)

std_phoneset = set()
with open(gt_phn_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line == '\n':
            continue
        std_phoneset.add(line.strip())
print(std_phoneset - phoneset)
print(phoneset - std_phoneset)
