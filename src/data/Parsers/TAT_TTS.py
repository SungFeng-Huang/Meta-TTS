import os
from tqdm import tqdm
import json

# TODO: not done yet (transposition problems...)

class TATTTSRawParser(object):
    def __init__(self, root):
        self.root = root
        self.data = None

    def parse(self):
        self.data = {"data": [], "data_info": [], "all_speakers": []}
        for speaker in tqdm(os.listdir(self.root)):
            self.data["all_speakers"].append(speaker)
            for partition in os.listdir(f"{self.root}/{speaker}"):
                if not os.isdir(f"{self.root}/{speaker}/{partition}"):
                    continue
                for filename in os.listdir(f"{self.root}/{speaker}/{partition}"):
                    if filename[-4:] != ".wav":
                        continue
                    basename = filename[:-4]
                    wav_path = f"{self.root}/{speaker}/{partition}/{basename}.wav"
                    text_path = f"{self.root}/{speaker}/{partition}/{basename}.json"
                    with open(text_path, "r", encoding="utf-8") as f:
                        text_labels = json.load(f)
                    data = {
                        "wav_path": wav_path,
                        "text": text_labels["台羅數字調"],
                    }
                    data_info = {
                        "spk": speaker,
                        "basename": basename,
                        "partition": partition,
                    }
                    self.data["data"].append(data)
                    self.data["data_info"].append(data_info)
