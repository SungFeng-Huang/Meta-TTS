import os
import numpy as np
import matplotlib.pyplot as plt


class CodebookAnalyzer(object):
    def __init__(self, output_dir):
        self.root = output_dir

    def tsne(self, labels, codes):
        os.makedirs(f"{self.root}/codebook/tsne", exist_ok=True)
    
    def visualize_matching(self, idx, infos):
        os.makedirs(f"{self.root}/codebook/matching", exist_ok=True)
        for info in infos:
            fig = plt.figure(figsize=(32, 16))
            ax = fig.add_subplot(111)
            ax.matshow(info["attn"])
            # fig.colorbar(cax)

            ax.set_title(info["title"], fontsize=28)
            ax.set_xticks(np.arange(len(info["x_labels"])))
            ax.set_xticklabels(info["x_labels"], rotation=90, fontsize=8)
            ax.set_yticks(np.arange(len(info["y_labels"])))
            ax.set_yticklabels(info["y_labels"], fontsize=8)
            plt.savefig(f"{self.root}/codebook/matching/{idx:03d}-{info['title']}.jpg")
            plt.clf()

            if info["quantized"]:
                fig = plt.figure(figsize=(16, 12))
                ax = fig.add_subplot(111)
                column_labels=["Code Index", "Phonemes"]
                ax.axis('off')
                code2phn = {x: [] for x in info["x_labels"]}
                max_positions = np.argmax(info["attn"], axis=1)
                for phn, pos in zip(info["y_labels"], max_positions):
                    code2phn[info["x_labels"][int(pos)]].append(phn)
                data = [[k, ", ".join(v)] for k, v in code2phn.items() if len(v) > 0]

                ax.set_title(info["title"], fontsize=28)
                ax.table(cellText=data,colLabels=column_labels, loc="center", fontsize=12)
                plt.savefig(f"{self.root}/codebook/matching/{idx:03d}-{info['title']}-table.jpg")
                plt.clf()
