import pandas as pd
import os
import json
import shutil
import glob
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import librosa

# spk = "p310"
# pipeline = "libri_mask-FT"
# dir = f"{root_dir}/{spk}/{pipeline}/lightning_logs/version_0/fit"

class MOS:
    def __init__(self, source_dir, root_dir):
        self.source_dir = source_dir
        self.root_dir = root_dir

    def last_stage_val_csv(self, dir, stage: str):
        stage_dir = {
            "libri_mask": "mask",
            "mask": "mask",
            "FT": "sup",
            "joint": "sup",
        }[stage]
        csv_name = sorted(
            os.listdir(f"{dir}/train/csv/{stage_dir}"),
            key=lambda x: int(x.split('.')[0].split('=')[-1])
        )[-1]
        return f"{dir}/validate/csv/validate/{csv_name}"

    def average(self, metric):
        outputs = []
        for spk in tqdm(sorted(os.listdir(self.root_dir))):
            output = {}
            for pipeline in sorted(os.listdir(f"{self.root_dir}/{spk}")):
                stages = pipeline.split('-')
                output["speaker"] = spk

                pipeline_dir = f"{self.root_dir}/{spk}/{pipeline}/lightning_logs"
                ver = sorted(
                    os.listdir(pipeline_dir),
                    key=lambda x: int(x.split('_')[-1])
                )[-1]
                dir = f"{pipeline_dir}/{ver}/fit"
                for stage in stages:
                    try:
                        csv_file = self.last_stage_val_csv(dir, stage)
                        df = pd.read_csv(csv_file)
                        score = df[metric].mean()
                        output[(pipeline, stage)] = score
                    except:
                        continue
            outputs.append(output)
        df = pd.DataFrame(outputs).set_index("speaker")#dropna()
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    def calculate_duration(self, paths):
        total_duration = 0

        def _duration(_paths):
            dur = 0
            for p in _paths:
                dur += librosa.get_duration(filename=p)
            return dur

        for spk in tqdm(paths, desc="Speaker"):
            for key in tqdm(paths[spk], desc="pipeline", leave=False):
                if key == "text":
                    continue
                elif key == "ref":
                    total_duration += _duration(paths[spk][key]) / 4
                elif isinstance(paths[spk][key], list):
                    assert key in {"raw", "recon"}
                    total_duration += _duration(paths[spk][key])
                elif isinstance(paths[spk][key], dict):
                    for stage in paths[spk][key]:
                        assert isinstance(paths[spk][key][stage], list)
                        total_duration += _duration(paths[spk][key][stage])
                else:
                    raise
        return total_duration

    def _copy(self, _paths, dst):
        for fpath in _paths:
            shutil.copy2(fpath, dst)

    def copy_files(self, target, paths, desc=None):
        os.makedirs(target, exist_ok=True)
        for key in tqdm(paths, leave=False, desc=desc):
            if isinstance(paths[key], list):
                dst = f"{target}/{key}"
                os.makedirs(dst, exist_ok=True)
                self._copy(paths[key], dst)
            elif isinstance(paths[key], dict):
                self.copy_files(f"{target}/{key}", paths[key], desc=key)
            else:
                raise (paths, target)

    def _get_fname(self, _paths, val_id=None, k=None, basename=True):
        if val_id is not None:
            outputs = []
            for _path in _paths:
                if _path.split('/')[-1].startswith(val_id):
                    if basename:
                        outputs.append(_path.split('/')[-1])
                    else:
                        outputs.append(_path)
            assert len(outputs) == 1
        elif k is not None:
            _paths = [p.split('/')[-1] for p in _paths]
            outputs = random.sample(_paths, k)
        else:
            raise
        return outputs

    def generate_question(self, questions, val_id, paths, prefix=None):
        if isinstance(prefix, str):
            paths = paths[prefix]
        elif isinstance(prefix, list):
            for pref in prefix:
                paths = paths[pref]
            prefix = '/'.join(prefix)

        """
        path keys: ["ref", "text", "raw", "recon", ...]
        """

        questions[val_id] = {"test": []}
        for key in tqdm(paths, desc="pipeline", leave=False):
            if key == "text":
                fname = self._get_fname(
                    paths[key], val_id=val_id, basename=False)[0]
                lab = open(fname, 'r').read()
                questions[val_id][key] = lab
            elif key == "ref":
                questions[val_id]["ref"] = [
                    f"{prefix}/ref/{fname}"
                    for fname in self._get_fname(paths[key], k=2)
                ]
            elif isinstance(paths[key], list):
                assert key in {"raw", "recon"}
                fname = self._get_fname(paths[key], val_id=val_id)[0]
                questions[val_id]["test"].append(f"{prefix}/{key}/{fname}")
            elif isinstance(paths[key], dict):
                for stage in paths[key]:
                    assert isinstance(paths[key][stage], list)
                    fname = self._get_fname(paths[key][stage],
                                        val_id=val_id)[0]
                    questions[val_id]["test"].append(f"{prefix}/{key}/{stage}/{fname}")
            else:
                raise
        random.shuffle(questions[val_id]["test"])

    def generate_mos_csv(self, paths, seed=531):
        random.seed(seed)

        questions = self.generate_questions(paths)
        outputs = self.questions_to_csv(questions)

        return outputs


class AccentMOS(MOS):
    def __init__(self, source_dir, root_dir):
        super().__init__(source_dir, root_dir)

    def average(self, metric, random_state=None):
        outputs = []
        paths = {}

        vctk_csv = f"{self.source_dir}/preprocessed_data/VCTK-speaker-info.csv"
        vctk_df = pd.read_csv(vctk_csv).set_index("ACCENTS")
        for accent in tqdm(vctk_df.index.unique(), desc="accent"):
            if accent == "Unknown":
                continue
            dfs = []
            accent_df = vctk_df[vctk_df.index == accent]
            for spk in tqdm(list(accent_df["pID"]), leave=False, desc="spk"):
                if spk not in os.listdir(self.root_dir):
                    continue
                pipeline_dir = f"{self.root_dir}/{spk}/joint/lightning_logs"
                ver = sorted(
                    os.listdir(pipeline_dir),
                    key=lambda x: int(x.split('_')[-1])
                )[-1]
                dir = f"{pipeline_dir}/{ver}/fit"
                csv_file = self.last_stage_val_csv(dir, "joint")
                df = pd.read_csv(csv_file)
                dfs.append(df)
            dfs = pd.concat(dfs, ignore_index=True)
            dfs = dfs.sample(n=30, random_state=random_state)

            paths[accent] = {}
            for val_id in tqdm(dfs["val_ids"], leave=False, desc="val_id"):
                spk = val_id.split('_')[0]
                paths[accent][val_id] = {}
                output = {"accent": accent, "val_id": val_id}

                for pipeline in sorted(os.listdir(f"{self.root_dir}/{spk}")):
                    stages = pipeline.split('-')

                    pipeline_dir = f"{self.root_dir}/{spk}/{pipeline}/lightning_logs"
                    ver = sorted(
                        os.listdir(pipeline_dir),
                        key=lambda x: int(x.split('_')[-1])
                    )[-1]
                    dir = f"{pipeline_dir}/{ver}/fit"

                    if pipeline == "joint":
                        get_ref_audio_paths(dir, self.source_dir, paths[accent][val_id])

                    for stage in stages:
                        try:
                            csv_file = self.last_stage_val_csv(dir, stage)
                            df = pd.read_csv(csv_file)
                            df = df[df["val_ids"].values == val_id]

                            if 'raw' not in paths[accent][val_id] or 'text' not in paths[accent][val_id]:
                                get_raw_audio_text_paths(df, self.source_dir, paths[accent][val_id])
                            get_audio_paths(df, dir, paths[accent][val_id], pipeline, stage)

                            score = df[metric].item()
                            output[(pipeline, stage)] = score
                        except Exception as e:
                            print(e)
                            continue
                outputs.append(output)
            # print(pd.DataFrame(outputs))

        df = pd.DataFrame(outputs).set_index(["accent", "val_id"])#dropna()
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df, paths

    def generate_questions(self, paths):
        questions = {}

        for accent in tqdm(paths, desc="Accent"):
            for val_id in paths[accent]:
                self.generate_question(questions, val_id, paths,
                                       prefix=[accent, val_id])

        return questions

    def questions_to_csv(self, questions):
        outputs = []
        val_ids = list(questions.keys())
        random.shuffle(val_ids)
        q_per_task = 4

        def _q_to_dict(q, _id):
            out = {}
            for _i, ref_wav in enumerate(q["ref"]):
                out[f"audio_ref_{_id+1}_{_i+1}"] = ref_wav
            for _i, test_wav in enumerate(q["test"]):
                out[f"audio_{_id+1}-{_i+1}"] = test_wav
            out[f"text_{_id+1}"] = q["text"]
            return out

        for i, val_id in enumerate(val_ids):
            if i % q_per_task == 0:
                outputs.append({})
            outputs[-1].update(_q_to_dict(questions[val_id], i%q_per_task))
        return outputs


class SpeakerMOS(MOS):
    def __init__(self, source_dir, root_dir):
        super().__init__(source_dir, root_dir)

    def average(self, metric, random_state=None):
        outputs = []
        paths = {}
        for spk in tqdm(sorted(os.listdir(self.root_dir))):
            output = {}
            output["speaker"] = spk
            paths[spk] = {}

            for pipeline in sorted(os.listdir(f"{self.root_dir}/{spk}")):
                stages = pipeline.split('-')

                pipeline_dir = f"{self.root_dir}/{spk}/{pipeline}/lightning_logs"
                ver = sorted(
                    os.listdir(pipeline_dir),
                    key=lambda x: int(x.split('_')[-1])
                )[-1]
                dir = f"{pipeline_dir}/{ver}/fit"

                if pipeline == "joint":
                    get_ref_audio_paths(dir, self.source_dir, paths[spk])

                for stage in stages:
                    try:
                        csv_file = self.last_stage_val_csv(dir, stage)
                        df = pd.read_csv(csv_file)
                        df = df.sample(n=3, random_state=random_state)

                        if 'raw' not in paths[spk] or 'text' not in paths[spk]:
                            get_raw_audio_text_paths(df, self.source_dir, paths[spk])
                        get_audio_paths(df, dir, paths[spk], pipeline, stage)

                        score = df[metric].mean()
                        output[(pipeline, stage)] = score
                    except Exception as e:
                        print(e)
                        continue
            outputs.append(output)

        df = pd.DataFrame(outputs).set_index("speaker")#dropna()
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df, paths

    def generate_questions(self, paths):
        questions = {}

        for spk in tqdm(paths, desc="Speaker"):
            val_ids = [p.split('/')[-1].split('.')[0] for p in paths[spk]['raw']]
            for val_id in val_ids:
                self.generate_question(questions, val_id, paths, prefix=spk)
                # prefix = f"{accent}/{val_id}"
                # paths[accent][val_id]

        return questions

    def questions_to_csv(self, questions):
        outputs = []
        val_ids = list(questions.keys())
        random.shuffle(val_ids)
        q_per_task = 4

        def _q_to_dict(q, _id):
            out = {}
            for _i, ref_wav in enumerate(q["ref"]):
                out[f"audio_ref_{_id+1}_{_i+1}"] = ref_wav
            for _i, test_wav in enumerate(q["test"]):
                out[f"audio_{_id+1}-{_i+1}"] = test_wav
            out[f"text_{_id+1}"] = q["text"]
            return out

        for i, val_id in enumerate(val_ids):
            if i % q_per_task == 0:
                outputs.append({})
            outputs[-1].update(_q_to_dict(questions[val_id], i%q_per_task))
        return outputs


def get_raw_audio_text_paths(df, source_dir, outputs):
    if 'raw' not in outputs:
        outputs['raw'] = []
    if 'text' not in outputs:
        outputs['text'] = []

    for row in df.itertuples():
        val_ids = row.val_ids

        spk_id = val_ids.split('_')[0]
        raw_wav = glob.glob(f"{source_dir}/raw_data/VCTK/*/*/{spk_id}/{val_ids}.wav")[0]
        outputs['raw'].append(raw_wav)

        text = glob.glob(f"{source_dir}/raw_data/VCTK/*/*/{spk_id}/{val_ids}.lab")[0]
        outputs['text'].append(text)

    return outputs

def get_ref_audio_paths(dir, source_dir, outputs):
    if 'ref' not in outputs:
        outputs['ref'] = []
    sup_dir = f"{dir}/train/audio/sup"
    sup_ids = sorted(os.listdir(sup_dir))
    spk_id = sup_ids[0].split('_')[0]
    for sup_id in sup_ids:
        ref_wav = glob.glob(f"{source_dir}/raw_data/VCTK/*/*/{spk_id}/{sup_id}.wav")[0]
        outputs['ref'].append(ref_wav)
    return outputs

def get_audio_paths(df, dir, outputs, pipeline, stage):
    audio_dir = f"{dir}/validate/audio/validate"
    if 'recon' not in outputs:
        outputs['recon'] = []
    if pipeline not in outputs:
        outputs[pipeline] = {stage: []}
    elif stage not in outputs[pipeline]:
        outputs[pipeline][stage] = []

    for row in df.itertuples():
        epoch = row.epoch
        val_ids = row.val_ids
        val_id_dir = f"{audio_dir}/{val_ids}"

        if len(outputs['recon']) < df.shape[0]:
            recon_wav = f"{val_ids}.recon.wav"
            outputs['recon'].append(f"{val_id_dir}/{recon_wav}")

        synth_wav = f"{val_ids}-epoch={epoch}-batch_idx=0.wav"
        outputs[pipeline][stage].append(f"{val_id_dir}/{synth_wav}")

    return outputs


if __name__ == "__main__":
    source_dir = "."
    root_dir = f"{source_dir}/output/learnable_structured_pipeline"
    target_dir = "/home/r06942045/myProjects/voice-clone-pruning-mos"
    # metric = 'sparsity'
    metric = 'val_accent_acc'

    # mos = MOS(source_dir, root_dir)
    # df = mos.average(metric)
    # print(df)
    # print(df.mean())
    # print(df.std())

    # accent_mos = AccentMOS(source_dir, root_dir)
    # _df, paths = accent_mos.average(metric, 531)
    # print(_df)
    # print(_df.mean())
    # print(_df.std())
    # print(json.dumps(paths, indent=4))
    #
    # accent_mos.copy_files(f"{target_dir}/amos", paths, "Accent MOS")
    # random.seed(531)
    # questions = accent_mos.generate_questions(paths)
    # print(json.dumps(questions, indent=4))
    # exit()

    # spk sample
    spk_mos = SpeakerMOS(source_dir, root_dir)
    _df, paths = spk_mos.average(metric, 531)
    print(_df)
    print(_df.mean())
    print(_df.std())
    # print(json.dumps(paths, indent=4))

    # spk_mos.copy_files(f"{target_dir}/audio", paths)

    # total_duration = mos.calculate_duration(paths)
    # print("Total duration:", total_duration)
    # print("Duration per speaker:", total_duration/108)

    # outputs = spk_mos.generate_mos_csv(paths)
    # print(outputs)
    # csv_df = pd.DataFrame(outputs)
    # # csv_df.to_csv(f"{target_dir}/mos.csv", index=False)
    #
    # import math
    # q_per_csv = math.floor(50 / 4.2)
    # n_csv = math.ceil(csv_df.shape[0] / q_per_csv)
    # for i in range(n_csv):
    #     row_slice = slice(i * q_per_csv, (i+1) * q_per_csv)
    #     csv_df[row_slice].to_csv(f"{target_dir}/mos_{i}.csv", index=False)
