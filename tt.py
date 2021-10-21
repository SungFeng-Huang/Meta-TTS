from lightning.datamodules.ml_meta_datamodule import MultilingualMetaDataModule


if __name__ == "__main__":
    import yaml
    import time
    preprocess_configs = ['config/preprocess/LibriTTS.yaml']
    train_configs = ['config/train/base.yaml', 'config/train/LibriTTS.yaml']
    algorithm_config = 'config/algorithm/ml_meta_dotsim_va_d.yaml'
    log_dir, result_dir = "./logs", "./results"

    st = time.time()
    # Read Config
    preprocess_configs = [yaml.load(
        open(path, "r"), Loader=yaml.FullLoader
    ) for path in preprocess_configs]
    train_config = yaml.load(
        open(train_configs[0], "r"), Loader=yaml.FullLoader)
    train_config.update(
        yaml.load(open(train_configs[1], "r"), Loader=yaml.FullLoader))
    algorithm_config = yaml.load(
        open(algorithm_config, "r"), Loader=yaml.FullLoader)
    print(f"Loading config: {time.time() - st:.2f}s")
    st = time.time()

    datamodule = MultilingualMetaDataModule(
        preprocess_configs, train_config, algorithm_config, log_dir, result_dir)
    datamodule.setup('fit')
    print(f"Loading datamodule: {time.time() - st:.2f}s")
    st = time.time()

    sup_out, qry_out, ref_p_embedding = datamodule.train_task_dataset.__getitem__(
        0)
    print(len(sup_out[0]), len(qry_out[0]))
    print(ref_p_embedding[0].shape)
    print(f"Loading data: {time.time() - st:.2f}s")

    dataloader = datamodule.train_dataloader()
    st = time.time()
    total_iter = 100
    for i, data in enumerate(dataloader):
        sup_out, qry_out, ref_p_embedding = data[0]
        print(len(sup_out[0][0]), len(qry_out[0][0]))
        print(ref_p_embedding[0].shape)
        if i == total_iter - 1:
            break
    print(f"Mean loading time: {(time.time() - st) / total_iter:.2f}s")
