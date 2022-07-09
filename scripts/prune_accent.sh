spk_id=p225
accents=English
prune=0.1
python main_cli_prune_accent.fit.py \
  -c cli_config/prune_v1/prune_pretrain.2.yaml \
  --data.init_args.key $spk_id \
  --trainer.default_root_dir output/prune_accent/$accents/$spk_id/prune=$prune
