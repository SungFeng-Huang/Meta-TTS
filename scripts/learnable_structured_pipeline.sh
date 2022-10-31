spk_id=$1
pipeline=$2
python main_cli_learnable_prune.py fit\
  -c cli_config/prune_v1/learnable_structured_pipeline.$pipeline.yaml \
  --data.init_args.key $spk_id \
  --trainer.default_root_dir output/learnable_structured_pipeline/$spk_id/$pipeline
