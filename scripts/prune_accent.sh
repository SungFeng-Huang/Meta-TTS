accents=$1
spk_id=$2
prune=$3
python main_cli_prune_accent.fit.py \
  -c cli_config/prune_v1/prune_pretrain.2.yaml \
  --data.init_args.key $spk_id \
  --trainer.default_root_dir new_output/prune_accent/$accents/$spk_id/prune=$prune \
  --model_pruning_callback.amount $prune
  # --data.init_args.steps_per_epoch 200 \
