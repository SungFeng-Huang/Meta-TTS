accents=$1
spk_id=$2
prune=$3
max_epochs=$4
python main_cli_prune_accent.fit.py \
  -c cli_config/prune_v1/prune_pretrain.2.yaml \
  --model.init_args.qry_patience 5 \
  --data.init_args.key $spk_id \
  --data.init_args.steps_per_epoch 200 \
  --trainer.default_root_dir new_output/prune_accent/$accents/$spk_id/prune=$prune \
  --trainer.max_epochs $max_epochs \
  --model_pruning_callback.amount $prune
