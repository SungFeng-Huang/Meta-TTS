INPUT=preprocessed_data/VCTK-speaker-info.csv
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }

prune=0.1
col_a="pID"
col_b="ACCENTS"
loc_col_a=$(head -1 $INPUT | tr ',' '\n' | nl |grep -w "$col_a" | tr -d " " | awk -F " " '{print $1}')
loc_col_b=$(head -1 $INPUT | tr ',' '\n' | nl |grep -w "$col_b" | tr -d " " | awk -F " " '{print $1}')
while IFS="," read -r pid accents
do
  for i in $(seq 1 5)
  do
    python main_cli_prune_accent.fit.py \
      -c cli_config/prune_v1/prune_pretrain.2.yaml \
      --data.init_args.key $pid \
      --trainer.default_root_dir output/prune_accent/$accents/$pid/prune=$prune
  done
done < <(cut -d "," -f${loc_col_a},${loc_col_b} $INPUT | tail -n +2)

