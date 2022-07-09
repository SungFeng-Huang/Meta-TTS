for i in `ls -d output/prune_accent/*/*/prune=*/lightning_logs/version_*/fit/`
do
  echo $i
  if [[ ! `ls $i` =~ "prune_log" && ! `ls $i` =~ "_prune_log" ]]
  then
    ls $i
    # output: $i/_prune_log/epoch=*.csv
    python scripts/parse_pruning_log.py $i
    ls $i/_prune_log/*
    ln -s _prune_log $i/prune_log
  fi
done
