for i in `ls -d output/prune_accent/*/*/prune=*/lightning_logs/version_*/fit/`
do
  # if [[ -d $i/_prune_log ]]
  # then
  #   echo $i
  #   ls $i
  #   rm -rf $i/_prune_log
  #   ls $i
  #   cat $i/pruning.log | wc -l
  #   echo
  # fi

  # Current status:
  # 2170 -> exist prune_log
  # < 2170 -> not exist prune_log

  # if [[ `cat $i/pruning.log | wc -l` -lt 2170 ]]
  if [[ -d $i/prune_log ]]
  then
    # echo $i
    # ls $i/prune_log
    for j in $(seq 0 9)
    do
      if [[ `cat $i/prune_log/epoch=$j.csv | wc -l` -lt 218 ]]
      then
        echo $i/prune_log/epoch=$j.csv
        cat $i/prune_log/epoch=$j.csv | wc -l
      fi
    done
    # cat $i/pruning.log | wc -l
  # elif [[ `cat $i/pruning.log | wc -l` -lt 2170 && -d $i/_prune_log ]]
  # then
  #   echo $i
  #   ls $i
  #   echo
  fi

  # if [[ ! `ls $i` =~ "prune_log" && ! `ls $i` =~ "_prune_log" ]]
  # then
  #   ls $i
  #   # output: $i/_prune_log/epoch=*.csv
  #   python scripts/parse_pruning_log.py $i
  #   ls $i/_prune_log/*
  #   ln -s _prune_log $i/prune_log
  # fi
done
