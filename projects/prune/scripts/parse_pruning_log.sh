for root_dir in `ls -d new_output/prune_accent/*/*/prune=*/lightning_logs/`
do
  for jj in `ls $root_dir/`
  do
    echo $root_dir
    echo $jj
    i="$root_dir/$jj"
    if [[ ! -d $i/fit ]]
    then
      echo "Failed: $i/fit"
      ls $i
      rm -rf $i
      echo

    elif [[ ! -d $i/fit/prune_log ]]
    then
      echo "Missed: $i/fit/prune_log"
      ls $i/fit
      cat $i/fit/pruning.log | wc -l
      rm -rf $i
      echo

    else
      echo $i
      # ls $i/fit/prune_log
      # echo "$((`ll $i/fit/prune_log | wc -l` - 1))"
      for j in $(seq 0 $((`ll $i/fit/prune_log | wc -l` - 1)))
      do
        if [[ `cat $i/fit/prune_log/epoch=$j.csv | wc -l` -lt 218 ]]
        then
          echo $i/fit/prune_log/epoch=$j.csv
          cat $i/fit/prune_log/epoch=$j.csv | wc -l
        fi
      done
      # cat $i/fit/pruning.log | wc -l
      # echo
    fi
    # if [[ -d $i/_prune_log ]]
    # then
    #   echo $i
    #   ls $i
    #   rm -rf $i/_prune_log
    #   ls $i
    #   cat $i/pruning.log | wc -l
    #   echo
    # fi
  done

  # Current status:
  # 2170 -> exist prune_log
  # < 2170 -> not exist prune_log

  # if [[ `cat $i/pruning.log | wc -l` -lt 2170 ]]
  # if [[ -d $i/prune_log ]]
  # then
  #   # echo $i
  #   # ls $i/prune_log
  #   for j in $(seq 0 9)
  #   do
  #     if [[ `cat $i/prune_log/epoch=$j.csv | wc -l` -lt 218 ]]
  #     then
  #       echo $i/prune_log/epoch=$j.csv
  #       cat $i/prune_log/epoch=$j.csv | wc -l
  #     fi
  #   done
    # cat $i/pruning.log | wc -l
  # elif [[ `cat $i/pruning.log | wc -l` -lt 2170 && -d $i/_prune_log ]]
  # then
  #   echo $i
  #   ls $i
  #   echo
  # fi

  # if [[ ! `ls $i` =~ "prune_log" && ! `ls $i` =~ "_prune_log" ]]
  # then
  #   ls $i
  #   # output: $i/_prune_log/epoch=*.csv
  #   python scripts/parse_pruning_log.py $i
  #   ls $i/_prune_log/*
  #   ln -s _prune_log $i/prune_log
  # fi
done
