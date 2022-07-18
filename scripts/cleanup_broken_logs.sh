for root_dir in `ls -d new_output/prune_accent/*/*/prune=*/lightning_logs/`
do
  # echo $j
  # ls -l $j
  # k=`ls -l $j | wc -l`
  # while [[ k -lt 5 ]]
  # do
  #   echo $k
  #   k=$(($k+1))
  # done
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
      for j in $(seq 0 $((`ll $i/fit/prune_log | wc -l` - 1)))
      do
        if [[ `cat $i/fit/prune_log/epoch=$j.csv | wc -l` -lt 218 ]]
        then
          echo $i/fit/prune_log/epoch=$j.csv
          cat $i/fit/prune_log/epoch=$j.csv | wc -l
        fi
      done
    fi
  done
done
