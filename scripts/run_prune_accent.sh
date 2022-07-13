INPUT=preprocessed_data/VCTK-speaker-info.csv
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }

prune=$1
max_epochs=$2
start_line=$3 # start line number of preprocessed_data/VCTK-speaker-info.csv
runs=$4
col_a="pID"
col_b="ACCENTS"
loc_col_a=$(head -1 $INPUT | tr ',' '\n' | nl |grep -w "$col_a" | tr -d " " | awk -F " " '{print $1}')
loc_col_b=$(head -1 $INPUT | tr ',' '\n' | nl |grep -w "$col_b" | tr -d " " | awk -F " " '{print $1}')
while IFS="," read -r pid accents
do
  for i in $(seq 1 $runs)
  do
    source scripts/prune_accent.sh $accents $pid $prune $max_epochs
  done
done < <(cut -d "," -f${loc_col_a},${loc_col_b} $INPUT | tail -n +$start_line)

