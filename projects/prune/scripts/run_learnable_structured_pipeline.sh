INPUT=preprocessed_data/VCTK-speaker-info.csv
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }

start_line=$1 # start line number of preprocessed_data/VCTK-speaker-info.csv
pipeline=$2

col_a="pID"
loc_col_a=$(head -1 $INPUT | tr ',' '\n' | nl |grep -w "$col_a" | tr -d " " | awk -F " " '{print $1}')

while IFS="," read -r pid
do
  dir="output/learnable_structured_pipeline/$pid/$pipeline/lightning_logs/"
  echo $dir
  source scripts/learnable_structured_pipeline.sh $pid $pipeline
done < <(cut -d "," -f${loc_col_a} $INPUT | tail -n +$start_line)
