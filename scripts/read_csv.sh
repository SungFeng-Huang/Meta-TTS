INPUT=preprocessed_data/VCTK-speaker-info.csv
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }

col_a="pID"
col_b="ACCENTS"
loc_col_a=$(head -1 $INPUT | tr ',' '\n' | nl |grep -w "$col_a" | tr -d " " | awk -F " " '{print $1}')
loc_col_b=$(head -1 $INPUT | tr ',' '\n' | nl |grep -w "$col_b" | tr -d " " | awk -F " " '{print $1}')
while IFS="," read -r rec1 rec2
do
  echo "$col_a : $rec1"
  echo "$col_b : $rec2"
  echo ""
done < <(cut -d "," -f${loc_col_a},${loc_col_b} $INPUT | tail -n +2)
