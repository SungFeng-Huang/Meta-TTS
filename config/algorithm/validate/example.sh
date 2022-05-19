for i in `seq 9 10 99`; do
  python main.py -a config/algorithm/validate/pretrain_LibriTTS.2.yaml -c epoch=${i}-step=${i}999.ckpt;
done
