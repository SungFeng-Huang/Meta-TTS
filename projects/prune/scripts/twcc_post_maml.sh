TWCC_CLI_CMD=/home/r06942045/.local/bin/twccli

echo "1. Creating CCS" #建立開發型容器
$TWCC_CLI_CMD mk ccs \
  -n "maml_test" \
  -itype "PyTorch" \
  -img "pytorch-22.08-py3:latest" \
  -gpu 1 \
  -wait -json > ccs_res.log

CCS_ID=$(cat ccs_res.log | jq '.id')
echo "2. CCS ID:" $CCS_ID #開發型容器 ID

echo "3. Install htop and tmux" #確認 GPU 狀態
ssh -t -o "StrictHostKeyChecking=no" `$TWCC_CLI_CMD ls ccs -gssh -s $CCS_ID` \
  "/bin/bash --login RUN_THIS_AFTER_NEW_CONTAINER.sh;"

echo "4. RUN job" #執行運算程式
ssh -t -o "StrictHostKeyChecking=no" `$TWCC_CLI_CMD ls ccs -gssh -s $CCS_ID` \
  "cd Meta-TTS; pip install -r requirements.txt -U; PYTHONPATH=. python projects/meta-tts/main_cli.py fit -c cli_config/prune_v1/post_maml/main.yaml --print_config;"
#可依據您的程式，修改 "cd gpu-burn;/bin/bash --login -c './gpu_burn 150'"

echo "5. GC GPU" #刪除開發型容器
$TWCC_CLI_CMD rm ccs -f -s $CCS_ID

echo "6. Checking CCS" #檢視容器狀態
$TWCC_CLI_CMD ls ccs
