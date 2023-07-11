export RANK_TABLE_FILE=/path/to/rank_table

export DEVICE_NUM=4
export RANK_SIZE=4

for ((i=0; i<${DEVICE_NUM};i++))
do
  export DEVICE_ID=$i
  export RANK_ID=$i
  pytest -s -v tests/test_auto_parallel_cpm_bee.py > train_log.log 2>&1 &

done