#!/usr/bin/env bash
# 信息抽取 ie
# 房山　pyenv activate python363tf111
# 成都　pyenv activate python352tf114
# usage:  bash train.sh hostname

start_tm=`date +%s%N`;
export HOST_NAME=$1
if [[ "wzk" == "$HOST_NAME" ]]
then
  # set gpu id to use
  export CUDA_VISIBLE_DEVICES=0
  export device="cuda:0"
else
  # not use gpu
  export CUDA_VISIBLE_DEVICES=""
  export device="cpu"
fi

export corpus_folder="/home/${HOST_NAME}/Mywork/corpus/knowledge/ie2019/preprocess"
export output_dir="/home/${HOST_NAME}/Mywork/corpus/knowledge/models"
export log_file="log.txt"

echo "save log to" ${log_file}
# not use nohup
#python -u main.py --corpus_folder=${corpus_folder} --output_dir=${output_dir} --device=${device} \
#      --hidden_dim=100 --batch_size=128 --dropout=0.2 --num_epochs=30 \
#      --learning_rate=0.005 --lr_decay=0.95 -subject_ratio=2.0

# use nohup
nohup python -u main.py --corpus_folder=${corpus_folder} --output_dir=${output_dir} --device=${device} \
      --hidden_dim=100 --batch_size=128 --dropout=0.2 --num_epochs=30 \
      --learning_rate=0.005 --lr_decay=0.98 --subject_ratio=2.5 > ${log_file}  2>&1 &
############# append pid to file
echo $! >> save_pid.txt
tail -f ${log_file}
echo "save pid" $! "to save_pid.txt"

end_tm=`date +%s%N`;
use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000 /3600}'`
echo "cost time" $use_tm "h"