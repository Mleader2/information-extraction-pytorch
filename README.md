# kg-baseline-pytorch
2019百度的关系抽取比赛，使用Pytorch实现LSTM模型在dev集可达到F1=74.7%，联合关系抽取（Joint Relation Extraction）.
如果用BERT,RoBERT或Ernie,分数会更高。

## 步骤
1.语料预处理
preprocess.py
2.训练2个模型
bash train.sh wzk   #　请自行理解脚本，并根据自己情况修改

## 模型
先用一个模型识别subject,再用另一个模型识别object,relation

## 数据
用同样的模型支持2019和2020年的百度的关系抽取比赛；
由于２者的语料存在一定差异，在预处理阶段（preprocess.py的process_spo函数中）会根据corpus_folder选择不同的预处理方式。

## 结果
在2020年的百度的关系抽取比赛的测试语料上，5个epoch后F1即高于73％，最高能到74.68%。

## 环境
Python 3.5+     python3.5~3.7版本下亲测可用，其他版本不清楚
Pytorch 1.3+  Pyorch1.x应该都可以
tqdm
both cpu or gpu is ok!