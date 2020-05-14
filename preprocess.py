#! -*- coding:utf-8 -*-

import json
from tqdm import tqdm
import codecs

corpus_folder = "/home/cloudminds/Mywork/corpus/knowledge/ie2019"
output_folder = "%s/preprocess" % corpus_folder
train_data_file = '%s/train_data.json'% corpus_folder
dev_data_file = '%s/dev_data.json'% corpus_folder
schema_file = '%s/all_50_schemas'% corpus_folder
# id2predicate, predicate2id = json.load(open('%s/schema.json'%corpus_folder))

all_50_schemas = set()

with open(schema_file) as f:
    for l in tqdm(f):
        a = json.loads(l)
        all_50_schemas.add(a['predicate'])

id2predicate = {i+1:j for i,j in enumerate(all_50_schemas)} # 0表示终止类别
predicate2id = {j:i for i,j in id2predicate.items()}

with codecs.open('%s/all_schemas_me.json'%output_folder, 'w', encoding='utf-8') as f:
    json.dump([id2predicate, predicate2id], f, indent=4, ensure_ascii=False)


chars = {}
min_count = 2


train_data = []
def process_spo(spo_dict):
    if "2019" in corpus_folder:
        spo_map = (spo_dict['subject'], spo_dict['predicate'], spo_dict['object'])
    elif "2020" in corpus_folder:
        spo_map = (spo_dict['subject'], spo_dict['predicate'], list(spo_dict['object'].values())[0])
    else:
        print("error")
    return spo_map

with open(train_data_file) as f:
    for l in tqdm(f):
        a = json.loads(l)
        train_data.append(
            {
                'text': a['text'],
                'spo_list': [process_spo(spo_dict) for spo_dict in a['spo_list']]
            }
        )
        for c in a['text']:
            chars[c] = chars.get(c, 0) + 1

with codecs.open('%s/train_data_me.json'%output_folder, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)


dev_data = []
with open(dev_data_file) as f:
    for l in tqdm(f):
        a = json.loads(l)
        dev_data.append(
            {
                'text': a['text'],
                'spo_list': [process_spo(spo_dict) for spo_dict in a['spo_list']]
            }
        )
        for c in a['text']:
            chars[c] = chars.get(c, 0) + 1

with codecs.open('%s/dev_data_me.json' % output_folder, 'w', encoding='utf-8') as f:
    json.dump(dev_data, f, indent=4, ensure_ascii=False)


with codecs.open('%s/all_chars_me.json' % output_folder, 'w', encoding='utf-8') as f:
    chars = {i:j for i,j in chars.items() if j >= min_count}
    id2char = {i+2:j for i,j in enumerate(chars)} # padding: 0, unk: 1
    char2id = {j:i for i,j in id2char.items()}
    json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)

