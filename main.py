#! -*- coding:utf-8 -*-
import json
import numpy as np
from random import choice
from tqdm import tqdm
import time
import argparse
import torch
import torch.utils.data as Data
torch.backends.cudnn.benchmark = True

import model
from curLine_file import curLine


def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--corpus_folder', type=str, default="/home/cloudminds/Mywork/corpus/knowledge",
                        help="corpus folder path")
    parser.add_argument('--output_dir', type=str, default="/home/cloudminds/Mywork/corpus/knowledge/models",
                        help="save models")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--digit2zero', action="store_true", default=True,
                        help="convert the number to 0, make it true is better")

    ##optimize hyperparameter
    # parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--learning_rate', type=float, default=0.005)  ##only for adam now
    parser.add_argument('--lr_decay', type=float, default=0.98)  # TODO change decay method
    parser.add_argument('--grad_clip', type=int, default=5.0, help="default grad clip is 5 (works well)")

    parser.add_argument('--batch_size', type=int, default=64,
                        help="default batch size is 10 (works well)")
    parser.add_argument('--num_epochs', type=int, default=10, help="Usually we set to 10.")
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")

    ##model hyperparameter
    parser.add_argument('--hidden_dim', type=int, default=64, help="hidden size")
    parser.add_argument('--char_dim', type=int, default=128)
    parser.add_argument('--subject_ratio', type=float, default=2.5, help="ratio of subject in total loass")

    parser.add_argument('--dropout', type=float, default=0.2, help="dropout for embedding")
    parser.add_argument('--l2', type=float, default=1e-7)
    args = parser.parse_args()
    return args


parser = argparse.ArgumentParser(description="information extraction")
args = parse_arguments(parser)
args.device = torch.device(args.device)

print(curLine(), "device:", args.device, "subject_ratio=%f, corpus_folder:%s"
      % (args.subject_ratio, args.corpus_folder))


def get_now_time():
    a = time.time()
    return time.ctime(a)


def seq_padding(X):
    L = [len(x) for x in X]
    ML = max(L)
    return [x + [0] * (ML - len(x)) for x in X]


def seq_padding_vec(X):
    L = [len(x) for x in X]
    ML = max(L)
    return [x + [[1, 0]] * (ML - len(x)) for x in X]


train_data = json.load(open('%s/train_data_me.json' % args.corpus_folder))
dev_data = json.load(open('%s/dev_data_me.json' % args.corpus_folder))
id2predicate, predicate2id = json.load(open('%s/all_schemas_me.json' % args.corpus_folder))
id2predicate = {int(i): j for i, j in id2predicate.items()}
id2char, char2id = json.load(open('%s/all_chars_me.json' % args.corpus_folder))
num_classes = len(id2predicate)
print(curLine(), "num_classes=", num_classes)

class data_generator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def pro_res(self):
        idxs = list(range(len(self.data)))
        np.random.shuffle(idxs)
        T, S1, S2, K1, K2, O1, O2, = [], [], [], [], [], [], []
        # T 对文本的序列化
        # S1,S2  所有subject的位置　one_hot编码
        # K1,K2  随机某个subject的位置  维度较小
        # O1,O2  各object对应的relation
        for i in idxs:
            d = self.data[i]
            text = d['text']
            items = {}
            for sp in d['spo_list']:
                subjectid = text.find(sp[0])
                objectid = text.find(sp[2])
                if subjectid != -1 and objectid != -1:
                    key = (subjectid, subjectid + len(sp[0]))  # start location and end location
                    if key not in items:
                        items[key] = []
                    items[key].append((objectid, objectid + len(sp[2]), predicate2id[sp[1]]))  # object and relation
            if items:
                T.append([char2id.get(c, 1) for c in text])  # 1是unk，0是padding
                # s1, s2 = [[1,0]] * len(text), [[1,0]] * len(text)
                s1, s2 = [0] * len(text), [0] * len(text)  # subject_location
                for subject_location in items:
                    s1[subject_location[0]] = 1
                    s2[subject_location[1] - 1] = 1
                random_subject = choice(list(items.keys()))  # 随机选择一个subject
                o1, o2 = [0] * len(text), [0] * len(text)  # 0是unk类
                for value in items[random_subject]:
                    o1[value[0]] = value[2]
                    o2[value[1] - 1] = value[2]
                S1.append(s1)
                S2.append(s2)
                K1.append([random_subject[0]])
                K2.append([random_subject[1] - 1])
                O1.append(o1)
                O2.append(o2)
        T = np.array(seq_padding(T))
        S1 = np.array(seq_padding(S1))
        S2 = np.array(seq_padding(S2))
        O1 = np.array(seq_padding(O1))
        O2 = np.array(seq_padding(O2))
        K1, K2 = np.array(K1), np.array(K2)
        return [T, S1, S2, K1, K2, O1, O2]


class myDataset(Data.Dataset):
    """
        初始化数据
    """

    def __init__(self, _T, _S1, _S2, _K1, _K2, _O1, _O2):
        # xy = np.loadtxt('../dataSet/diabetes.csv.gz', delimiter=',', dtype=np.float32) # 使用numpy读取数据
        self.x_data = _T
        self.y1_data = _S1
        self.y2_data = _S2
        self.k1_data = _K1
        self.k2_data = _K2
        self.o1_data = _O1
        self.o2_data = _O2
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y1_data[index], self.y2_data[index], self.k1_data[index], self.k2_data[index], \
               self.o1_data[index], self.o2_data[index]

    def __len__(self):
        return self.len


def collate_fn(data):
    t = np.array([item[0] for item in data], np.int32)
    s1 = np.array([item[1] for item in data], np.int32)
    s2 = np.array([item[2] for item in data], np.int32)
    k1 = np.array([item[3] for item in data], np.int32)

    k2 = np.array([item[4] for item in data], np.int32)
    o1 = np.array([item[5] for item in data], np.int32)
    o2 = np.array([item[6] for item in data], np.int32)
    return {
        'T': torch.LongTensor(t),  # targets_i
        'S1': torch.FloatTensor(s1),
        'S2': torch.FloatTensor(s2),
        'K1': torch.LongTensor(k1),
        'K2': torch.LongTensor(k2),
        'O1': torch.LongTensor(o1),
        'O2': torch.LongTensor(o2),
    }


dg = data_generator(train_data)
T, S1, S2, K1, K2, O1, O2 = dg.pro_res()


torch_dataset = myDataset(T, S1, S2, K1, K2, O1, O2)
loader = Data.DataLoader(
    dataset=torch_dataset,  # torch TensorDataset format
    batch_size=args.batch_size,  # mini batch size
    shuffle=True,  # random shuffle for training
    num_workers=8,
    collate_fn=collate_fn,  # subprocesses for loading data
)


s_m = model.s_model(len(char2id) + 2, args.char_dim, args.hidden_dim, args).to(args.device)  # .cuda()
po_m = model.po_model(args.char_dim, num_classes=num_classes, args=args).to(args.device)  # .cuda()
params = list(s_m.parameters())

params += list(po_m.parameters())
optimizer = torch.optim.Adam(params, lr=args.learning_rate)

loss = torch.nn.CrossEntropyLoss().to(args.device)
b_loss = torch.nn.BCEWithLogitsLoss().to(args.device)


def extract_items(text_in):
    #　　验证测试时，batch_size=1
    R = []
    _s = [char2id.get(c, 1) for c in text_in]
    _s = np.array([_s])
    _k1, _k2, t, h, mask = s_m(torch.LongTensor(_s).to(args.device))
    _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
    # _kk1s = []
    for i, _kk1 in enumerate(_k1):
        if _kk1 > 0.5:
            _subject = ''
            for j, _kk2 in enumerate(_k2[i:]):
                if _kk2 > 0.5:
                    _subject = text_in[i: i + j + 1]
                    break
            if _subject:
                _k1, _k2 = torch.LongTensor([[i]]).to(args.device), torch.LongTensor([[i + j]]).to(args.device)  # np.array([i]), np.array([i+j])
                _o1, _o2 = po_m(t, h, _k1, _k2) # 对于某个subject(k1,k2表示), 识别各个可能的关系o(内容表示关系，索引表示object的位置）

                _o1, _o2 = _o1.cpu().data.numpy(), _o2.cpu().data.numpy()

                _o1, _o2 = np.argmax(_o1[0], 1), np.argmax(_o2[0], 1)
                # print(curLine(), "_o1:", _o1, ",_o2:", _o2)
                for i, _oo1 in enumerate(_o1):
                    if _oo1 > 0:
                        for j, _oo2 in enumerate(_o2[i:]):
                            if _oo2 == _oo1:
                                _object = text_in[i: i + j + 1]
                                _predicate = id2predicate[_oo1]
                                R.append((_subject, _predicate, _object))
                                # print(curLine(), _subject, _predicate, _object)
                                # input(curLine())
                                break
        # _kk1s.append(_kk1.data.cpu().numpy()) # _kk1s: list of float
    # _kk1s = np.array(_kk1s)
    return list(set(R))  # R:三元组的列表，这一步骤是为了去重


def evaluate():
    A, B, C = 1e-10, 1e-10, 1e-10
    cnt = 0
    for d in tqdm(iter(dev_data)):
        R = set(extract_items(d['text']))
        T = set([tuple(i) for i in d['spo_list']])
        A += len(R & T)
        B += len(R)
        C += len(T)
        # if cnt % 1000 == 0:
        #     print('iter: %d f1: %.4f, precision: %.4f, recall: %.4f\n' % (cnt, 2 * A / (B + C), A / B, A / C))
        cnt += 1
    return 2 * A / (B + C), A / B, A / C

def train_step(loader_res):
    t_s = loader_res["T"].to(args.device)  # cuda()
    k1 = loader_res["K1"].to(args.device)  # cuda()
    k2 = loader_res["K2"].to(args.device)  # cuda()
    s1 = loader_res["S1"].to(args.device)  # cuda()
    s2 = loader_res["S2"].to(args.device)  # cuda()
    o1 = loader_res["O1"].to(args.device)  # cuda()
    o2 = loader_res["O2"].to(args.device)  # cuda()

    ps_1, ps_2, t, t_max, mask = s_m(t_s)
    po_1, po_2 = po_m(t, t_max, k1, k2)

    s1 = torch.unsqueeze(s1, 2)
    s2 = torch.unsqueeze(s2, 2)

    s1_loss = b_loss(ps_1, s1)
    s1_loss = torch.sum(s1_loss.mul(mask)) / torch.sum(mask)
    s2_loss = b_loss(ps_2, s2)
    s2_loss = torch.sum(s2_loss.mul(mask)) / torch.sum(mask)

    po_1 = po_1.permute(0, 2, 1)
    po_2 = po_2.permute(0, 2, 1)

    o1_loss = loss(po_1, o1)
    o1_loss = torch.sum(o1_loss.mul(mask[:, :, 0])) / torch.sum(mask)
    o2_loss = loss(po_2, o2)
    o2_loss = torch.sum(o2_loss.mul(mask[:, :, 0])) / torch.sum(mask)

    loss_sum = args.subject_ratio * (s1_loss + s2_loss) + (o1_loss + o2_loss)  # TODO
    optimizer.zero_grad()
    loss_sum.backward()
    optimizer.step()
    return loss_sum


def update_lr(optimizer, coefficient):
    previous = optimizer.param_groups[0]['lr']
    print(curLine(), "learning rate from %f drop to %f" % (previous, previous * coefficient))
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * coefficient
    return param_group['lr']


if __name__ == '__main__':
    best_f1 = 0
    best_epoch = 0
    start_time = time.time()
    for epoch_id in range(1, args.num_epochs + 1):
        for step, loader_res in tqdm(iter(enumerate(loader))):
            loss_sum = train_step(loader_res=loader_res)
        f1, precision, recall = evaluate()
        cost_time = (time.time() - start_time) / 3600.0
        learning_rate = update_lr(optimizer=optimizer, coefficient=args.lr_decay)
        print("%s epoch:%d/%d, learning_rate=%f, loss=%f, f1=%f."
              % (curLine(), epoch_id, args.num_epochs, learning_rate, loss_sum.data.item(), f1))
        print(curLine(), 'f1: %.4f, precision: %.4f, recall: %.4f, bestf1: %.4f, bestepoch: %d, cost %fh.' % (
            f1, precision, recall, best_f1, best_epoch, cost_time))
        if f1 >= best_f1:
            best_f1 = f1
            best_epoch = epoch_id
            s_model_path = "%s/s_epoch%d_f1%f.pkl" % (args.output_dir, epoch_id, f1)
            torch.save(s_m, s_model_path)
            po_model_path = "%s/po_epoch%d_f1%f.pkl" % (args.output_dir, epoch_id, f1)
            torch.save(po_m, po_model_path)
            print("%s have saved model to %s at epoch=%d.\n" % (curLine(), args.output_dir, epoch_id))
            # print("%s f1=%f, precision=%f, recall=%f\n" % (curLine(), f1, precision, recall))
    cost_time = (time.time()-start_time)/3600.0
    print(curLine(), "finish train %d epoches, cost %f hours." % (args.num_epochs, cost_time))
