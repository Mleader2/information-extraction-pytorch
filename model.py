import torch 
from torch import nn
from curLine_file import curLine


def seq_max_pool(x):
    """
    seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq = seq - (1 - mask) * 1e10
    return torch.max(seq, 1)
def seq_mean_pool(x):
    """
    seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做mean_pooling。
    """
    seq, mask = x
    seq = seq * mask
    mean = torch.sum(seq, 1)/torch.sum(mask, 1)
    return mean

def seq_and_vec(x):
    """
    seq是[None, seq_len, s_size]的格式，
    vec是[None, v_size]的格式，将vec重复seq_len次，拼到seq上，
    得到[None, seq_len, s_size+v_size]的向量。
    """
    seq , vec  = x
    vec = torch.unsqueeze(vec, 1)
    vec = torch.zeros_like(seq[:, :, :1]) + vec # same shape as seq
    return torch.cat([seq, vec], 2)

def seq_gather(x, device):
    """
    seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    seq, idxs = x
    batch_idxs = torch.arange(0,seq.size(0)).to(device)

    batch_idxs = torch.unsqueeze(batch_idxs,1)

    idxs = torch.cat([batch_idxs, idxs], 1)

    res = []
    for i in range(idxs.size(0)):
        vec = seq[idxs[i][0], idxs[i][1],:]
        res.append(torch.unsqueeze(vec,0))
    
    res = torch.cat(res)
    return res


class s_model(nn.Module):
    def __init__(self,word_dict_length,word_emb_size,lstm_hidden_size, args):
        super(s_model,self).__init__()
        self.device = args.device
        self.embeds = nn.Embedding(word_dict_length, word_emb_size).to(self.device)
        self.fc1_dropout = nn.Sequential(
            nn.Dropout(args.dropout).to(self.device),  # random drop the neuron
        ).to(self.device)

        self.lstm1 = nn.LSTM(
            input_size = word_emb_size,
            hidden_size = int(word_emb_size/2),
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        ).to(self.device)

        self.lstm2 = nn.LSTM(
            input_size = word_emb_size,
            hidden_size = int(word_emb_size/2),
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        ).to(self.device)

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=word_emb_size*2, #输入的深度
                out_channels=word_emb_size, #filter 的个数，输出的高度
                kernel_size = 3, #filter的长与宽
                stride=1,#每隔多少步跳一下
                padding=1,#周围围上一圈 if stride= 1, pading=(kernel_size-1)/2
            ).to(self.device),
            nn.ReLU().to(self.device),
        ).to(self.device)

        self.fc_ps1 = nn.Sequential(
            nn.Linear(word_emb_size,1),
        ).to(self.device)

        self.fc_ps2 = nn.Sequential(
            nn.Linear(word_emb_size,1),
        ).to(self.device)
        self.word_emb_size = word_emb_size

    def forward(self,t):
        if "cuda" in str(self.device):
            mask = torch.gt(torch.unsqueeze(t, 2), 0).type(torch.cuda.FloatTensor)  #(batch_size,sent_len,1)
        else:
            mask = torch.gt(torch.unsqueeze(t, 2), 0).type(torch.FloatTensor)
        mask.requires_grad = False
        outs = self.embeds(t)

        # t = outs
        t = self.fc1_dropout(outs)

        t = t.mul(mask)  # (batch_size,sent_len,char_size)

        t, (_, _) = self.lstm1(t,None)
        t, (_, _) = self.lstm2(t,None)

        # t_vector = seq_mean_pool([t, mask])  # 平均池化,没有最大池化的效果好
        t_vector, t_max_index = seq_max_pool([t, mask])  # 最大池化

        # t_vector = h_n.permute(0,2,1).reshape(-1, self.word_emb_size) # TODO
        #
        # if t_vector.shape[0] != 128:
        #     print(curLine(), mask.shape, "mask:", mask[0])
        #     ht = t * mask
        #     print(curLine(), ht.shape, "ht:", ht[0])
        #     print(curLine(), t_vector.shape, "t_vector:", t_vector[0])
        #     input(curLine())
        h = seq_and_vec([t, t_vector])

        layer = h.permute(0,2,1)
       
        layer = self.conv1(layer)

        layer = layer.permute(0,2,1)

        ps1 = self.fc_ps1(layer)
        ps2 = self.fc_ps2(layer)
        return [ps1.to(self.device),ps2.to(self.device),t.to(self.device),h.to(self.device), mask.to(self.device)]

class po_model(nn.Module):
    def __init__(self, word_emb_size, num_classes, args):
        super(po_model, self).__init__()
        self.device = args.device
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=word_emb_size*4, # 输入的深度
                out_channels=word_emb_size, # filter 的个数，输出的高度
                kernel_size = 3, # filter的长与宽
                stride=1, # 每隔多少步跳一下
                padding=1, # 周围围上一圈 if stride= 1, pading=(kernel_size-1)/2
            ).to(args.device), # cuda(),
            nn.ReLU().to(args.device), # cuda(),
        ).to(args.device)  # cuda()

        self.fc_ps1 = nn.Sequential(
            nn.Linear(word_emb_size,num_classes+1).to(args.device),
        ).to(args.device)

        self.fc_ps2 = nn.Sequential(
            nn.Linear(word_emb_size,num_classes+1).to(args.device),
        ).to(args.device)
    
    def forward(self,t, h, k1, k2):
        # K1,K2  随机某个subject的位置(start,end)  维度较小
        k1 = seq_gather([t,k1], device=self.device)
        k2 = seq_gather([t,k2], device=self.device)

        k = torch.cat([k1,k2],1)

        h = seq_and_vec([h,k])
        h = h.permute(0,2,1)
        h = self.conv1(h)
        h = h.permute(0,2,1)

        po1 = self.fc_ps1(h)
        po2 = self.fc_ps2(h)

        return [po1.to(self.device),po2.to(self.device)]