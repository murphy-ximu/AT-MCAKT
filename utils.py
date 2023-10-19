import time

import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()

                # p_adv = torch.FloatTensor(epsilon * _l2_normalize_adv(param.grad.data))
                # param.data.add_(p_adv.to(param.device))
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

import torch
class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='emb.', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()

                p_adv = torch.FloatTensor(epsilon * _l2_normalize_adv(param.grad.data))
                param.data.add_(p_adv.to(param.device))
                param.data = self.project(name, param.data, epsilon)
                # norm = torch.norm(param.grad)
                # if norm != 0 and not torch.isnan(norm):
                #     r_at = alpha * param.grad / norm
                #     param.data.add_(r_at)
                #     param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return param_data + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

def _l2_normalize_adv_at(d):
    if isinstance(d, Variable):
        d = d.data.cpu().numpy()
    elif isinstance(d, torch.FloatTensor) or isinstance(d, torch.cuda.FloatTensor):
        d = d.cpu().numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(0, 1))).reshape((1, 1)) + 1e-16)
    return torch.from_numpy(d)

def _l2_normalize_adv(d):
    if isinstance(d, Variable):
        d = d.data.cpu().numpy()
    elif isinstance(d, torch.FloatTensor) or isinstance(d, torch.cuda.FloatTensor):
        d = d.cpu().numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1,1, 1)) + 1e-16)
    return torch.from_numpy(d)


def randomSamples(all_outs, samp, samp_pad,sampleIndex=None):
    # all_outs: seq_len*b_s  *  2hdim
    # sampleIndex: 样本下标
    # samp_pad: bs seq
    # return all_outs[0:100,:]
    device = all_outs.device
    bs = samp_pad.shape[0]
    size = all_outs.shape[0]
    len = int(size / bs)
    if sampleIndex is not None:
        samples = torch.index_select(all_outs, 0, sampleIndex).to(device)
        return samples
    pad_weight = np.array(samp_pad.reshape(-1).cpu())
    p = pad_weight/pad_weight.sum()
    indexlist = np.random.choice(torch.arange(0, size), size=int(size * samp), replace=True, p=p)
    sampleIndex = torch.tensor(np.array(indexlist)).to(device)  # 随机采样
    # sampleIndex = torch.arange(0,size,step=len).to(device)
    # sampleIndex = torch.cat([sampleIndex+i for i in range(int(len*samp))], dim=0) # 120

    samples = torch.index_select(all_outs, 0, sampleIndex).to(device)
    return samples,sampleIndex

# https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


def get_now_time():
    """
    获取当前日期时间
    :return:当前日期时间
    """
    now =  time.localtime()
    now_time = time.strftime("%Y%m%d%H%M", now)
    # now_time = time.strftime("%Y-%m-%d ", now)
    return now_time


def load_checkpoint(model, checkpoint):
    if checkpoint != 'No':
        print("loading checkpoint...")
        model_dict = model.state_dict()
        modelCheckpoint = torch.load(checkpoint)
        pretrained_dict = modelCheckpoint['state_dict']
        for k, v in pretrained_dict.items():
            if k not in model_dict.keys():
                print(k)
        # 过滤操作
        new_dict = {k: v for k, v in pretrained_dict.items()
                    if k in model_dict.keys()}
        model_dict.update(new_dict)
        # 打印出来，更新了多少的参数
        print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        model.load_state_dict(model_dict)
        print("loaded finished!")
    else:
        print('No checkpoint is included')
    return model