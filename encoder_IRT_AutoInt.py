import math
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch import optim
from torch.autograd import Variable
from torch.distributed import get_rank

from RAdam import *
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

from config import Config
from data_analysis import get_new_exe_data
from dataset import get_dataloaders, get_test_dataloaders
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from time import *
from deepctr_torch.layers.interaction import InteractingLayer
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding


class Encoder_IRT_Modle(pl.LightningModule):
    def __init__(self, config):
        super(Encoder_IRT_Modle, self).__init__()
        self.LR = config["LR"]
        self.DECAY = config["DECAY"]
        self.EMBED_DIMS = config["EMBED_DIMS"]
        self.H_DIMS = config["H_DIMS"]
        self.ENC_HEADS = config["ENC_HEADS"]
        self.NUM_ENCODER = config["NUM_ENCODER"]
        self.BATCH_SIZE = config["BATCH_SIZE"]
        self.MA_DROP_OUT = config["MA_DROP_OUT"]
        self.PL_DROP_OUT = config["PL_DROP_OUT"]
        self.FFN_DROP_OUT = config["FFN_DROP_OUT"]
        self.LAYER_NUM = config["LAYER_NUM"]

        self.loss = nn.BCELoss()
        # self.loss = BCFocalLoss()
        self.encoder_layer = StackedNMultiHeadAttention(n_stacks=self.NUM_ENCODER,
                                                        n_dims=Config.CON * self.EMBED_DIMS,
                                                        n_heads=self.ENC_HEADS,
                                                        seq_len=Config.MAX_SEQ,
                                                        n_multihead=1,
                                                        ma_dropout=self.MA_DROP_OUT,
                                                        ffn_dropout=self.FFN_DROP_OUT)
        self.encoder_embedding = EncoderEmbedding(n_exercises=Config.TOTAL_EXE,
                                                  n_categories=Config.TOTAL_CAT,
                                                  n_responses=Config.RESPONSES_NUM,
                                                  n_dims=self.EMBED_DIMS, seq_len=Config.MAX_SEQ,
                                                  max_cnum=Config.MAX_CATS_PER,
                                                  bs=self.BATCH_SIZE,
                                                  att_layer_num=self.LAYER_NUM)
        self.elapsed_time = nn.Linear(1, Config.CON * self.EMBED_DIMS)  # 时间信息
        # self.lag_time_s = nn.Linear(1, Config.CON * self.EMBED_DIMS)  # 时间信息
        # self.lag_time_d = nn.Linear(1, Config.CON * self.EMBED_DIMS)  # 时间信息
        # self.lag_time_m = nn.Linear(1, Config.CON * self.EMBED_DIMS)  # 时间信息
        self.pos_embed = PositionalBias(max_seq=Config.MAX_SEQ, embed_dim=Config.CON * self.EMBED_DIMS,
                                        num_heads=self.ENC_HEADS,
                                        bidirectional=False, num_buckets=32,
                                        max_distance=Config.MAX_SEQ)
        # self.elapsed_time = nn.Linear(1, Config.EMBED_DIMS)  # 时间信息
        # self.lag_time_s = nn.Linear(1, Config.EMBED_DIMS)  # 时间信息
        # self.lag_time_d = nn.Linear(1, Config.EMBED_DIMS)  # 时间信息
        # self.lag_time_m = nn.Linear(1, Config.EMBED_DIMS)  # 时间信息
        # self.learning_layer = LearningLayer(n_dim=2 * Config.EMBED_DIMS,
        #                                     h_dim=2 * Config.EMBED_DIMS)
        # self.decoder_embedding = DecoderEmbedding(
        #     n_responses=Config.RESPONSES_NUM, n_dims=self.EMBED_DIMS, seq_len=Config.MAX_SEQ)  # responses: -1 0 1
        # self.decoder_layer = StackedNMultiHeadAttention(n_stacks=self.NUM_ENCODER,
        #                                                 n_dims=Config.CON*self.EMBED_DIMS,
        #                                                 n_heads=self.ENC_HEADS,
        #                                                 seq_len=Config.MAX_SEQ,
        #                                                 n_multihead=2,
        #                                                 ma_dropout=self.MA_DROP_OUT,
        #                                                 ffn_dropout=self.FFN_DROP_OUT)

        if Config.Variants=="R2PL":
            self.pred_mlp = nn.Sequential(nn.Linear(2 * self.EMBED_DIMS, self.H_DIMS), nn.Dropout(0.2),
                                         nn.Linear(self.H_DIMS, 1),
                                        nn.Sigmoid())

        else:
            self.predict_layer = PredictLayer(n_categories=Config.TOTAL_CAT,
                                          # n_fc= Config.EMBED_DIMS,)
                                          n_fc=Config.CON * self.EMBED_DIMS,
                                          h_dim=self.H_DIMS,
                                          dropout=self.PL_DROP_OUT)
        # self.pred_mlp = nn.Sequential(nn.Linear(2 * self.EMBED_DIMS, self.H_DIMS), nn.Dropout(0.2),
        #                               nn.Linear(self.H_DIMS, 1),
        #                               nn.Sigmoid())
        # self.fc = nn.Linear(2 * Config.EMBED_DIMS, 1)

        self.train_auc = []
        self.train_acc = []
        self.val_auc = []
        self.val_acc = []
        self.test_auc = []
        self.test_acc = []
        self.epoch_num = 0

    def forward(self, x, y, labels):
        # y_padd
        # "input_ids": exe_ids
        # "input_rtime": input_rtime.astype(np.int)
        # "input_cat": exe_cat
        enc, enc_wo_r, cat_params = self.encoder_embedding(
            exercises=x["input_ids"], categories=x['input_cat'],
            cate_num=x["input_cnum"], exe_diff=x["input_diff"],
            lt_s=x["input_lag_time_s"], lt_m=x["input_lag_time_m"], lt_d=x["input_lag_time_d"],
            responses=labels)
        # elapsed_time = x["input_rtime"].unsqueeze(-1).float()
        # ela_time = self.elapsed_time(elapsed_time)
        # enc = enc + ela_time
        # this encoder
        # if torch.any(torch.isnan(enc)):
        #     print("x:", x)
        #     print("enc:", enc)
        #     print("cat_params:", cat_params)
        #     input("PAUSE")
        pos_bias_embed = self.pos_embed(torch.arange(Config.MAX_SEQ).unsqueeze(0).to(Config.device)) if Config.Pos_Bias else None
        encoder_output = self.encoder_layer(input_k=enc,
                                            input_q=enc_wo_r,
                                            input_v=enc,
                                            pos_embed=pos_bias_embed,
                                            ltime=x["input_rtime"]
                                            )
        elapsed_time = x["input_rtime"].unsqueeze(-1).float()
        ela_time = self.elapsed_time(elapsed_time)
        encoder_output = encoder_output + ela_time
        # fully connected layer
        # lstm_out = self.learning_layer(x=encoder_output)
        # this is decoder
        # dec = self.decoder_embedding(responses=y)

        # lstm_out = lstm_out + ela_time  # decodeXr_embedding
        # decoder_output = self.decoder_layer(input_k=dec,
        #                                     input_q=dec,
        #                                     input_v=dec,
        #                                     encoder_output=encoder_output,
        #                                     break_layer=1)
        # fully connected layer
        # lt_s = x["input_lag_time_s"].unsqueeze(-1).float()
        # lt_s = self.lag_time_s(lt_s)
        # lt_m = x["input_lag_time_m"].unsqueeze(-1).float()
        # lt_m = self.lag_time_m(lt_m)
        # lt_d = x["input_lag_time_d"].unsqueeze(-1).float()
        # lt_d = self.lag_time_d(lt_d)
        #
        # lag_time = lt_s + lt_m + lt_d
        if Config.Variants == "R2PL":
            out = self.pred_mlp(encoder_output)
        else:
            out = self.predict_layer(x=encoder_output, cat_params=cat_params, categories=x['input_cat'],
                                     lt_s=x["input_lag_time_s"].unsqueeze(-1).float(),
                                     lt_m=x["input_lag_time_m"].unsqueeze(-1).float(),
                                     lt_d=x["input_lag_time_d"].unsqueeze(-1).float())
        # out = self.pred_mlp(encoder_output)
        # out =self.fc(encoder_output)
        return out.squeeze()

    def configure_optimizers(self):
        optimizer = RAdam(self.parameters(),
                               lr=self.LR,
                               weight_decay=self.DECAY,
                               )
        return optimizer
        # return torch.optim.AdamW(self.parameters(), lr=self.LR)

    def training_step(self, batch, batch_ids):
        input, ans, labels = batch
        # input, labels = batch
        target_mask = (input["input_ids"] != 0)
        out = self(input, ans, labels)
        # out = self(input, labels)
        if out.shape != labels.shape:
            labels = labels.squeeze(-1)
        # out = torch.sigmoid(out)
        # loss = self.loss(out.float(), labels.float())
        out = torch.masked_select(out, target_mask)

        labels = torch.masked_select(labels, target_mask)
        loss = self.loss(out.float(), labels.float())
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return {"loss": loss, "outs": out, "labels": labels}

    def training_epoch_end(self, training_ouput):
        out = np.concatenate([i["outs"].cpu().detach().numpy()
                              for i in training_ouput]).reshape(-1)
        labels = np.concatenate([i["labels"].cpu().detach().numpy()
                                 for i in training_ouput]).reshape(-1)
        auc = roc_auc_score(labels, out)
        acc = accuracy_score(labels, out.round())
        self.train_auc.append(auc)
        self.train_acc.append(acc)
        self.print("")
        self.print("train auc", auc)
        self.log("train_auc", auc)
        self.print("train acc", acc)
        self.log("train_acc", acc)
        self.epoch_num += 1
        # if self.epoch_num == Config.EPOCH_NUM:
        #     self.print("[train auc]", self.train_auc)
        #     self.print("[train acc]", self.train_acc)

    def validation_step(self, batch, batch_ids):
        _input, ans, labels = batch

        # input, labels = batch
        # print("==========input============")
        # print(_input['input_ids'][0,:])
        # print(_input['input_cat'][0,:,:])
        # input("PAUSE")
        # print("==========labels============")
        # print(labels[0,:])
        target_mask = (_input["input_ids"] != 0)
        out = self(_input, ans, labels)

        # out = self(input, labels)
        # out = torch.sigmoid(out)
        # loss = self.loss(out.float(), labels.float())
        out = torch.masked_select(out, target_mask)


        # print("==========out============")
        # print(out)
        labels = torch.masked_select(labels, target_mask)
        if out.shape != labels.shape:
            labels = labels.squeeze(-1)

        loss = self.loss(out.float(), labels.float())
        # print("==========labels============")
        # print(labels)
        self.log("val_loss", loss, on_step=True, prog_bar=True)
        output = {"outs": out, "labels": labels}
        return {"val_loss": loss, "outs": out, "labels": labels}

    def validation_epoch_end(self, validation_ouput):
        out = np.concatenate([i["outs"].cpu().detach().numpy()
                              for i in validation_ouput]).reshape(-1)
        labels = np.concatenate([i["labels"].cpu().detach().numpy()
                                 for i in validation_ouput]).reshape(-1)
        auc = roc_auc_score(labels, out)
        acc = accuracy_score(labels, out.round())
        self.val_auc.append(auc)
        self.val_acc.append(acc)
        self.print("")
        # self.print("out", out)
        # self.print("labels", labels)
        self.print("val auc", auc)
        self.log("val_auc", auc)
        self.print("val acc", acc)
        self.log("val_acc", acc)
        # if self.epoch_num == Config.EPOCH_NUM:
        #     self.print("[val auc]", self.val_auc)
        #     self.print("[val acc]", self.val_acc)

    def test_step(self, batch, batch_ids):
        _input, ans, labels = batch
        target_mask = (_input["input_ids"] != 0)
        out = self(_input, ans, labels)
        # out = torch.sigmoid(out)
        # loss = self.loss(out.float(), labels.float())
        eid = _input["input_ids"]  # bs sl
        eid = torch.masked_select(eid, target_mask)
        out = torch.masked_select(out, target_mask)
        labels = torch.masked_select(labels, target_mask)
        if out.shape != labels.shape:
            labels = labels.squeeze(-1)
        loss = self.loss(out.float(), labels.float())
        self.log("test_loss", loss, on_step=True, prog_bar=True)
        return {"test_loss": loss,"outs": out, "labels": labels, "exes": eid}

    def test_step_end(self, output_results):
        out = output_results["outs"].cpu().detach().numpy()
        labels = output_results["labels"].cpu().detach().numpy()

        auc = roc_auc_score(labels, out)
        acc = accuracy_score(labels, out.round())
        # print("auc",auc," acc",acc)
        self.test_auc.append(auc)
        self.test_acc.append(acc)
        # self.print("")
        # self.print("out", out)
        # self.print("labels", labels)
        # self.print("test auc", auc)
        self.log("test_auc", auc)
        # self.print("test acc", acc)
        self.log("test_acc", acc)
        # if self.epoch_num == Config.EPOCH_NUM:
        #     self.print("[test auc]", self.test_auc)
        #     self.print("[test acc]", self.test_acc)


class PredictLayer(nn.Module):
    def __init__(self, n_categories, n_fc, h_dim, dropout=0):
        super(PredictLayer, self).__init__()
        self.n_fc = n_fc
        self.h_dim = h_dim
        # category参数
        # self.c_param = nn.Embedding(n_categories, 1)  # 猜测
        # self.a_param = nn.Embedding(n_categories, 1)  # 区分度
        # self.b_param = nn.Embedding(n_categories, 1)  # 难度
        self.fc_a = nn.Linear(n_fc//Config.CON, 1)
        # self.fc_a2 = nn.Linear(self.h_dim, self.h_dim//2)
        # self.fc_a3 = nn.Linear(self.h_dim, 1)

        self.fc_theta = nn.Linear(n_fc+n_fc//Config.CON, self.h_dim)
        self.norm_layer = nn.LayerNorm(n_fc//Config.CON)
        # self.ffn = FFN(n_fc)
        self.fc_theta2 = nn.Linear(self.h_dim, self.h_dim//2)
        self.fc_theta3 = nn.Linear(self.h_dim//2, 1)
        # self.fc_forget = nn.Linear(1, 1)
        self.l1 = nn.Parameter(torch.rand(1))
        # self.fc_theta = nn.Linear(n_fc, Config.TOTAL_CAT)
        # self.fc = nn.Linear(n_fc//Config.CON, 1)
        self.theta_b = nn.Parameter(torch.rand(1))
        # self.d = nn.Parameter(torch.randn(1))
        self.dropout = nn.Dropout(dropout)
        # self.lag_layer = nn.Linear(3, 1)
        self.ls_w = nn.Parameter(torch.rand(1))
        self.lm_w = nn.Parameter(torch.rand(1))
        self.ld_w = nn.Parameter(torch.rand(1))

    def forward(self, x, cat_params, categories, lt_s, lt_m, lt_d):
        # c_p = torch.sigmoid(self.c_param(categories))
        # a_p = 8 * (torch.sigmoid(self.a_param(categories)) - 0.5)
        # b_p = 8 * (torch.sigmoid(self.b_param(categories)) - 0.5)
        # c_p = torch.sigmoid(cat_params['c_p'])
        # a_p = cat_params['a_p']  # bs sl 10
        b_p = cat_params['b_p']  # bs sl 1
        # b_p = self.fc(b_emb)
        c_emb = cat_params['c_weighted']

        # lag_time = self.lag_layer(torch.cat([lt_s,lt_m,lt_d],dim=-1))

        c_emb_input = self.norm_layer(c_emb)
        a_p = F.relu(self.fc_a(c_emb_input))
        # a_p = self.dropout(a_p)
        # # a_p = self.fc_a2(a_p)
        # # a_p = self.dropout(a_p)
        # a_p = self.fc_a3(a_p)
        a_p = 4 * torch.sigmoid(a_p)  # 0 - 4

        # x = self.norm_layers(x)
        in_x = torch.cat([x, c_emb], dim=-1)
        theta = F.relu(self.fc_theta(in_x))
        theta = self.dropout(theta)
        theta = self.fc_theta2(theta)
        theta = self.dropout(theta)
        theta = self.fc_theta3(theta)

        # theta = self.norm_layers(x)
        # theta = self.ffn(x)
        # theta = self.fc_theta(theta)

        # theta = self.fc_theta(x)
        forget = self.ls_w * torch.exp(-torch.abs(lt_s)) + \
                 self.lm_w * torch.exp(-torch.abs(lt_m)) + \
                 self.ld_w * torch.exp(-torch.abs(lt_d))
        # forget = self.fc_forget(forget)
        forget = F.softmax(forget, dim=1)
        if Config.Forget:
            theta = (1 - self.l1) * theta + self.l1 * forget + self.theta_b
        else:
            theta = theta + self.theta_b
        # a_p = F.softmax(a_p, dim=-1)
        # b_p = F.softmax(b_p, dim=-1)
        # theta_g = self.fc_theta(x)  # bs
        # sl cnum
        # theta = torch.zeros(a_p.shape)
        # t_t_g = torch.split(theta_g, 1, dim=1)
        # t_t_g = torch.cat(t_t_g, dim=0).squeeze(1)  # sl*bs cnum
        # t_cat = torch.split(categories, 1, dim=1)
        # t_cat = torch.cat(t_cat, dim=0).squeeze(1)  # sl*bs 10
        # temp = []
        # for i in range(Config.MAX_CATS_PER):
        #     indices = t_cat[:,i]
        #     data = torch.diag(torch.index_select(t_t_g[:,:], dim=-1, index=indices))
        #     temp.append(data)
        # temp = torch.stack(temp, dim=-1)  # sl*bs 10
        # temp = torch.split(temp, x.shape[0], dim=0)
        # theta = torch.stack(temp, dim=1)  # bs sl 10
        # mul = torch.sum(torch.mul(a_p,theta), dim=-1).unsqueeze(-1)
        # exp = 1.7 * (mul - b_p)
        # if Config.Forget:
        #     theta = (1 - self.l1) * theta + self.l1 * forget + self.theta_b
        # else:
        #     theta = theta + self.theta_b
        # theta = theta + self.theta_b
        exp = 1.7 * a_p * (theta - b_p)
        # exp = 2 * (torch.sigmoid(torch.exp(exp_i)) - 0.5)
        # output = c_p + (1 - c_p) * torch.sigmoid(exp)
        # output = (1-self.l1)*exp + self.l1*forget
        # print("a_p",a_p[0,:,:])
        # input()
        # print("theta",theta[0,:,:])
        # input()
        # print("b_p",b_p[0,:,:])
        # input()
        # print("forget",(forget.squeeze(-1))[0,:])
        # input()
        # print("self.l1",self.l1)
        # input()
        output = exp
        # output = output + torch.sigmoid(output)
        # output = self.fc(output.expand(output.shape[0],output.shape[1],self.n_fc))
        # output = self.dropout(output) + output
        # output = torch.sigmoid(output)
        return torch.sigmoid(output)


# 两层前馈网络
class FFN(nn.Module):
    def __init__(self, in_feat, ffn_dropout):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(in_feat, in_feat // 2)
        self.linear2 = nn.Linear(in_feat // 2, in_feat)
        self.dropout = nn.Dropout(ffn_dropout)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)
        return out


class SENETLayer_weight(nn.Module):
    def __init__(self, filed_size, reduction_ratio=3, seed=1024, device=Config.device):
        super(SENETLayer_weight, self).__init__()
        self.seed = seed
        self.filed_size = filed_size
        self.reduction_size = max(1, filed_size // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(self.filed_size, self.reduction_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.reduction_size, self.filed_size, bias=False),
            nn.ReLU()
        )
        self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        Z = torch.mean(inputs, dim=-1, out=None)
        A = self.excitation(Z)
        # V = torch.mul(inputs, torch.unsqueeze(A, dim=2))
        return A


class EncoderEmbedding(nn.Module):
    def __init__(self, n_exercises, n_categories, n_responses, n_dims, seq_len, max_cnum, bs, att_layer_num=1):
        super(EncoderEmbedding, self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.max_cnum = max_cnum
        self.bsize = bs
        # 不用eid作索引 只用category
        # self.exercise_embed = nn.Embedding(n_exercises, n_dims)
        self.category_embed = nn.Embedding(n_categories, n_dims, padding_idx=0)
        self.response_embed = nn.Embedding(n_responses, n_dims)
        # self.position_embed = nn.Embedding(seq_len,  n_dims)
        self.position_embed = nn.Embedding(seq_len, Config.CON * n_dims)
        # self.rope = RotaryEmbedding(dim=Config.CON * n_dims)
        # self.bl_embed = nn.Embedding(Config.DIFF_NUM+1, n_dims)

        # self.lag_time_s = nn.Embedding(Config.LAG_S, n_dims)
        # self.lag_time_m = nn.Embedding(Config.LAG_M, n_dims)
        # self.lag_time_d = nn.Embedding(Config.LAG_D, n_dims)
        # self.lag_layer = nn.Linear(3*n_dims, n_dims)

        # self.ls_w = nn.Parameter(torch.rand(1))
        # self.lm_w = nn.Parameter(torch.rand(1))
        # self.ld_w = nn.Parameter(torch.rand(1))

        # category参数
        # self.a_param = torch.randn(n_exercises, max_cnum)  # 区分度
        # self.c_param = torch.Tensor(n_categories + 1, 1)  # 猜测
        # self.c_param = nn.Parameter(self.c_param)
        # self.a_param = nn.Parameter(self.a_param)
        # torch.nn.init.xavier_uniform_(self.a_param)
        self.b_param = nn.Linear(1, n_dims)
        self.b_param_2 = nn.Linear(n_dims, 1)

        self.emb_layer = nn.Linear(2 * n_dims, n_dims)

        # AutoInt
        self.c_AI = nn.ModuleList(
            [InteractingLayer(self.n_dims, 2, True, device=Config.device)
             for _ in range(att_layer_num)])
        # self.c_p_AI = nn.ModuleList(
        #     [InteractingLayer(1, 1, True, device=Config.device)
        #      for _ in range(att_layer_num)])
        # self.a_p_AI = nn.ModuleList(
        #     [InteractingLayer(1, 1, True, device=Config.device)
        #      for _ in range(1)])
        # self.lag_AI = nn.ModuleList(
        #     [InteractingLayer(self.n_dims, 2, True, device=Config.device)
        #      for _ in range(att_layer_num)])
        # SENet
        self.SE = SENETLayer_weight(filed_size=Config.MAX_CATS_PER, reduction_ratio=2, seed=1024, device=Config.device)

    def forward(self, exercises, categories, cate_num, exe_diff, lt_s, lt_m, lt_d, responses):
        # excises: batch_size * 100
        # categories: batch_size * 100
        self.bsize = categories.shape[0]
        # print(self.a_param)
        # input()
        # print("lt_s.shape", lt_s.shape)
        # input()
        # print("lt_s", lt_s)
        # input()

        r = self.response_embed(responses.long())
        no_res = torch.ones_like(responses).to(Config.device) * 2
        no_r = self.response_embed(no_res.long())
        seq = torch.arange(self.seq_len, device=Config.device).unsqueeze(0)
        # seq: [[1, 2, 3, 4, ... , seq_len-1]]
        p = self.position_embed(seq)

        # 多concept嵌入
        if Config.Variants == "Mean":
            c = self.category_embed(categories[:, :, 0].long())
            for i in range(1, self.max_cnum):
                temp_c = self.category_embed(categories[:, :, i].long())
                c += temp_c
        else:
            # temp_c_p = [self.c_param[categories[:, :, 0]]]
            tag = torch.full(categories[:, :, 0].shape, Config.TOTAL_CAT - 1, device=Config.device)
            categories[:, :, 0] = torch.where(categories[:, :, 0] == 0,
                                              tag,
                                              categories[:, :, 0])
            # temp_a_p = [self.a_param[categories[:, :, 0]]]
            # t_exe = torch.split(exercises, 1, dim=1)
            # t_exe = torch.cat(t_exe, dim=0).squeeze(1)
            # temp_a_p = torch.index_select(self.a_param, dim=0, index=t_exe)
            # temp_a_p = torch.split(temp_a_p, self.bsize, dim=0)

            # temp_a_p = [self.a_param[exercises[:, 0],:]]  # B_S * 10
            temp_c = [self.category_embed(categories[:, :, 0])]
            # for i in range(1, self.seq_len):
            #     temp_a_p.append(self.a_param[exercises[:, i],:])
            for i in range(1, Config.MAX_CATS_PER):
                temp_c.append(self.category_embed(categories[:, :, i]))
                # temp_a_p.append(self.a_param[categories[:, :, i]])
            # a_p = torch.stack(temp_a_p, dim=-2)  # B_S * SEQ_LEN * 10
            # a_p = torch.stack(temp_a_p, dim=1)  # bs sl 10
            c = torch.stack(temp_c, dim=-2)  # B_S * SEQ_LEN * 10 * DIM

        # multi_concept: weighted
        if Config.Variants == "Mean":
            c_n = cate_num.unsqueeze(-1)
            cn = torch.where(c_n == 0, 1, c_n)
            c_weighted = c / cn
        elif Config.Variants == "SENet":
            t_c = torch.split(c, 1, dim=1)  # seq_len * (b_s * 10 * dim)
            t_c = torch.cat(t_c, dim=0).squeeze(1)  # seq_len*b_s  * 10  * dim
            t_c_w = self.SE(t_c.float())  # seq_len*b_s  * 10
            t_c_w = torch.split(t_c_w, self.bsize, dim=0)  # seq_len * (b_s * 10)
            t_c_w = torch.stack(t_c_w, dim=1)  # B_S * SEQ_LEN * 10
            c_w = t_c_w
            # c_w : B_S * SEQ_LEN * 10
            # c   : B_S * SEQ_LEN * 10 * DIM
            t_c = torch.mul(c, torch.unsqueeze(c_w, dim=-1))
            # Residual
            t_c = t_c + c
            c_weighted = torch.sum(t_c, dim=-2)
            # c_weighted   : B_S * SEQ_LEN * DIM
        else:
            # multi_concept: weighted
            t_c = torch.split(c, 1, dim=1)  # seq_len * (b_s * 10 * dim)
            t_c = torch.cat(t_c, dim=0).squeeze(1)  # seq_len*b_s  * 10  * dim
            # t_c_w = torch.Tensor(2,self.bsize*self.seq_len,self.max_cnum,self.max_cnum)
            # t_a_p = torch.split(torch.unsqueeze(a_p, -1), 1, dim=1)  # seq_len * (b_s * 1 * 10 * 1)
            # t_a_p = torch.cat(t_a_p, dim=0).squeeze(1)  # seq_len*b_s  * 10  * 1
            for layer in self.c_AI:
                t_c = layer(t_c.float())
                # seq_len*b_s  * 10  * dim
                # 2 seq_len*b_s  * 10  * 10
            t_c = torch.sum(t_c, dim=-2)  # seq_len*b_s  * dim
            t_c = torch.split(t_c, self.bsize, dim=0)  # seq_len * (b_s * dim)
            c_weighted = torch.stack(t_c, dim=1)  # B_S * SEQ_LEN * dim

        # t_c = torch.split(c, 1, dim=1)  # seq_len * (b_s * 10 * dim)
        # t_c = torch.cat(t_c, dim=0).squeeze(1)  # seq_len*b_s  * 10  * dim
        # # t_c_w = torch.Tensor(2,self.bsize*self.seq_len,self.max_cnum,self.max_cnum)
        # # t_a_p = torch.split(torch.unsqueeze(a_p, -1), 1, dim=1)  # seq_len * (b_s * 1 * 10 * 1)
        # # t_a_p = torch.cat(t_a_p, dim=0).squeeze(1)  # seq_len*b_s  * 10  * 1
        # for layer in self.c_AI:
        #     t_c = layer(t_c.float())
        #     # seq_len*b_s  * 10  * dim
        #     # 2 seq_len*b_s  * 10  * 10
        # t_c = torch.sum(t_c, dim=-2)  # seq_len*b_s  * dim
        # t_c = torch.split(t_c, self.bsize, dim=0)  # seq_len * (b_s * dim)
        # c_weighted = torch.stack(t_c, dim=1)  # B_S * SEQ_LEN * dim
        # c_weighted   : B_S * SEQ_LEN * DIM

        # for layer in self.a_p_AI:
        #     t_a_p = layer(t_a_p.float())
        #     # bs*sl 10 1
        # t_a_p = torch.sum(t_a_p, dim=-2)
        # t_a_p = torch.split(t_a_p, self.bsize, dim=0)
        # a_p_weighted = torch.stack(t_a_p, dim=1)
        # print(a_p_weighted.shape)
        # input()
        # print(t_a_p.shape)
        # input()

        # t_c_w = torch.sum(t_c_w, dim=0)  # seq_len*b_s 10 10
        # t_a_p = torch.matmul(t_c_w, t_a_p)  # bs*sl 10 1
        # t_a_p = torch.squeeze(t_a_p, dim=-1)  # bs*sl 10
        # t_a_p = torch.split(t_a_p, self.bsize, dim=0)  # seq_len * (b_s * 10)
        # t_a_p = torch.stack(t_a_p, dim=1)  # B_S * SEQ_LEN * 10
        # ap_weighted = torch.sum(t_a_p, dim=-1).unsqueeze(-1)  # B_S * SEQ_LEN * 1

        # 1- or not ?
        # b_p = exe_diff
        b_p = torch.ones(exe_diff.shape).to(Config.device) - exe_diff
        # b_p_level = torch.floor(b_p * Config.DIFF_NUM).long()
        # b_emb = self.bl_embed(b_p_level)
        b_p = b_p.to(torch.float32)
        # b_p = exe_diff.to(torch.float32)
        b_p = b_p.unsqueeze(-1)
        exe_params1 = self.b_param(b_p)
        exe_params = self.b_param_2(exe_params1)
        # cat_params = c_p_weighted + a_p_weighted + b_p
        # cat_params = a_p_weighted
        # cat_params = cat_params.expand(cat_params.shape[0], self.seq_len, c_weighted.shape[2])

        # lt_s_emb = self.lag_time_s(lt_s)
        # lt_m_emb = self.lag_time_m(lt_m)
        # lt_d_emb = self.lag_time_d(lt_d)  # bs sl dim

        # t_lag = torch.stack([lt_s_emb,lt_m_emb,lt_d_emb], dim=-2)  # bs sl 3 dim
        # t_lag = torch.split(t_lag, 1, dim=1)  # seq_len * (b_s * 3 * dim)
        # t_lag = torch.cat(t_lag, dim=0).squeeze(1)  # seq_len*b_s  * 10  * dim
        # for layer in self.lag_AI:
        #     t_lag = layer(t_lag.float())
        #     # seq_len*b_s  * 3  * dim
        # t_lag = torch.sum(t_lag, dim=-2)  # seq_len*b_s  * dim
        # t_lag = torch.split(t_lag, self.bsize, dim=0)  # seq_len * (b_s * dim)
        # lag_emb = torch.stack(t_lag, dim=1)  # B_S * SEQ_LEN * dim
        #
        # lag_emb = self.ls_w * lt_s_emb + self.lm_w * lt_m_emb + self.ld_w * lt_d_emb
        # lag_emb = torch.cat([lt_s_emb,lt_m_emb,lt_d_emb],dim=-1)
        # lag_emb = self.lag_layer(lag_emb)
        # print("lag_emb.shape", lag_emb.shape)
        # input()
        # print("lag_emb", lag_emb)
        # input()
        # lag_time = self.ls_w * lt_s + self.lm_w * lt_m + self.ld_w * lt_d

        # emb = r + c_weighted + exe_params
        # conemb
        # conemb = torch.cat([r, c_weighted + exe_params], dim=-1)
        # emb = self.emb_layer(conemb)
        # r+e or e+r
        # e = self.exercise_embed(exercises)
        emb = c_weighted + exe_params
        conemb = torch.cat([r, emb], dim=-1)
        conemb_nor = torch.cat([no_r, emb], dim=-1)
        # if Config.CON == 2:
        #     conemb = torch.cat([r, emb], dim=-1)
        #     conemb_nor = torch.cat([no_r, emb], dim=-1)
        # else:
        #     conemb = emb + r
        #     conemb_nor = emb
        return conemb + p, conemb_nor + p , {"b_p": exe_params, "c_weighted": c_weighted+exe_params}


class DecoderEmbedding(nn.Module):
    def __init__(self, n_responses, n_dims, seq_len):
        super(DecoderEmbedding, self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.response_embed = nn.Embedding(n_responses, Config.CON * n_dims)
        self.position_embed = nn.Embedding(seq_len, Config.CON * n_dims)

    def forward(self, responses):
        r = self.response_embed(responses.long())
        seq = torch.arange(self.seq_len, device=Config.device).unsqueeze(0)
        p = self.position_embed(seq)
        return p + r


class PositionalBias(nn.Module):
    def __init__(self, max_seq, embed_dim, num_heads, bidirectional=True, num_buckets=32, max_distance=Config.MAX_SEQ):
        super(PositionalBias, self).__init__()
        self.d_model = embed_dim
        self.d_k = embed_dim // num_heads
        self.h = num_heads
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        self.pos_embed = nn.Embedding(max_seq, embed_dim)  # Encoder position Embedding
        self.pos_query_linear = nn.Linear(embed_dim, embed_dim)
        self.pos_key_linear = nn.Linear(embed_dim, embed_dim)
        self.pos_layernorm = nn.LayerNorm(embed_dim)

        self.relative_attention_bias = nn.Embedding(32, num_heads)

    def forward(self, pos_seq):
        bs = pos_seq.size(0)

        pos_embed = self.pos_embed(pos_seq)
        pos_embed = self.pos_layernorm(pos_embed)

        pos_query = self.pos_query_linear(pos_embed)
        pos_key = self.pos_key_linear(pos_embed)

        pos_query = pos_query.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        pos_key = pos_key.view(bs, -1, self.h, self.d_k).transpose(1, 2)

        absolute_bias = torch.matmul(pos_query, pos_key.transpose(-2, -1)) / math.sqrt(self.d_k)
        relative_position = pos_seq[:, None, :] - pos_seq[:, :, None]

        relative_buckets = 0
        num_buckets = self.num_buckets
        if self.bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_bias = torch.abs(relative_position)
        else:
            relative_bias = -torch.min(relative_position, torch.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = relative_bias < max_exact

        relative_bias_if_large = max_exact + (
                torch.log(relative_bias.float() / max_exact)
                / math.log(self.max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_bias_if_large = torch.min(
            relative_bias_if_large, torch.full_like(relative_bias_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_bias, relative_bias_if_large)
        relative_position_buckets = relative_buckets.to(pos_seq.device)

        relative_bias = self.relative_attention_bias(relative_position_buckets)
        relative_bias = relative_bias.permute(0, 3, 1, 2)

        position_bias = absolute_bias + relative_bias
        return position_bias


def attention(q, k, v, d_k, pad_zero, positional_bias=None, mask=None, dropout=None,
              memory_decay=False, memory_gamma=None, ltime=None):
    # ltime shape [batch, seq_len]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [bs, nh, s, s]
    bs, nhead, seqlen = scores.size(0), scores.size(1), scores.size(2)

    if mask is not None:
        mask = mask.unsqueeze(1)

    if memory_decay and memory_gamma is not None and ltime is not None:
        time_seq = torch.cumsum(ltime.float(), dim=-1) - ltime.float()  # [bs, s]
        index_seq = torch.arange(seqlen).unsqueeze(-2).to(q.device)

        dist_seq = time_seq + index_seq
        x1 = torch.arange(seqlen).expand(seqlen, -1).to(Config.device)
        x2 = x1.transpose(0, 1).contiguous()

        with torch.no_grad():
            if mask is not None:
                scores_ = scores.masked_fill(mask, -1e9)
            scores_ = F.softmax(scores_, dim=-1)  # bs h sl sl
            scores_ = torch.cat([pad_zero, scores_[:, :, 1:, :]], dim=-2)
            distcum_scores = torch.cumsum(scores_, dim=-1)
            distotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
            # position_diff = dist_seq[:, None, :] - dist_seq[:, :, None]
            # position_effect = torch.abs(position_diff)[:, None, :, :].type(torch.FloatTensor).to(q.device)
            position_effect = torch.abs(
                x1-x2)[None, None, :, :].type(torch.FloatTensor).to(Config.device)  # 1, 1, seqlen, seqlen
            dist_scores = torch.clamp((distotal_scores - distcum_scores) * position_effect, min=0.)
            dist_scores = dist_scores.sqrt().detach()

        m = nn.Softplus()
        memory_gamma = -1. * m(memory_gamma)
        total_effect = torch.clamp(torch.clamp((dist_scores * memory_gamma).exp(), min=1e-5), max=1e5)
        scores = total_effect * scores

    if positional_bias is not None:
        scores = scores + positional_bias

    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    scores = F.softmax(scores, dim=-1)  # [bs, nh, s, s]

    if dropout is not None:
        scores = dropout(scores)
    # 对sl=1 scores置0
    scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=-2)
    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, kq_same=Config.KQ_SAME, bias=True, rope_v=True):
        super(MultiHeadAttention, self).__init__()

        self.d_model = embed_dim
        self.d_k = embed_dim // num_heads
        self.h = num_heads
        self.kq_same = kq_same
        self.proj_bias = bias

        if kq_same is False:
            self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gammas = nn.Parameter(torch.zeros(num_heads, Config.MAX_SEQ, 1))
        torch.nn.init.xavier_uniform_(self.gammas)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.rope_v = rope_v
        self.rope = RotaryEmbedding(dim=embed_dim//num_heads) if self.rope_v else None
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.k_linear.weight)
        torch.nn.init.xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            torch.nn.init.xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            torch.nn.init.constant_(self.k_linear.bias, 0.)
            torch.nn.init.constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                torch.nn.init.constant_(self.q_linear.bias, 0.)
            torch.nn.init.constant_(self.out.bias, 0.)

    def forward(self, q, k, v, ltime=None, gamma=None, positional_bias=None,
                attn_mask=None):
        # cache with a key that is the sequence length, so that it does not need to recompute
        if self.rope_v:
            freqs = self.rope(torch.arange(Config.MAX_SEQ).to(Config.device), cache_key=Config.MAX_SEQ)
            freqs = freqs[:Config.MAX_SEQ]
            freqs = freqs[None,None,...]

        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.rope_v:
            # apply the rotations to your queries and keys after the heads have been split out
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)
            # but prior to the dot product and subsequent softmax (attention)

        gamma = self.gammas
        pad_zero = torch.zeros(bs, self.h, 1, Config.MAX_SEQ).to(Config.device)  # [bs, nh, 1, s]

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, pad_zero, positional_bias, attn_mask, self.dropout,
                           memory_decay=True, memory_gamma=gamma, ltime=ltime)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


def future_mask(seq_length):
    future_mask = np.triu(np.ones((1, seq_length, seq_length)), k=0).astype('bool')  # 对角线mask
    return torch.from_numpy(future_mask)


# 多头注意力
class StackedNMultiHeadAttention(nn.Module):
    def __init__(self, n_stacks, n_dims, n_heads, seq_len, n_multihead=1, ma_dropout=0, ffn_dropout=0):
        super(StackedNMultiHeadAttention, self).__init__()
        self.n_stacks = n_stacks
        self.n_multihead = n_multihead
        self.n_dims = n_dims
        self.norm_layers = nn.LayerNorm(n_dims)
        self.max_seq = seq_len
        # n_stacks has n_multiheads each
        self.multihead_layers = nn.ModuleList(
            n_stacks * [nn.ModuleList(n_multihead * [MultiHeadAttention(embed_dim=n_dims,
                                                                        num_heads=n_heads,
                                                                        dropout=ma_dropout,
                                                                        rope_v=Config.RoPE), ]), ])
        self.ffn = nn.ModuleList(n_stacks * [FFN(n_dims,ffn_dropout)])
        # self.mask = torch.triu(torch.ones(seq_len, seq_len),
        #                        diagonal=1).to(dtype=torch.bool)

    def forward(self, input_q, input_k, input_v, pos_embed=None, ltime=None, encoder_output=None, break_layer=None):
        for stack in range(self.n_stacks):
            # in_q = input_q
            # in_k = input_k
            # in_v = input_v
            for multihead in range(self.n_multihead):
                # if multihead != 0 or multihead != 1:
                #     ltime = None
                # ltime = None
                # norm_q = self.norm_layers(in_q)
                # norm_k = self.norm_layers(in_k)
                # norm_v = self.norm_layers(in_v)
                norm_q = self.norm_layers(input_q)
                norm_k = self.norm_layers(input_k)
                norm_v = self.norm_layers(input_v)
                attn_mask = future_mask(self.max_seq).to(Config.device)
                heads_output = self.multihead_layers[stack][multihead](q=norm_q,  # 变换tensor维度
                                                                       k=norm_k,
                                                                       v=norm_v,
                                                                       positional_bias=pos_embed,
                                                                       ltime=ltime,
                                                                       attn_mask=attn_mask)
                # heads_output = heads_output.permute(1, 0, 2)
                # assert encoder_output != None and break_layer is not None
                if encoder_output != None and multihead == break_layer:
                    # 有来自encoder的信息
                    assert break_layer <= multihead, " break layer should be less than multihead layers and postive integer"
                    # in_k = in_v = encoder_output
                    # in_q = in_q + heads_output
                    input_k = input_v = encoder_output
                    input_q = input_q + heads_output
                else:
                    # in_q = input_q + heads_output
                    # in_k = input_k + heads_output
                    # in_v = input_v + heads_output
                    input_q = input_q + heads_output
                    input_k = input_k + heads_output
                    input_v = input_v + heads_output
            last_norm = self.norm_layers(heads_output)
            # last_norm = self.norm_layers(heads_output+in_q)
            ffn_output = self.ffn[stack](last_norm)
            # ffn_output = ffn_output + last_norm
            ffn_output = ffn_output + heads_output
        # after loops = input_q = input_k = input_v
        return ffn_output


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
        # if k == "encoder_embedding.a_param"}

        model_dict.update(new_dict)
        # 打印出来，更新了多少的参数
        print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        model.load_state_dict(model_dict)
        print("loaded finished!")
    else:
        print('No checkpoint is included')
    return model


def tune_asha(num_samples=10, num_epochs=Config.EPOCH_NUM, gpus_per_trial=Config.DEVICE_NUM):
    config = {
        "LR": tune.choice([1e-4, 5e-4, 1e-3, 3e-3]),
        "DECAY": tune.uniform(0, 1),
        "EMBED_DIMS": tune.choice([64, 128, 256]),
        "H_DIMS": tune.choice([64, 128, 256]),
        "ENC_HEADS": tune.choice([2, 4, 8, 16]),
        "NUM_ENCODER": tune.choice([1, 2, 3, 4, 5, 6, 7, 8]),
        "BATCH_SIZE": tune.choice([32, 64, 128, 256,512]),
        "MA_DROP_OUT": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        "PL_DROP_OUT": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        "FFN_DROP_OUT": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        "LAYER_NUM": tune.choice([1, 2, 3]),
    }
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2
    )
    reporter = CLIReporter(
        parameter_columns=["LR", "DECAY", "EMBED_DIMS", "H_DIMS",
                           "ENC_HEADS", "NUM_ENCODER", "BATCH_SIZE", "MA_DROP_OUT",
                           "PL_DROP_OUT", "FFN_DROP_OUT", "LAYER_NUM"],
        metric_columns=["val_loss", "val_auc", "val_acc", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_tune,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": 1,
            "gpu": 1
        },
        local_dir="ray_results",
        metric="val_loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=Config.DATASET+"_"+Config.MODEL_NAME+"_tune")
    print("Best hyperparameters found were: ", analysis.best_config)
    return analysis.best_config


def train_tune(config, num_epochs=10, num_gpus=Config.DEVICE_NUM):
    model = Encoder_IRT_Modle(config)
    train_loader, val_loader = get_dataloaders()
    ray_checkpoint_callback = TuneReportCheckpointCallback(
        metrics={"val_loss": "val_loss",
                 "val_auc": "val_auc",
                 "val_acc": "val_acc"},
        filename="tune_{epoch:02d}-{step}-{val_auc:.2f}-{val_acc:.2f}",
        on="validation_end")
    trainer = pl.Trainer(gpus=num_gpus, max_epochs=num_epochs,
                         distributed_backend="ddp",
                         # gradient_clip_val=Config.CLIP,
                         gradient_clip_val=Config.CLIP,
                         # plugins=DDPPlugin(find_unused_parameters=False),
                         callbacks=[
                             ray_checkpoint_callback]
                         )
    # train
    trainer.fit(model=model,
                train_dataloader=train_loader,
                val_dataloaders=[val_loader, ])

if __name__ == "__main__":
    start_time = time()
    if not Config.DDP:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        ddp_str = None
    else:
        ddp_str = "ddp"
    torch.manual_seed(Config.seed)  #为CPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.seed)  #为所有GPU设置随机种子
    print("============" + Config.MODEL_NAME + "============")
    print("DATASET: ", Config.DATASET)
    print("INFO: ", Config.INFO)

    # search hp
    # tune_asha()
    config = Config.hp_conf

    train_loader, val_loader = get_dataloaders(bs=config["BATCH_SIZE"])
    test_loader  = get_test_dataloaders(bs=config["BATCH_SIZE"])
    model = Encoder_IRT_Modle(config)
    auc_checkpoint_callback = ModelCheckpoint(
        monitor="val_auc",
        filename=Config.MODEL_NAME +"_{epoch:02d}-{step}-{val_auc:.2f}",
        save_top_k=1,
        mode="max",
    )
    acc_checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        filename=Config.MODEL_NAME +"_{epoch:02d}-{step}-{val_acc:.2f}",
        save_top_k=1,
        mode="max",
    )
    loss_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename=Config.MODEL_NAME +"_{epoch:02d}-{step}",
        save_top_k=1,
        mode="min",
    )
    # model = load_checkpoint(model, "lightning_logs/version_230/checkpoints/epoch=2-step=2219.ckpt")
    trainer = pl.Trainer(gpus=Config.DEVICE_NUM, max_epochs=Config.EPOCH_NUM,
                         # distributed_backend=ddp_str,
                         gradient_clip_val=Config.CLIP,
                         # plugins=DDPPlugin(find_unused_parameters=False),
                         callbacks=[
                             auc_checkpoint_callback,
                             acc_checkpoint_callback,
                             loss_checkpoint_callback
                         ]
                         )
    # train
    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=[val_loader, ])


    # test
    # model = load_checkpoint(model, "/home/b418/kr/ACAKT/visual/MCAKT/as12/version_66/checkpoints/MCAKT-A_epoch=67-step=22712.ckpt")
    # trainer.test(model, dataloaders=test_loader)
    # model = load_checkpoint(model, "/home/b418/kr/ACAKT/visual/MCAKT/as12/version_66/checkpoints/MCAKT-A_epoch=69-step=23380-val_acc=0.77.ckpt")
    # trainer.test(model, dataloaders=test_loader)
    # model = load_checkpoint(model, "/home/b418/kr/ACAKT/visual/MCAKT/as12/version_66/checkpoints/MCAKT-A_epoch=69-step=23380-val_acc=0.77.ckpt")
    # trainer.test(model, dataloaders=test_loader)
    model_test = Encoder_IRT_Modle(config)
    path0 = trainer.checkpoint_callbacks[0].best_model_path  # empty here !
    print("path0: ", path0)
    model0 = load_checkpoint(model,path0)
    trainer.test(model0, dataloaders=test_loader, ckpt_path=path0)
    path1 = trainer.checkpoint_callbacks[1].best_model_path  # empty here !
    print("path1: ", path1)
    model1 = load_checkpoint(model,path1)
    trainer.test(model1, dataloaders=test_loader, ckpt_path=path1)
    path2 = trainer.checkpoint_callbacks[-1].best_model_path  # empty here !
    print("path2: ", path2)
    model2 = load_checkpoint(model_test,path2)
    trainer.test(model2, dataloaders=test_loader, ckpt_path=path2)

    print("TOTAL TIME: ", time() - start_time)
