import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import argparse
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from torch.autograd import grad, Variable

from RAdam import *

from config import Config
from dataset import get_dataloaders, get_test_dataloaders
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from time import *

from model import StackedNMultiHeadAttention, EncoderEmbedding, PositionalBias, PredictLayer
from utils import FGM, load_checkpoint, randomSamples, Similarity, _l2_normalize_adv


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
                                                  att_layer_num=self.LAYER_NUM,
                                                  mcakt=Config.MCAKT,
                                                  autoint=Config.AutoInt)
        self.elapsed_time = nn.Linear(1, Config.CON * self.EMBED_DIMS)  # 时间信息

        self.pos_embed = PositionalBias(max_seq=Config.MAX_SEQ, embed_dim=Config.CON * self.EMBED_DIMS,
                                        num_heads=self.ENC_HEADS,
                                        bidirectional=False, num_buckets=32,
                                        max_distance=Config.MAX_SEQ)
        # self.pred_mlp = nn.Sequential(nn.Linear(2*self.EMBED_DIMS, self.H_DIMS), nn.Dropout(0.2),
        #                                nn.Linear(self.H_DIMS, 1),
        #                               nn.Sigmoid())
        self.predict_layer = PredictLayer(n_categories=Config.TOTAL_CAT,
                                          # n_fc= Config.EMBED_DIMS,)
                                          n_fc=Config.CON * self.EMBED_DIMS,
                                          h_dim=self.H_DIMS,
                                          dropout=self.PL_DROP_OUT)

        self.fgm = FGM(self)
        self.automatic_optimization = False if Config.AT=='FGM' else True
        self.samp = Config.samp
        self.cl_loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        self.sim = Similarity(temp=Config.temp)
        self.cl_mlp = nn.Sequential(nn.Linear(2*self.EMBED_DIMS, self.H_DIMS), nn.ReLU(),
                                    nn.Linear(self.H_DIMS, 2*self.EMBED_DIMS))

        if Config.VISUAL:
            self.exes_list = None
            self.exe_emb_list = None
            self.cat_list = None


    def forward(self, x, y, labels, at_dict=None):
        perturbation = None
        genSamp = None
        samp = None
        sampleIndex = None
        ori_samps = None
        cl_loss = None
        adv_samps = None
        mask_labels = torch.where(x["input_ids"]==0,0,1)

        if at_dict is not None:
            perturbation = at_dict['perturbation']
            genSamp = at_dict['genSamp']
            samp = at_dict['samp']
            sampleIndex = at_dict['sampleIndex']
            ori_samps = at_dict['ori_samps']
            cl_loss = 0
            adv_samps = None

        enc, enc_wo_r, cat_params = self.encoder_embedding(
            exercises=x["input_ids"], categories=x['input_cat'],
            cate_num=x["input_cnum"], exe_diff=x["input_diff"],
            lt_s=None, lt_m=None, lt_d=None,
            responses=labels, at_dict=at_dict,mcakt = Config.MCAKT,autoint=Config.AutoInt)

        pos_bias_embed = self.pos_embed(torch.arange(Config.MAX_SEQ).unsqueeze(0).to(Config.device)) if Config.Pos_Bias else None
        encoder_output = self.encoder_layer(input_k=enc,
                                            input_q=enc_wo_r,
                                            input_v=enc,
                                            pos_embed=pos_bias_embed,
                                            ltime=x["input_rtime"]
                                            )
        if Config.NEG:
            neg_encoder_output = self.encoder_layer(input_k=cat_params['neg_concept_answer_embedding'],
                                            input_q=enc_wo_r,
                                            input_v=cat_params['neg_concept_answer_embedding'],
                                            pos_embed=pos_bias_embed,
                                            ltime=x["input_rtime"]
                                            )

        # gen adv_samps
        if sampleIndex is not None and ori_samps is not None:
            pad_out_adv = torch.mul(encoder_output, (mask_labels.unsqueeze(-1)))  # bs seq dim
            all_out_adv = torch.split(pad_out_adv, 1, dim=0)  # seq_len * (b_s * 1 * dim)
            all_out_adv = torch.cat(all_out_adv, dim=1).squeeze(0)  # seq_len*b_s  *  dim
            adv_samps = randomSamples(all_out_adv, samp, mask_labels, sampleIndex) # 随机采样
            # adv_samps = pad_out_adv.sum(1) / mask_labels.sum(-1).unsqueeze(-1) # 池化

            adv_samps_out = self.cl_mlp(adv_samps)
            ori_samps_out = self.cl_mlp(ori_samps)
            # contrastive loss
            out_cos_sim = self.sim(adv_samps_out.unsqueeze(1), ori_samps_out.unsqueeze(0))  # bs bs

            if Config.NEG:
                neg_pad_out_adv = torch.mul(neg_encoder_output, (mask_labels.unsqueeze(-1)))  # bs seq dim
                neg_all_out_adv = torch.split(neg_pad_out_adv, 1, dim=0)  # seq_len * (b_s * 1 * dim)
                neg_all_out_adv = torch.cat(neg_all_out_adv, dim=1).squeeze(0)  # seq_len*b_s  *  dim
                neg_samps = randomSamples(neg_all_out_adv, samp, mask_labels, sampleIndex)  # 随机采样
                neg_samps_out  = self.cl_mlp(neg_samps)
                neg_inter_cos_sim = self.sim(neg_samps_out.unsqueeze(1), ori_samps_out.unsqueeze(0))
                out_cos_sim = torch.cat([out_cos_sim, neg_inter_cos_sim], 1)

                # weights = torch.tensor(
                #     [
                #         [0.0] * (out_cos_sim.size(-1) - neg_inter_cos_sim.size(-1))
                #         + [0.0] * i
                #         + [1.0]
                #         + [0.0] * (neg_inter_cos_sim.size(-1) - i - 1)
                #         for i in range(neg_inter_cos_sim.size(-1))
                #     ]
                # ).to(out_cos_sim.device)
                # out_cos_sim = out_cos_sim + weights

            out_labels = torch.arange(out_cos_sim.size(0)).long().to(
                out_cos_sim.device)  # 表示每个样本最相似的样本下标，即zq_+1_i&zq_+2_i
            cl_loss = self.cl_loss_fn(out_cos_sim, out_labels)  # 对比loss

        # gen ori_samps
        if genSamp:
            pad_out_ori = torch.mul(encoder_output, (mask_labels.unsqueeze(-1)))  # bs seq dim
            all_out_ori = torch.split(pad_out_ori, 1, dim=0)  # seq_len * (b_s * 1 * 2hdim)
            all_out_ori = torch.cat(all_out_ori, dim=1).squeeze(0)  # seq_len*b_s  *  2hdim
            ori_samps, sampleIndex = randomSamples(all_out_ori, samp, mask_labels)
            # ori_samps = pad_out_ori.sum(1) / mask_labels.sum(-1).unsqueeze(-1)

        elapsed_time = x["input_rtime"].unsqueeze(-1).float()
        ela_time = self.elapsed_time(elapsed_time)
        encoder_output = encoder_output + ela_time

        if Config.MCAKT:
            out = self.predict_layer(x=encoder_output, cat_params=cat_params, categories=x['input_cat'],
                                     lt_s=x["input_lag_time_s"].unsqueeze(-1).float(),
                                     lt_m=x["input_lag_time_m"].unsqueeze(-1).float(),
                                     lt_d=x["input_lag_time_d"].unsqueeze(-1).float())
            # out = self.pred_mlp(encoder_output)
        else:
            out = self.pred_mlp(encoder_output)

        out = out.squeeze()
        return {'out':out, 'features':cat_params['c_weighted'],
                'ori_samps': ori_samps, 'sampleIndex': sampleIndex,
                'adv_samps': adv_samps, 'cl_loss': cl_loss}

    def configure_optimizers(self):
        optimizer = RAdam(self.parameters(),
                               lr=self.LR,
                               weight_decay=self.DECAY,
                               )
        return optimizer
        # return torch.optim.AdamW(self.parameters(), lr=self.LR)

    def training_step(self, batch, batch_ids):
        _input, ans, labels = batch
        target_mask = (_input["input_ids"] != 0)

        if Config.AT == "FGM":
            opt = self.optimizers()
            opt.zero_grad()

            at_params = {
                'perturbation': None,
                'genSamp': True,
                'samp': self.samp,
                'sampleIndex': None,
                'ori_samps': None,
            }

            output_dict = self(_input, ans, labels, at_params)
            out = output_dict['out']
            if len(out.shape)==1:
                out = out.unsqueeze(0)
            loss = self.loss(out.float(), labels.float())

            features_grad = grad(loss, output_dict['features'], retain_graph=True)
            p_adv = torch.FloatTensor(Config.epsilon * _l2_normalize_adv(features_grad[0].data))
            p_adv = Variable(p_adv).to(out.device)

            # self.manual_backward(loss,retain_graph=True)
            # loss.backward(retain_graph=True)
            # self.fgm.attack(emb_name="category_embed", epsilon=Config.epsilon)

            at_params = {
                'perturbation': p_adv,
                'genSamp': False,
                'samp': self.samp,
                'sampleIndex': output_dict['sampleIndex'],
                'ori_samps': output_dict['ori_samps'],
            }

            out_adv_dict = self(_input, ans, labels, at_params)
            out_adv = out_adv_dict['out']
            if len(out_adv.shape)==1:
                out_adv = out_adv.unsqueeze(0)
            loss_adv = self.loss(out_adv.float(), labels.float())
            if Config.LOSS == 0:
                #SCAL
                total_loss = 0.5*(loss +loss_adv) + Config.theta * out_adv_dict['cl_loss']
            elif Config.LOSS == 1:
                # ROL
                total_loss = 0*loss + loss_adv + Config.theta * out_adv_dict['cl_loss']
            elif Config.LOSS == 2:
                # RAL
                total_loss = loss + Config.theta * out_adv_dict['cl_loss']
            elif Config.LOSS == 3:
                # RCL
                # total_loss =  0.5*(loss +loss_adv)
                total_loss = loss +  Config.theta *  loss_adv
            else:
                total_loss = 0.5*(loss +loss_adv) + Config.theta * out_adv_dict['cl_loss']
            # total_loss = loss + Config.beta*loss_adv + Config.theta * out_adv_dict['cl_loss']
            # self.fgm.restore(emb_name="category_embed")
            self.manual_backward(total_loss)
            # total_loss.backward()
            opt.step()

        # out = self(_input, ans, labels)
        # if out.shape != labels.shape:
        #     labels = labels.squeeze(-1)

        else:
            # opt = self.optimizers()
            # opt.zero_grad()
            output_dict = self(_input, ans, labels)
            out = output_dict['out']
            total_loss = self.loss(out.float(), labels.float())
            # total_loss.backward()

        out = torch.masked_select(out, target_mask)

        labels = torch.masked_select(labels, target_mask)
        # loss = self.loss(out.float(), labels.float())
        self.log("train_loss", total_loss, on_step=True, prog_bar=True)
        return {"loss": total_loss, "outs": out, "labels": labels}

    def training_epoch_end(self, training_ouput):
        out = np.concatenate([i["outs"].cpu().detach().numpy()
                              for i in training_ouput]).reshape(-1)
        labels = np.concatenate([i["labels"].cpu().detach().numpy()
                                 for i in training_ouput]).reshape(-1)
        auc = roc_auc_score(labels, out)
        acc = accuracy_score(labels, out.round())
        self.print("[EPOCH ",self.current_epoch,"] train auc", auc," train acc", acc)
        self.print("[EPOCH ", self.current_epoch, "] train auc", auc, " train acc", acc,file=log)
        self.log("train_auc", auc)
        self.log("train_acc", acc)

    def validation_step(self, batch, batch_ids):
        opt = self.optimizers()
        opt.zero_grad()
        _input, ans, labels = batch
        target_mask = (_input["input_ids"] != 0)
        out_dict = self(_input, ans, labels)
        out = out_dict['out']
        out = torch.masked_select(out, target_mask)
        labels = torch.masked_select(labels, target_mask)
        if out.shape != labels.shape:
            labels = labels.squeeze(-1)

        loss = self.loss(out.float(), labels.float())

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
        self.print("[EPOCH ",self.current_epoch,"] val auc", auc," val_acc", acc)
        self.print("[EPOCH ", self.current_epoch, "] val auc", auc, " val_acc", acc,file=log)
        self.log("val_auc", auc)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_ids):
        _input, ans, labels = batch
        target_mask = (_input["input_ids"] != 0)
        out_dict = self(_input, ans, labels)
        out = out_dict['out']
        eid = _input["input_ids"]  # bs sl
        eid = torch.masked_select(eid, target_mask)
        out = torch.masked_select(out, target_mask)
        if Config.VISUAL:
            cat = _input['input_cat']
            cat = torch.masked_select(cat, target_mask.unsqueeze(-1)).reshape(-1,10)
            exe_emb = out_dict['features']
            exe_emb = torch.masked_select(exe_emb, target_mask.unsqueeze(-1)).reshape(-1,128)
        labels = torch.masked_select(labels, target_mask)
        if out.shape != labels.shape:
            labels = labels.squeeze(-1)
        loss = self.loss(out.float(), labels.float())
        self.log("test_loss", loss, on_step=True, prog_bar=True)
        if Config.VISUAL:
            return {"test_loss": loss, "outs": out, "labels": labels, "exes": eid,"exe_emb": exe_emb,"cat": cat}
        return {"test_loss": loss,"outs": out, "labels": labels, "exes": eid}

    def test_step_end(self, output_results):
        out = output_results["outs"].cpu().detach().numpy()
        labels = output_results["labels"].cpu().detach().numpy()

        auc = roc_auc_score(labels, out)
        acc = accuracy_score(labels, out.round())
        # print("auc",auc," acc",acc)
        # print("auc", auc, " acc", acc,file=log)
        self.log("test_auc", auc)
        self.log("test_acc", acc)
        if Config.VISUAL:
            if self.exes_list is None:
                self.exes_list = output_results["exes"]
                self.exe_emb_list = output_results["exe_emb"]
                self.cat_list = output_results["cat"]
            else:
                self.exes_list = torch.cat([self.exes_list, output_results["exes"]], 0)
                self.exe_emb_list = torch.cat([self.exe_emb_list, output_results["exe_emb"]], 0)
                self.cat_list = torch.cat([self.cat_list, output_results["cat"]], 0)
            log_path = "/home/b418/kr/ACAKT/visual/374total"
            exes_file = os.path.join(log_path + '/exes.pt')
            exe_emb_file = os.path.join(log_path + '/exe_emb.pt')
            cat_file = os.path.join(log_path + '/cat.pt')

            torch.save(self.exes_list, exes_file)
            torch.save(self.exe_emb_list, exe_emb_file)
            torch.save(self.cat_list, cat_file)



def tune_asha(num_samples=10, num_epochs=Config.EPOCH_NUM, gpus_per_trial=Config.DEVICE_NUM):
    config = {
        "LR": tune.choice([1e-4, 5e-4, 1e-3]),
        "EMBED_DIMS": tune.choice([64, 128, 256]),
        "H_DIMS": tune.choice([64, 128, 256]),
        "BATCH_SIZE": tune.choice([32, 64, 128, 256, 512]),
        "MA_DROP_OUT": tune.choice([0, 0.1, 0.2, 0.5]),
        "FFN_DROP_OUT": tune.choice([0, 0.1, 0.2, 0.5]),
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

def get_param():
    parser = argparse.ArgumentParser(description='Script to train KT')
    parser.add_argument('--hdim', type=int, default=64, help='Dimension of concept embedding.')
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--dataset', type=str, default="Statics")
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--encoders', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int,default=64)
    parser.add_argument('--adp', type=float, default=0.5)
    parser.add_argument('--fdp', type=float, default=0.5)
    parser.add_argument('--seq', type=int, default=200)
    parser.add_argument('--model', type=str, default="ACAKT")
    parser.add_argument('--at', type=str, default="FGM")
    parser.add_argument('--samp', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--theta', type=float, default=0.1)
    parser.add_argument('--temp', type=float, default=0.02)
    parser.add_argument('--epsilon', type=float, default=10.)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gpu', type=int,choices=[0,1], default=0)
    parser.add_argument('--neg', type=bool,default=False)
    parser.add_argument('--SENet', default=False, action='store_true', help='run prepare_data or not')
    parser.add_argument('--loss', type=int, default=0)
    parser.add_argument('--worope', default=False, action='store_true', help='run prepare_data or not')
    parser.add_argument('--wof', default=False, action='store_true', help='run prepare_data or not')
    parser.add_argument('--wob', default=False, action='store_true', help='run prepare_data or not')
    parser.add_argument('--ln', type=int, default=1)
    params = parser.parse_args()
    Config.DATASET = params.dataset

    if Config.DATASET.startswith("Statics"):
        Config.Forget = False
        Config.hp_conf["LR"] = params.lr
        Config.hp_conf["EMBED_DIMS"] = 128
        Config.hp_conf["H_DIMS"] = 128
        Config.hp_conf["NUM_ENCODER"] = 4
        Config.hp_conf["BATCH_SIZE"] = params.batch_size
        Config.TOTAL_EXE = 189297
        Config.TOTAL_CAT = 1223
        Config.DATA_FILE_PATH = "../data/pre/" + Config.DATASET
        Config.TEST_FILE_PATH = "../data/pre/" + Config.DATASET
        Config.LAG_S = 5001
        Config.LAG_M = 501
        Config.LAG_D = 51
        Config.EPOCH_NUM = params.epoch

    if Config.DATASET in {"AS09_lag","AS09_lag_sc"}:
        Config.Forget = False
        Config.hp_conf["LR"] = params.lr
        Config.hp_conf["EMBED_DIMS"] = 128
        Config.hp_conf["H_DIMS"] = 128
        Config.hp_conf["NUM_ENCODER"] = 4
        Config.hp_conf["BATCH_SIZE"] = params.batch_size
        Config.TOTAL_EXE = 207349
        Config.TOTAL_CAT = 401
        Config.DATA_FILE_PATH = "../data/pre/" + Config.DATASET
        Config.TEST_FILE_PATH = "../data/pre/" + Config.DATASET
        Config.LAG_S = 5001
        Config.LAG_M = 501
        Config.LAG_D = 51
        Config.EPOCH_NUM = params.epoch
    if Config.DATASET in {"AS12_lag"}:
        Config.Forget = False
        Config.hp_conf["LR"] = params.lr
        Config.hp_conf["EMBED_DIMS"] = 128
        Config.hp_conf["H_DIMS"] = 128
        Config.hp_conf["NUM_ENCODER"] = 4
        Config.hp_conf["BATCH_SIZE"] = params.batch_size
        Config.TOTAL_EXE = 47205
        Config.TOTAL_CAT = 243
        Config.DATA_FILE_PATH = "../data/pre/" + Config.DATASET
        Config.TEST_FILE_PATH = "../data/pre/" + Config.DATASET
        Config.LAG_S = 5001
        Config.LAG_M = 501
        Config.LAG_D = 51
        Config.EPOCH_NUM = params.epoch
    if Config.DATASET in {"AS12_lag2"}:
        Config.Forget = False
        Config.hp_conf["LR"] = params.lr
        Config.hp_conf["EMBED_DIMS"] = 128
        Config.hp_conf["H_DIMS"] = 128
        Config.hp_conf["NUM_ENCODER"] = 4
        Config.hp_conf["BATCH_SIZE"] = params.batch_size
        Config.TOTAL_EXE = 50918
        Config.TOTAL_CAT = 245
        Config.DATA_FILE_PATH = "../data/pre/" + Config.DATASET
        Config.TEST_FILE_PATH = "../data/pre/" + Config.DATASET
        Config.LAG_S = 5001
        Config.LAG_M = 501
        Config.LAG_D = 51
        Config.EPOCH_NUM = params.epoch
    if Config.DATASET in {"AS12_lag3"}:
        Config.Forget = not params.wof
        Config.hp_conf["LR"] = params.lr
        Config.hp_conf["EMBED_DIMS"] = 128
        Config.hp_conf["H_DIMS"] = 128
        Config.hp_conf["NUM_ENCODER"] = params.encoders
        Config.hp_conf["BATCH_SIZE"] = params.batch_size
        Config.TOTAL_EXE = 767144
        Config.TOTAL_CAT = 1642
        Config.DATA_FILE_PATH = "../data/pre/" + Config.DATASET
        Config.TEST_FILE_PATH = "../data/pre/" + Config.DATASET
        Config.LAG_S = 5001
        Config.LAG_M = 501
        Config.LAG_D = 51
        Config.EPOCH_NUM = params.epoch
    if Config.DATASET in {"EdNet_lag6","EdNet_lag7"}:
        Config.Forget = True
        Config.hp_conf["LR"] = params.lr
        Config.hp_conf["EMBED_DIMS"] = 128
        Config.hp_conf["H_DIMS"] = 64
        Config.hp_conf["NUM_ENCODER"] = 4
        Config.hp_conf["BATCH_SIZE"] = 128
        Config.TOTAL_EXE = 20000
        Config.TOTAL_CAT = 350
        Config.DATA_FILE_PATH = "../data/pre/" + Config.DATASET
        Config.TEST_FILE_PATH = "../data/pre/" + Config.DATASET
        Config.LAG_S = 5001
        Config.LAG_M = 501
        Config.LAG_D = 51
        Config.EPOCH_NUM = 15

    Config.hp_conf["H_DIMS"] = params.hdim
    Config.hp_conf["EMBED_DIMS"] = params.dim
    # Config.hp_conf["ENC_HEADS"] = params.heads
    Config.hp_conf["MA_DROP_OUT"] = params.adp
    Config.hp_conf["FFN_DROP_OUT"] = params.fdp
    Config.MAX_SEQ = params.seq
    Config.MODEL_NAME = params.model
    Config.AT = params.at
    Config.samp = params.samp
    Config.theta = params.theta
    Config.temp = params.temp
    Config.epsilon = params.epsilon
    Config.NEG = params.neg
    Config.LOSS = params.loss
    Config.AutoInt = not params.SENet
    Config.hp_conf["LAYER_NUM"] = params.ln
    Config.RoPE = not params.worope
    Config.EXEB = not params.wob
    Config.DEVICE_NUM.append(params.gpu)

log = None
if __name__ == "__main__":
    get_param()
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
    test_loader = get_test_dataloaders(bs=config["BATCH_SIZE"])
    model = Encoder_IRT_Modle(config)

    # model = nn.parallel.DistributedDataParallel(model,
    #                                             device_ids=Config.DEVICE_NUM,
    #                                             broadcast_buffers=False,
    #                                             find_unused_parameters=True)
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

    log_path = trainer.log_dir
    # log_path = "/home/b418/kr/ACAKT/lightning_logs/version_53"
    if not os.path.exists(os.path.join(log_path)):
        os.mkdir(log_path)
    log_file = os.path.join(log_path+'/{}_test_result.txt'.format(Config.DATASET))
    log = open(log_file, 'w')
    print('\n'.join(['%s:%s' % item for item in Config.__dict__.items()]),file=log)

    # train
    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=[val_loader, ])

    # test
    # model = load_checkpoint(model, "/home/b418/kr/ACAKT/lightning_logs/version_357/checkpoints/ACAKT_epoch=23-step=3264-val_acc=0.77.ckpt")
    # trainer.test(model, dataloaders=test_loader)
    # model = load_checkpoint(model, "/home/b418/kr/ACAKT/lightning_logs/version_357/checkpoints/ACAKT_epoch=25-step=3536-val_auc=0.80.ckpt")
    # trainer.test(model, dataloaders=test_loader)
    # model = load_checkpoint(model, "/home/b418/kr/ACAKT/lightning_logs/version_357/checkpoints/ACAKT_epoch=27-step=3808.ckpt")
    # trainer.test(model, dataloaders=test_loader)
    model_test = Encoder_IRT_Modle(config)
    path0 = trainer.checkpoint_callbacks[0].best_model_path  # empty here !
    print("path0: ", path0)
    print("path0: ", path0,file=log)
    model0 = load_checkpoint(model_test,path0)
    ans = trainer.test(model0, dataloaders=test_loader, ckpt_path=path0)
    print("auc: ", ans[0]['test_auc'],"acc: ", ans[0]['test_acc'], file=log)
    path1 = trainer.checkpoint_callbacks[1].best_model_path  # empty here !
    print("path1: ", path1)
    print("path1: ", path1, file=log)
    model1 = load_checkpoint(model_test,path1)
    ans = trainer.test(model1, dataloaders=test_loader, ckpt_path=path1)
    print("auc: ", ans[0]['test_auc'], "acc: ", ans[0]['test_acc'], file=log)
    path2 = trainer.checkpoint_callbacks[-1].best_model_path  # empty here !
    print("path2: ", path2)
    print("path2: ", path2, file=log)
    model2 = load_checkpoint(model_test,path2)
    ans = trainer.test(model2, dataloaders=test_loader, ckpt_path=path2)
    print("auc: ", ans[0]['test_auc'], "acc: ", ans[0]['test_acc'], file=log)

    print("TOTAL TIME: ", time() - start_time)
    print("TOTAL TIME: ", time() - start_time,file=log)