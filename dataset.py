import os

from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from config import Config
from sklearn.model_selection import train_test_split
import gc


class DKTDataset(Dataset):
    def __init__(self, samples, max_seq, min_seq, start_token=0):
        super().__init__()
        self.samples = samples
        self.max_seq = max_seq
        self.min_seq = min_seq
        # start_token
        self.start_token = start_token
        self.data = []
        for id in self.samples.index:
            exe_ids, answers, ela_time, categories, cat_num, popularity, difficulty, lag_time_s, lag_time_m, lag_time_d = self.samples[id]
            if len(exe_ids) > max_seq:
                for l in range((len(exe_ids)+max_seq-1)//max_seq):
                    self.data.append(
                        (exe_ids[l:l+max_seq], answers[l:l+max_seq], ela_time[l:l+max_seq], np.stack(categories[l:l+max_seq]).astype(int),
                         cat_num[l:l+max_seq], popularity[l:l+max_seq], difficulty[l:l+max_seq],
                         lag_time_s[l:l+max_seq], lag_time_m[l:l+max_seq], lag_time_d[l:l+max_seq]))
            elif len(exe_ids) < self.max_seq and len(exe_ids) > self.min_seq:
                self.data.append((exe_ids, answers, ela_time, np.stack(categories).astype(int), cat_num, popularity, difficulty, lag_time_s, lag_time_m, lag_time_d))
            else:
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question_ids, answers, ela_time, exe_category, category_num, popularity, difficulty, lag_time_s, lag_time_m, lag_time_d = self.data[idx]
        seq_len = len(question_ids)

        exe_ids = np.zeros(self.max_seq, dtype=int)
        ans = np.zeros(self.max_seq, dtype=int)
        elapsed_time = np.zeros(self.max_seq, dtype=int)
        exe_cat = np.zeros((self.max_seq, Config.MAX_CATS_PER), dtype=list)
        cat_num = np.zeros(self.max_seq, dtype=int)
        pop = np.zeros(self.max_seq, dtype=float)
        diff = np.zeros(self.max_seq, dtype=float)
        lt_s = np.zeros(self.max_seq, dtype=int)
        lt_m = np.zeros(self.max_seq, dtype=int)
        lt_d = np.zeros(self.max_seq, dtype=int)
        if seq_len < self.max_seq:
            exe_ids[-seq_len:] = question_ids
            ans[-seq_len:] = answers
            elapsed_time[-seq_len:] = ela_time
            exe_cat[-seq_len:, :] = exe_category
            cat_num[-seq_len:] = category_num
            pop[-seq_len:] = popularity
            diff[-seq_len:] = difficulty
            lt_s[-seq_len:] = lag_time_s
            lt_m[-seq_len:] = lag_time_m
            lt_d[-seq_len:] = lag_time_d
        else:
            exe_ids[:] = question_ids[-self.max_seq:]
            ans[:] = answers[-self.max_seq:]
            elapsed_time[:] = ela_time[-self.max_seq:]
            exe_cat[:, :] = exe_category[-self.max_seq:]
            cat_num[-seq_len:] = category_num[-self.max_seq:]
            pop[-seq_len:] = popularity[-self.max_seq:]
            diff[-seq_len:] = difficulty[-self.max_seq:]
            lt_s[-seq_len:] = lag_time_s[-self.max_seq:]
            lt_m[-seq_len:] = lag_time_m[-self.max_seq:]
            lt_d[-seq_len:] = lag_time_d[-self.max_seq:]

        input_rtime = np.zeros(self.max_seq, dtype=int)
        input_rtime = np.insert(elapsed_time, 0, 0)
        input_rtime = np.delete(input_rtime, -1)
        # #
        # input_stime = np.zeros(self.max_seq, dtype=int)
        # input_stime = np.insert(lt_s, 0, 0)
        # input_stime = np.delete(input_stime, -1)
        # input_mtime = np.zeros(self.max_seq, dtype=int)
        # input_mtime = np.insert(lt_m, 0, 0)
        # input_mtime = np.delete(input_mtime, -1)
        # input_dtime = np.zeros(self.max_seq, dtype=int)
        # input_dtime = np.insert(lt_d, 0, 0)
        # input_dtime = np.delete(input_dtime, -1)


        input = {"input_ids": exe_ids, "input_rtime": input_rtime.astype(np.int),
                 "input_cat": np.stack(exe_cat).astype(int), "input_cnum": cat_num,
                 "input_pop": pop, "input_diff": diff,
                 "input_lag_time_s": lt_s, "input_lag_time_m": lt_m, "input_lag_time_d": lt_d}
        answers = np.append([0],ans[:-1]) #start token
        # assert ans.shape[0]==answers.shape[0] and answers.shape[0]==input_rtime.shape[0], "both ans and label should be same len with start-token"
        assert ans.shape[0]==answers.shape[0], "both ans and label should be same len with start-token"
        return input,answers,ans
        # return input, ans


def get_dataloaders(bs=16):
    if Config.VISUAL:
        # train = pd.read_pickle("../"+f"{Config.DATA_FILE_PATH}.train")
        # val = pd.read_pickle("../"+f"{Config.DATA_FILE_PATH}.valid")
        train = pd.read_pickle(f"{Config.DATA_FILE_PATH}.train")
        val = pd.read_pickle(f"{Config.DATA_FILE_PATH}.valid")
    else:
        train = pd.read_pickle(f"{Config.DATA_FILE_PATH}.train")
        val = pd.read_pickle(f"{Config.DATA_FILE_PATH}.valid")
    train_dataset = DKTDataset(train, max_seq=Config.MAX_SEQ, min_seq=Config.MIN_SEQ)
    val_dataset = DKTDataset(val, max_seq=Config.MAX_SEQ, min_seq=Config.MIN_SEQ)
    train_loader = DataLoader(train_dataset,
                              batch_size=bs,
                              num_workers=8,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=bs,
                            num_workers=8,
                            shuffle=False)
    del train_dataset, val_dataset
    gc.collect()
    return train_loader, val_loader

def get_test_dataloaders(bs=16):
    if Config.VISUAL:
        # test = pd.read_pickle("../"+f"{Config.TEST_FILE_PATH}.test")
        test = pd.read_pickle(f"{Config.TEST_FILE_PATH}.test")
    else:
        test = pd.read_pickle(f"{Config.TEST_FILE_PATH}.test")
    test_dataset = DKTDataset(test, max_seq=Config.MAX_SEQ, min_seq=Config.MIN_SEQ)
    test_loader = DataLoader(test_dataset,
                              batch_size=bs,
                              num_workers=8,
                              shuffle=False)
    del test_dataset
    gc.collect()
    return test_loader

def get_total_dataloaders(bs=16):
    if Config.VISUAL:
        # test = pd.read_pickle("../"+f"{Config.TEST_FILE_PATH}.test")
        test = pd.read_pickle(f"{Config.TEST_FILE_PATH}.test")
        train = pd.read_pickle(f"{Config.DATA_FILE_PATH}.train")
        val = pd.read_pickle(f"{Config.DATA_FILE_PATH}.valid")
    else:
        test = pd.read_pickle(f"{Config.TEST_FILE_PATH}.test")
        train = pd.read_pickle(f"{Config.DATA_FILE_PATH}.train")
        val = pd.read_pickle(f"{Config.DATA_FILE_PATH}.valid")
    total =  pd.concat([train,val])
    total = pd.concat([total, test])
    total_dataset = DKTDataset(total, max_seq=Config.MAX_SEQ, min_seq=Config.MIN_SEQ)
    total_loader = DataLoader(total_dataset,
                             batch_size=bs,
                             num_workers=8,
                             shuffle=False)
    del total_dataset
    gc.collect()
    return total_loader