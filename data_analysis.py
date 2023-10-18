import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from config import Config
from sklearn.model_selection import train_test_split
import gc
from time import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from config import Config
from sklearn.model_selection import train_test_split
import gc

from dataset import DKTDataset

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
        answers = np.append([0],ans[:-1]) #start_token
        # answers = ans #wo start_token
        # assert ans.shape[0]==answers.shape[0] and answers.shape[0]==input_rtime.shape[0], "both ans and label should be same len with start-token"
        assert ans.shape[0]==answers.shape[0], "both ans and label should be same len with start-token"
        return input,answers,ans
        # return input, ans

def get_dataloaders(DATASET,DATA_TYPE="train",NEW=True):
    if DATASET == "AS09":
        DATA_FILE_PATH = "../data/pre/AS09_lag"
        TOTAL_EXE = 207349
        TOTAL_CAT = 401
    elif DATASET == "EdNet":
        DATA_FILE_PATH = "../data/pre/EdNet_lag6"
        TOTAL_EXE = 20000
        TOTAL_CAT = 350
    DATA = pd.read_pickle(f"{DATA_FILE_PATH}."+DATA_TYPE)
    dataset = DKTDataset(DATA, max_seq=Config.MAX_SEQ, min_seq=Config.MIN_SEQ)
    EXE = [0 for i in range(TOTAL_EXE+1)]
    CAT = [0 for i in range(TOTAL_CAT+1)]
    for records in tqdm(dataset):
        # records[0]: input
        # records[1]: answers-padding
        # records[2]: ans
        data = records[0]
        e_seq = data["input_ids"]
        cat_seq = data["input_cat"]
        seq_len = len(data["input_ids"])
        for item_index in range(seq_len):
            eid = e_seq[item_index]
            if EXE[eid] != 1:
                EXE[eid] = 1
            for cat in cat_seq[item_index]:
                cid = cat
                if CAT[cid] != 1:
                    CAT[cid] = 1

    if NEW:
        new_exe = []
        new_cat = []
        for i, element in enumerate(EXE):
            if element == 0:
                new_exe.append(i)
        for i, element in enumerate(CAT):
            if element == 0:
                new_cat.append(i)
        print("new_exe_num:",len(new_exe))
        print("new_cat_num",len(new_cat))

        f1=open("new_exe_"+DATASET+"_"+DATA_TYPE+".txt","w")
        for line in new_exe:
            f1.write(str(line)+'\n')
        f1.close()
        f2=open("new_cat_"+DATASET+"_"+DATA_TYPE+".txt","w")
        for line in new_cat:
            f2.write(str(line)+'\n')
        f2.close()
    else:
        involve_exe = []
        involve_cat = []
        for i, element in enumerate(EXE):
            if element == 1:
                involve_exe.append(i)
        for i, element in enumerate(CAT):
            if element == 1:
                involve_cat.append(i)
        print("involve_exe_num:",len(involve_exe))
        print("involve_cat_num",len(involve_cat))

        f1=open("involve_exe_"+DATASET+"_"+DATA_TYPE+".txt","w")
        for line in involve_exe:
            f1.write(str(line)+'\n')
        f1.close()
        f2=open("involve_cat_"+DATASET+"_"+DATA_TYPE+".txt","w")
        for line in involve_cat:
            f2.write(str(line)+'\n')
        f2.close()


def get_new_exe_data(datafile="new_exe_AS09_train.txt"):
    new_exe=[]
    with open(datafile,'r') as f:
        for line in f:
            new_exe.append(int(line.strip('\n')))
    return new_exe



if __name__ == "__main__":
    start_time = time()
    # DATASET = "EdNet"
    # DATATYPE = "train"
    # NEW = False
    # get_dataloaders(DATASET,DATATYPE,NEW)

    get_new_exe_data()

    print("TOTAL TIME: ", time() - start_time)


